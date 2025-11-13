"""
Sepsis Phenotyping Pipeline - Yin et al. (2020) Implementation
Modular pipeline for DTW-based weighted k-means clustering
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from scipy.stats import f_oneway
import duckdb
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class WeightedKMeans:
    """
    Weighted K-Means from Yin et al. (2020) Equation 18
    Optimized weighted k-means with caching
    """
    
    def __init__(self, n_clusters=4, max_iter=100, random_state=42, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.tol = tol  # Convergence tolerance
        self.labels_ = None
        self.weights_ = None
        self.inertia_ = None
        
    def _compute_weights_vectorized(self, distance_matrix, labels):
        """
        Vectorized weight computation

       Latency issue fix 13112025:
        - removed the computation of the symmetrical distance matrix again
        - so only computing the upper triangle without the double calcualtion of the lower one as well
        - adding caching as well
        
        """
        n_samples = len(labels)
        weights = np.ones(n_samples, dtype=np.float32)
        
        for k in range(self.n_clusters):
            cluster_mask = (labels == k)
            cluster_size = cluster_mask.sum()
            
            if cluster_size > 1:
                # Extract cluster submatrix once
                cluster_indices = np.where(cluster_mask)[0]
                cluster_dist = distance_matrix[cluster_indices][:, cluster_indices]
                
                # Vectorized mean distance (exclude self-distance)
                avg_dists = (cluster_dist.sum(axis=1) - np.diag(cluster_dist)) / (cluster_size - 1)
                
                # Apply sigmoid with scaled distances
                weights[cluster_indices] = 1.0 / (1.0 + np.exp(avg_dists / cluster_size))
        
        return weights
    
    def _compute_cluster_distances_cached(self, distance_matrix, labels, weights):
        """Compute distances to clusters with weight caching"""
        n_samples = distance_matrix.shape[0]
        cluster_distances = np.full((n_samples, self.n_clusters), np.inf, dtype=np.float32)
        
        for k in range(self.n_clusters):
            cluster_mask = (labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Vectorized weighted distance computation
            cluster_weights = weights[cluster_indices]
            weight_sum = cluster_weights.sum()
            
            if weight_sum > 0:
                # Broadcast multiplication: (n_samples, n_cluster) * (n_cluster,)
                weighted_dists = distance_matrix[:, cluster_indices] @ cluster_weights
                cluster_distances[:, k] = weighted_dists / weight_sum
        
        return cluster_distances
    
    def fit_predict(self, distance_matrix):
        np.random.seed(self.random_state)
        n_samples = distance_matrix.shape[0]
        
        # K-means++ initialization on distance matrix
        labels = self._kmeans_plusplus_init(distance_matrix)
        
        prev_inertia = np.inf
        
        for iteration in range(self.max_iter):
            # Update weights
            self.weights_ = self._compute_weights_vectorized(distance_matrix, labels)
            
            # Compute cluster distances
            cluster_distances = self._compute_cluster_distances_cached(
                distance_matrix, labels, self.weights_
            )
            
            # Assign to nearest cluster
            new_labels = np.argmin(cluster_distances, axis=1)
            
            # Check convergence
            if np.array_equal(labels, new_labels):
                print(f"  Converged at iteration {iteration+1}")
                break
            
            # Compute inertia for convergence check
            current_inertia = np.sum([cluster_distances[i, new_labels[i]] 
                                     for i in range(n_samples)])
            
            if abs(prev_inertia - current_inertia) < self.tol:
                print(f"  Converged (inertia change < {self.tol}) at iteration {iteration+1}")
                break
            
            labels = new_labels
            prev_inertia = current_inertia
        
        self.labels_ = labels
        self.inertia_ = prev_inertia
        
        return labels
    
    def _kmeans_plusplus_init(self, distance_matrix):
        """Fast k-means++ initialization"""
        n_samples = distance_matrix.shape[0]
        centers_idx = [np.random.randint(n_samples)]
        
        for _ in range(self.n_clusters - 1):
            dists = distance_matrix[:, centers_idx].min(axis=1)
            probs = dists / dists.sum()
            next_center = np.random.choice(n_samples, p=probs)
            centers_idx.append(next_center)
        
        # Initial assignment
        return np.argmin(distance_matrix[:, centers_idx], axis=1)




def load_and_prepare_data(db_path, table_name, time_window_hours, feature_columns, subsample_fraction=None, random_state=42):
    """Load data and filter to patients with complete time window"""
    con = duckdb.connect(db_path, read_only=True)
    
    # Get patients with full time window
    valid_patients_df = con.sql(f"""
        SELECT stay_id 
        FROM (
            SELECT stay_id, COUNT(DISTINCT hour) as n_hours
            FROM {table_name}
            GROUP BY stay_id
        )
        WHERE n_hours >= {time_window_hours}
    """).fetchdf()
    
    # Subsample if requested
    if subsample_fraction is not None and 0 < subsample_fraction < 1:
        n_original = len(valid_patients_df)
        valid_patients_df = valid_patients_df.sample(frac=subsample_fraction, random_state=random_state)
        print(f"  Subsampled: {len(valid_patients_df)} / {n_original} patients ({subsample_fraction*100:.1f}%)")
    
    valid_patients = valid_patients_df['stay_id'].tolist()
    
    # Load data
    df = con.sql(f"""
        SELECT * FROM {table_name}
        WHERE stay_id IN ({','.join(map(str, valid_patients))})
        AND hour < {time_window_hours}
        ORDER BY stay_id, hour
    """).fetchdf()
    con.close()
    
    return df, valid_patients


def prepare_timeseries_array(df, feature_columns, max_hours):
    """Convert to 3D array [n_patients, n_timesteps, n_features]"""
    stay_ids = df['stay_id'].unique()
    n_patients = len(stay_ids)
    n_features = len(feature_columns)
    
    X = np.full((n_patients, max_hours, n_features), np.nan)
    
    for i, stay_id in enumerate(stay_ids):
        patient_data = df[df['stay_id'] == stay_id].sort_values('hour')
        for _, row in patient_data.iterrows():
            hour = int(row['hour'])
            if hour < max_hours:
                X[i, hour, :] = row[feature_columns].values
    
    # Fill missing
    for i in range(n_patients):
        X[i] = pd.DataFrame(X[i]).ffill().bfill().values
    X = np.nan_to_num(X, nan=0.0)
    
    return X, stay_ids


def compute_dtw_matrix(X, chunk_size=500, use_cache=True):
    """
    Compute pairwise DTW distance matrix with edge case handling especially for 1 patient
    
    """
    n_patients = X.shape[0]
    distance_matrix = np.zeros((n_patients, n_patients), dtype=np.float32)
    
    print(f"Computing DTW distance matrix for {n_patients} patients...")
    
    # Edge case: insufficient patients for clustering
    if n_patients < 2:
        print(f"Warning: Only {n_patients} patient(s) - cannot compute meaningful distances")
        return distance_matrix
    
    # Compute only upper triangle
    n_chunks = (n_patients + chunk_size - 1) // chunk_size
    total_chunks = (n_chunks * (n_chunks + 1)) // 2
    
    with tqdm(total=total_chunks, desc="Computing DTW", unit="chunk") as pbar:
        for i in range(n_chunks):
            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, n_patients)
            
            for j in range(i, n_chunks):
                start_j = j * chunk_size
                end_j = min((j + 1) * chunk_size, n_patients)
                
                n_available = max(1, os.cpu_count() - 2)
                # n_cores = min(8, n_available)
                n_cores=1
                print(f"Using {n_cores} cores for computation.")
                chunk_distances = cdist_dtw(
                    X[start_i:end_i], 
                    X[start_j:end_j], 
                    n_jobs=n_cores,
                    verbose=0
                )
                
                distance_matrix[start_i:end_i, start_j:end_j] = chunk_distances
                
                if i != j:
                    distance_matrix[start_j:end_j, start_i:end_i] = chunk_distances.T
                
                pbar.update(1)
    
    # Safe statistics
    non_diag_mask = ~np.eye(n_patients, dtype=bool)
    non_diag_dists = distance_matrix[non_diag_mask]
    
    if len(non_diag_dists) > 0 and non_diag_dists.max() > 0:
        print(f"DTW matrix: shape={distance_matrix.shape}, "
              f"range=[{non_diag_dists[non_diag_dists>0].min():.3f}, {non_diag_dists.max():.3f}]")
    else:
        print(f"DTW matrix: shape={distance_matrix.shape} (warning: all zeros)")
    
    return distance_matrix


def evaluate_k_range(distance_matrix, k_range, n_mds_components=10):
    """
    Evaluate clustering for multiple k value
    Fast evaluation using reduced MDS dimensionality
    """
    print("\n[5/6] Evaluating k values...")
    
    # Compute MDS once for all k values (reduced dimensions for speed)
    n_available = max(1, os.cpu_count() - 2)
    # n_cores = min(8, n_available)
    n_cores=1
    print(f"  Computing MDS with {n_mds_components} components...")
    mds = MDS(n_components=n_mds_components, dissimilarity='precomputed', 
              random_state=42, n_jobs=n_cores, max_iter=100)  # Reduced iterations
    X_euclidean = mds.fit_transform(distance_matrix)
    
    results = []
    for k in tqdm(k_range, desc="  Testing k values", unit="k"):
        wkmeans = WeightedKMeans(n_clusters=k, max_iter=50, random_state=42)  # Reduced iterations
        labels = wkmeans.fit_predict(distance_matrix)
        
        # Fast silhouette with sampling for large datasets
        if len(labels) > 5000:
            sil = silhouette_score(distance_matrix, labels, metric='precomputed', sample_size=5000)
        else:
            sil = silhouette_score(distance_matrix, labels, metric='precomputed')
        
        dbi = davies_bouldin_score(X_euclidean, labels)
        chi = calinski_harabasz_score(X_euclidean, labels)
        
        unique, counts = np.unique(labels, return_counts=True)
        
        results.append({
            'k': k,
            'silhouette': sil,
            'davies_bouldin': dbi,
            'calinski_harabasz': chi,
            'inertia': wkmeans.inertia_,
            'cluster_sizes': dict(zip(unique.tolist(), counts.tolist()))
        })
    
    return pd.DataFrame(results)


def plot_validation_metrics(results_df, output_dir):
    """Plot clustering validation metrics"""
    output_dir = Path(output_dir)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Clustering Validation Metrics', fontsize=16, fontweight='bold')
    
    k_vals = results_df['k'].values
    
    # Silhouette
    ax = axes[0, 0]
    ax.plot(k_vals, results_df['silhouette'], 'o-', linewidth=2, markersize=8)
    best_k = k_vals[results_df['silhouette'].argmax()]
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    ax.set_xlabel('k'); ax.set_ylabel('Silhouette')
    ax.set_title('Silhouette (Higher Better)'); ax.grid(True, alpha=0.3); ax.legend()
    
    # Davies-Bouldin
    ax = axes[0, 1]
    ax.plot(k_vals, results_df['davies_bouldin'], 'o-', linewidth=2, markersize=8, color='orange')
    best_k = k_vals[results_df['davies_bouldin'].argmin()]
    ax.axvline(best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    ax.set_xlabel('k'); ax.set_ylabel('Davies-Bouldin')
    ax.set_title('Davies-Bouldin (Lower Better)'); ax.grid(True, alpha=0.3); ax.legend()
    
    # Calinski-Harabasz
    ax = axes[1, 0]
    ax.plot(k_vals, results_df['calinski_harabasz'], 'o-', linewidth=2, markersize=8, color='green')
    ax.set_xlabel('k'); ax.set_ylabel('Calinski-Harabasz')
    ax.set_title('Calinski-Harabasz (Higher Better)'); ax.grid(True, alpha=0.3)
    
    # Inertia
    ax = axes[1, 1]
    ax.plot(k_vals, results_df['inertia'], 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia')
    ax.set_title('Inertia (Elbow)'); ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'validation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_dimensionality_reduction(distance_matrix, labels, output_dir, method='both'):
    """Plot t-SNE and/or PCA visualization of clusters"""
    output_dir = Path(output_dir)
    n_clusters = len(np.unique(labels))
    
    if method in ['tsne', 'both']:
        print("  Computing t-SNE...")
        tsne = TSNE(n_components=2, metric='precomputed', init='random', random_state=42, n_jobs=1)
        X_tsne = tsne.fit_transform(distance_matrix)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, 
                            cmap='tab10', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f't-SNE Visualization ({n_clusters} clusters)', fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.tight_layout()
        plt.savefig(output_dir / 'tsne_clusters.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    if method in ['pca', 'both']:
        print("  Computing PCA...")
        mds = MDS(n_components=50, dissimilarity='precomputed', random_state=42, n_jobs=1)
        X_mds = mds.fit_transform(distance_matrix)
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_mds)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                            cmap='tab10', s=50, alpha=0.6)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'PCA Visualization ({n_clusters} clusters)', fontsize=14, fontweight='bold')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_clusters.png', dpi=150, bbox_inches='tight')
        plt.close()


def plot_cluster_sizes(labels, output_dir):
    """Bar plot of cluster sizes"""
    output_dir = Path(output_dir)
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(unique, counts, color='steelblue', alpha=0.7)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(labels)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Patients', fontsize=12)
    plt.title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    plt.xticks(unique)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()


def analyze_clinical_outcomes(db_path, table_name, stay_ids, labels, output_dir):
    """Analyze and plot clinical outcomes by cluster"""
    output_dir = Path(output_dir)
    
    # Load SOFA
    con = duckdb.connect(db_path, read_only=True)
    outcome_df = con.sql(f"""
        WITH BaselineSOFA AS (
            SELECT stay_id, sofa_total
            FROM {table_name}
            WHERE hour = 0
        )
        SELECT stay_id, sofa_total
        FROM BaselineSOFA
    """).fetchdf()
    con.close()
    
    # Merge
    results_df = pd.DataFrame({'stay_id': stay_ids, 'cluster': labels})
    results_df = results_df.merge(outcome_df, on='stay_id', how='left')
    
    # ANOVA
    cluster_groups = [results_df[results_df['cluster'] == k]['sofa_total'].dropna() 
                      for k in sorted(results_df['cluster'].unique())]
    f_stat, p_value = f_oneway(*cluster_groups)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Boxplot
    ax = axes[0]
    results_df.boxplot(column='sofa_total', by='cluster', ax=ax)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Baseline SOFA Score', fontsize=12)
    ax.set_title(f'SOFA by Cluster (ANOVA p={p_value:.2e})', fontsize=12)
    plt.sca(ax)
    plt.xticks(rotation=0)
    
    # Mean ± SEM
    ax = axes[1]
    cluster_stats = results_df.groupby('cluster')['sofa_total'].agg(['mean', 'sem'])
    ax.bar(cluster_stats.index, cluster_stats['mean'], 
           yerr=cluster_stats['sem'], capsize=5, alpha=0.7)
    ax.set_xlabel('Cluster', fontsize=12)
    ax.set_ylabel('Mean SOFA Score', fontsize=12)
    ax.set_title('Mean SOFA ± SEM by Cluster', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'clinical_outcomes.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save stats
    stats_df = results_df.groupby('cluster')['sofa_total'].describe()
    stats_df.to_csv(output_dir / 'cluster_clinical_stats.csv')
    
    print(f"\nClinical Analysis:")
    print(f"  ANOVA: F={f_stat:.2f}, p={p_value:.2e}")
    
    return results_df


def run_kmeans_dtw_pipeline(
    db_path,
    table_name="clipped_brits_saits",
    time_window_hours=24,
    output_dir="output",
    manual_k=None,
    k_range=(2, 8),
    feature_columns=None,
    dtw_chunk_size=500,
    subsample_fraction=None,
    random_state=42
):
    """
    Main pipeline for sepsis phenotyping using DTW and weighted k-means
    
    Parameters:
    -----------
    db_path : str
        Path to DuckDB database
    table_name : str
        Table name in database
    time_window_hours : int
        Time window for analysis (24, 48, or 72)
    output_dir : str
        Output directory for results and plots
    manual_k : int or None
        If specified, skip optimal k search and use this value
    k_range : tuple
        Range of k values to test (min, max)
    feature_columns : list
        List of feature column names
    dtw_chunk_size : int
        Chunk size for DTW computation
    subsample_fraction : float or None
        Fraction of patients to subsample (e.g., 0.01 for 1%)
        Applied BEFORE DTW computation for speed
    random_state : int
        Random seed
        
    Returns:
    --------
    dict with keys:
        - model_name: str
        - optimal_k: int
        - labels: np.ndarray
        - stay_ids: list
        - cluster_sizes: dict
        - metrics: dict (silhouette, davies_bouldin, calinski_harabasz)
        - plots: dict (paths to generated plots)
        - results: pd.DataFrame (cluster assignments with weights)
    """
    
    # Setup
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    if feature_columns is None:
        feature_columns = [
            'age_at_admission', 'Arterial O2 pressure (PaO2)', 'Bicarbonate',
            'Chloride', 'Creatinine', 'Diastolic Blood Pressure', 'Glucose',
            'Heart Rate', 'INR', 'Inspired O2 Fraction (FiO2)', 'Lactate',
            'Mean Blood Pressure', 'O2 saturation pulseoxymetry (SpO2)',
            'Platelet Count', 'Respiratory Rate', 'Sodium',
            'Systolic Blood Pressure', 'Temperature_C', 'Urine_Output_Hourly'
        ]
    
    print("="*80)
    print("SEPSIS PHENOTYPING PIPELINE - YIN ET AL. (2020)")
    print("="*80)
    print(f"Time window: {time_window_hours}h")
    print(f"Output dir: {output_dir}")
    print(f"Manual k: {manual_k if manual_k else 'Auto-select'}")
    print(f"Subsample: {subsample_fraction*100 if subsample_fraction else 'None'}%")
    
    # Load data with optional subsampling
    print("\n[1/6] Loading data...")
    df, valid_patients = load_and_prepare_data(
        db_path, table_name, time_window_hours, feature_columns, 
        subsample_fraction=subsample_fraction, random_state=random_state
    )
    print(f"  Using {len(valid_patients)} patients with {time_window_hours}h data")
    
    # Prepare array
    print("\n[2/6] Preparing time series array...")
    X, stay_ids = prepare_timeseries_array(df, feature_columns, time_window_hours)
    
    # Normalize
    print("\n[3/6] Normalizing...")
    scaler = TimeSeriesScalerMeanVariance()
    X_norm = scaler.fit_transform(X)
    
    # Compute DTW
    # Compute DTW - use optimized function
    subsample_suffix = f"_subsample{subsample_fraction}" if subsample_fraction else ""
    dtw_path = output_dir / f'dtw_matrix_{time_window_hours}h{subsample_suffix}.npy'

    if dtw_path.exists():
        print(f"\n[4/6] Loading existing DTW matrix from {dtw_path}...")
        distance_matrix = np.load(dtw_path)
    else:
        print(f"\n[4/6] Computing DTW distance matrix (optimized)...")
        distance_matrix = compute_dtw_matrix(X_norm, chunk_size=dtw_chunk_size, use_cache=True)
        np.save(dtw_path, distance_matrix)
        print(f"  Saved to {dtw_path}")

    # Evaluate k range - use optimized function
    k_evaluation_df = None
    if manual_k is None:
        print(f"\n[5/6] Evaluating k={k_range[0]} to {k_range[1]-1} (fast mode)...")
        k_evaluation_df = evaluate_k_range(distance_matrix, range(*k_range))
        k_evaluation_df.to_csv(output_dir / 'k_evaluation.csv', index=False)
        
        plot_validation_metrics(k_evaluation_df, output_dir)
        
        optimal_k = int(k_evaluation_df.loc[k_evaluation_df['silhouette'].idxmax(), 'k'])
        print(f"  Optimal k: {optimal_k} (silhouette: {k_evaluation_df['silhouette'].max():.4f})")
    else:
        optimal_k = manual_k
        print(f"\n[5/6] Using manual k={optimal_k}")

    # Final clustering
    print(f"\n[6/6] Final clustering with k={optimal_k}...")
    wkmeans = WeightedKMeans(n_clusters=optimal_k, max_iter=100, random_state=random_state)
    labels = wkmeans.fit_predict(distance_matrix)
        
    # Cluster sizes
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    print(f"\nCluster sizes: {cluster_sizes}")
    
    # Compute final metrics

    mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42, n_jobs=1)
    X_euclidean = mds.fit_transform(distance_matrix)
    
    final_metrics = {
        'silhouette': float(silhouette_score(distance_matrix, labels, metric='precomputed')),
        'davies_bouldin': float(davies_bouldin_score(X_euclidean, labels)),
        'calinski_harabasz': float(calinski_harabasz_score(X_euclidean, labels)),
        'inertia': float(wkmeans.inertia_)
    }
    
    # Save cluster assignments
    assignments_df = pd.DataFrame({
        'stay_id': stay_ids,
        'cluster': labels,
        'weight': wkmeans.weights_
    })
    assignments_df.to_csv(output_dir / 'cluster_assignments.csv', index=False)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_cluster_sizes(labels, output_dir)
    plot_dimensionality_reduction(distance_matrix, labels, output_dir, method='both')
    clinical_df = analyze_clinical_outcomes(db_path, table_name, stay_ids, labels, output_dir)
    
    # Collect plot paths
    plot_paths = {
        'validation_metrics': str(output_dir / 'validation_metrics.png') if manual_k is None else None,
        'cluster_sizes': str(output_dir / 'cluster_sizes.png'),
        'tsne': str(output_dir / 'tsne_clusters.png'),
        'pca': str(output_dir / 'pca_clusters.png'),
        'clinical_outcomes': str(output_dir / 'clinical_outcomes.png')
    }
    
    print(f"\n{'='*80}")
    print(f"COMPLETE! Results saved to {output_dir}")
    print(f"{'='*80}")
    
    # Return results following kmeans_pipeline.py structure
    return {
        'model_name': f'DTW_WeightedKMeans_{time_window_hours}h_k{optimal_k}',
        'optimal_k': optimal_k,
        'labels': labels,
        'stay_ids': stay_ids,
        'cluster_sizes': cluster_sizes,
        'metrics': final_metrics,
        'plots': plot_paths,
        'results': assignments_df,
        'k_evaluation': k_evaluation_df,
        'distance_matrix_path': str(dtw_path)
    }


if __name__ == "__main__":
    results = run_kmeans_dtw_pipeline(
        db_path="fixed_hour_length_issue_BRITSSAITS_10112025.duckdb", # using the fixed dataset
        time_window_hours=24, # by default the time frame will be 24, but can change
        output_dir="output_24h_full", # results
        manual_k=None,
        k_range=(2, 8),
        subsample_fraction=0.01  # by default using 0.01 for demo only 
    )
    
    # results_demo = run_sepsis_phenotyping_pipeline(
    #     db_path="fixed_hour_length_issue_BRITSSAITS_10112025.duckdb",
    #     time_window_hours=24,
    #     output_dir="output_24h_demo",
    #     manual_k=3,
    #     subsample_fraction=0.01  # 1% of patients for fast demo
    # )