# -*- coding: utf-8 -*-
"""
Group-Based Trajectory Modeling (GBTM) Pipeline
Using Gaussian Mixture Models with 6-metric composite scoring
Based on TAME notebook methodology for phenotype discovery

This implementation matches the notebooks:
- consensus_clustering_results_updated.duckdb (≤24h) → k=4
- fixed_hour_length_issue.duckdb (≤24h) → k=5
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.impute import SimpleImputer
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class GroupBasedTrajectoryModel:
    """
    Group-Based Trajectory Modeling using Gaussian Mixture Models
    with 6-metric composite scoring:
    - BIC (lower is better)
    - AIC (lower is better)
    - Inertia (lower is better)
    - Silhouette (higher is better)
    - Davies-Bouldin (lower is better)
    - Calinski-Harabasz (higher is better)
    """
    
    def __init__(self, k_range=range(2, 7), n_init=10, max_iter=200, random_state=42):
        """
        Initialize GBTM parameters.
        
        Args:
            k_range: Range of cluster numbers to test
            n_init: Number of random initializations for GMM
            max_iter: Maximum iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.k_range = k_range
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.models = []
        self.bic_scores = {}
        self.aic_scores = {}
        self.inertia_scores = {}
        self.silhouette_scores = {}
        self.davies_bouldin_scores = {}
        self.calinski_harabasz_scores = {}
        self.composite_scores = {}
        self.optimal_k = None
        self.optimal_model = None
        self.labels = None
    
    def preprocess_data(self, df, exclude_cols):
        """
        Preprocess clinical data for GBTM.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from clustering
            
        Returns:
            X_scaled: Standardized feature matrix
            trajectory_cols: List of feature names used
        """
        print("\n" + "="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        # Select numeric trajectory features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        trajectory_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        print(f"\n✓ Selected {len(trajectory_cols)} trajectory features")
        
        # Handle missing values
        print("\nHandling missing values...")
        X_trajectory = df[trajectory_cols].copy()
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_trajectory),
            columns=trajectory_cols,
            index=df.index
        )
        print(f"✓ Missing values imputed")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed).astype(np.float32)
        print(f"✓ Features standardized")
        print(f"  Shape: {X_scaled.shape}")
        
        print("\n" + "="*80)
        print(f"PREPROCESSING COMPLETE: {X_scaled.shape[1]} features, {X_scaled.shape[0]} samples")
        print("="*80)
        
        return X_scaled, trajectory_cols
    
    def train_models(self, X):
        """
        Train Gaussian Mixture Models for all k values.
        
        Args:
            X: Preprocessed feature matrix
        """
        print("\n" + "="*80)
        print("TRAINING GBTM MODELS")
        print("="*80)
        print(f"\nDataset: {X.shape[0]:,} samples × {X.shape[1]} features")
        print(f"Training {len(self.k_range)} models (k={min(self.k_range)}-{max(self.k_range)})\n")
        
        self.models = []
        
        with tqdm(total=len(self.k_range), desc="Training Models", 
                  unit="model", colour='green', ncols=100) as pbar:
            
            for k in self.k_range:
                pbar.set_description(f"Training k={k}")
                
                # Fit Gaussian Mixture Model
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type='full',
                    random_state=self.random_state,
                    n_init=self.n_init,
                    max_iter=self.max_iter,
                    verbose=0
                )
                gmm.fit(X)
                
                self.models.append(gmm)
                pbar.update(1)
        
        print(f"\n✓ All models trained")
    
    def compute_metrics(self, X):
        """
        Compute all 6 metrics for each k value.
        
        Args:
            X: Preprocessed feature matrix
        """
        print("\n" + "="*80)
        print("COMPUTING MODEL SELECTION METRICS")
        print("="*80)
        print("Using 6 metrics: BIC, AIC, Inertia, Silhouette, Davies-Bouldin, Calinski-Harabasz")
        print("="*80 + "\n")
        
        with tqdm(total=len(self.models), desc="Calculating Metrics", 
                  unit="model", colour='cyan', ncols=100) as pbar:
            
            for idx, model in enumerate(self.models):
                k = self.k_range[idx]
                pbar.set_description(f"Metrics k={k}")
                
                # Get predictions
                labels = model.predict(X)
                
                # Metric 1: BIC (lower is better)
                bic = model.bic(X)
                self.bic_scores[k] = bic
                
                # Metric 2: AIC (lower is better)
                aic = model.aic(X)
                self.aic_scores[k] = aic
                
                # Metric 3: Inertia (lower is better)
                # Calculate weighted distances to cluster centers
                weighted_distances = -model.score_samples(X)
                inertia = np.sum(weighted_distances)
                self.inertia_scores[k] = inertia
                
                # Metric 4: Silhouette (higher is better)
                if k > 1:
                    try:
                        sil = silhouette_score(X, labels)
                    except:
                        sil = -1
                else:
                    sil = -1
                self.silhouette_scores[k] = sil
                
                # Metric 5: Davies-Bouldin (lower is better)
                if k > 1:
                    try:
                        db = davies_bouldin_score(X, labels)
                    except:
                        db = np.inf
                else:
                    db = np.inf
                self.davies_bouldin_scores[k] = db
                
                # Metric 6: Calinski-Harabasz (higher is better)
                if k > 1:
                    try:
                        ch = calinski_harabasz_score(X, labels)
                    except:
                        ch = 0
                else:
                    ch = 0
                self.calinski_harabasz_scores[k] = ch
                
                print(f"  k={k}: BIC={bic:,.0f}, Silhouette={sil:.3f}")
                pbar.update(1)
        
        print(f"\n✓ All metrics computed")
    
    def compute_composite_scores(self):
        """
        Compute composite scores using MinMaxScaler normalization.
        Average of all 6 normalized metrics (lower composite = better).
        """
        print("\n" + "="*80)
        print("COMPUTING COMPOSITE SCORES")
        print("="*80)
        
        k_values = sorted(self.bic_scores.keys())
        
        # Extract values
        bic_values = np.array([self.bic_scores[k] for k in k_values]).reshape(-1, 1)
        aic_values = np.array([self.aic_scores[k] for k in k_values]).reshape(-1, 1)
        inertia_values = np.array([self.inertia_scores[k] for k in k_values]).reshape(-1, 1)
        sil_values = np.array([self.silhouette_scores[k] for k in k_values]).reshape(-1, 1)
        db_values = np.array([self.davies_bouldin_scores[k] for k in k_values]).reshape(-1, 1)
        ch_values = np.array([self.calinski_harabasz_scores[k] for k in k_values]).reshape(-1, 1)
        
        # Normalize using MinMaxScaler (0=best, 1=worst)
        scaler = MinMaxScaler()
        
        # For metrics where LOWER is better (BIC, AIC, Inertia, Davies-Bouldin)
        bic_norm = scaler.fit_transform(bic_values).flatten()
        aic_norm = scaler.fit_transform(aic_values).flatten()
        inertia_norm = scaler.fit_transform(inertia_values).flatten()
        db_norm = scaler.fit_transform(db_values).flatten()
        
        # For metrics where HIGHER is better (Silhouette, Calinski-Harabasz) - invert
        sil_norm = 1 - scaler.fit_transform(sil_values).flatten()
        ch_norm = 1 - scaler.fit_transform(ch_values).flatten()
        
        # Composite score (average of normalized metrics, lower = better)
        for idx, k in enumerate(k_values):
            composite = (bic_norm[idx] + aic_norm[idx] + inertia_norm[idx] + 
                        sil_norm[idx] + db_norm[idx] + ch_norm[idx]) / 6
            self.composite_scores[k] = composite
        
        print("\nComposite Scores (lower = better):")
        for k in k_values:
            print(f"  k={k}: {self.composite_scores[k]:.4f}")
        
        print(f"\n✓ Composite scores computed")
    
    def print_selection_criteria(self):
        """
        Print comprehensive selection criteria showing all metrics.
        """
        print("\n" + "="*80)
        print("MODEL SELECTION CRITERIA")
        print("="*80)
        
        k_values = sorted(self.composite_scores.keys())
        
        # Create rankings
        bic_ranks = {k: rank+1 for rank, k in enumerate(sorted(k_values, key=lambda x: self.bic_scores[x]))}
        sil_ranks = {k: rank+1 for rank, k in enumerate(sorted(k_values, key=lambda x: self.silhouette_scores[x], reverse=True))}
        comp_ranks = {k: rank+1 for rank, k in enumerate(sorted(k_values, key=lambda x: self.composite_scores[x]))}
        
        header = f"\n{'k':<5} {'BIC':<12} {'Silhouette':<15} {'Composite':<12} {'Rank'}"
        print(header)
        print("-" * 60)
        
        # Sort by composite score
        sorted_k = sorted(k_values, key=lambda x: self.composite_scores[x])
        
        for rank, k in enumerate(sorted_k, 1):
            bic = self.bic_scores[k]
            sil = self.silhouette_scores[k]
            comp = self.composite_scores[k]
            
            marker = " ← OPTIMAL" if rank == 1 else ""
            print(f"{k:<5} {bic:<12.1f} {sil:<15.4f} {comp:<12.4f} {rank}{marker}")
        
        print("\nDetailed Rankings:")
        print(f"{'k':<5} {'BIC_rank':<12} {'Sil_rank':<12} {'Comp_rank'}")
        print("-" * 45)
        for k in sorted_k:
            print(f"{k:<5} {bic_ranks[k]:<12} {sil_ranks[k]:<12} {comp_ranks[k]}")
    
    def select_optimal_k(self, X):
        """
        Select optimal k based on lowest composite score.
        
        Args:
            X: Feature matrix (needed for predictions)
            
        Returns:
            optimal_k: Selected number of clusters
        """
        print("\n" + "="*80)
        print("OPTIMAL MODEL SELECTION")
        print("="*80)
        
        # Select k with lowest composite score
        self.optimal_k = min(self.composite_scores.items(), key=lambda x: x[1])[0]
        
        # Get corresponding model
        optimal_idx = list(self.k_range).index(self.optimal_k)
        self.optimal_model = self.models[optimal_idx]
        
        # Get final labels
        self.labels = self.optimal_model.predict(X)
        
        print(f"\n✓ OPTIMAL NUMBER OF CLASSES: {self.optimal_k}")
        print(f"  Composite Score: {self.composite_scores[self.optimal_k]:.4f}")
        print(f"\nDetailed Metrics:")
        print(f"  BIC: {self.bic_scores[self.optimal_k]:,.1f} (rank: {sorted(self.bic_scores.items(), key=lambda x: x[1]).index((self.optimal_k, self.bic_scores[self.optimal_k])) + 1})")
        print(f"  Silhouette: {self.silhouette_scores[self.optimal_k]:.4f} (rank: {sorted(self.silhouette_scores.items(), key=lambda x: x[1], reverse=True).index((self.optimal_k, self.silhouette_scores[self.optimal_k])) + 1})")
        
        return self.optimal_k
    
    def plot_selection_metrics(self, figsize=(18, 10), save_path=None):
        """
        Plot all 6 metrics plus composite score.
        """
        k_values = sorted(self.bic_scores.keys())
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()
        
        metric_configs = [
            ('BIC', [self.bic_scores[k] for k in k_values], 'steelblue', 'lower'),
            ('AIC', [self.aic_scores[k] for k in k_values], 'orange', 'lower'),
            ('Inertia', [self.inertia_scores[k] for k in k_values], 'brown', 'lower'),
            ('Silhouette', [self.silhouette_scores[k] for k in k_values], 'green', 'higher'),
            ('Davies-Bouldin', [self.davies_bouldin_scores[k] for k in k_values], 'crimson', 'lower'),
            ('Calinski-Harabasz', [self.calinski_harabasz_scores[k] for k in k_values], 'teal', 'higher'),
            ('Composite Score', [self.composite_scores[k] for k in k_values], 'black', 'lower')
        ]
        
        for idx, (name, scores, color, direction) in enumerate(metric_configs):
            ax = axes[idx]
            ax.plot(k_values, scores, 'o-', linewidth=2, markersize=10, color=color)
            ax.axvline(x=self.optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={self.optimal_k}')
            ax.set_xlabel('Number of Classes (k)', fontsize=11)
            ax.set_ylabel(f'{name} ({direction} is better)', fontsize=11)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(len(metric_configs), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_tsne(self, X, perplexity=30, figsize=(10, 8), max_samples=10000, save_path=None):
        """
        Create t-SNE visualization of phenotypes.
        """
        print(f"\nComputing t-SNE for k={self.optimal_k}...")
        
        # Sample data if too large
        if X.shape[0] > max_samples:
            print(f"  Sampling {max_samples} points for visualization...")
            sample_idx = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[sample_idx]
            labels_sample = self.labels[sample_idx]
        else:
            X_sample = X
            labels_sample = self.labels
        
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=self.random_state)
        X_tsne = tsne.fit_transform(X_sample)
        
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=labels_sample,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(f't-SNE Visualization (k={self.optimal_k})', fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Phenotype', fontsize=12)
        
        # Add cluster sizes
        unique, counts = np.unique(labels_sample, return_counts=True)
        size_text = "\n".join([f"Class {i}: n={c:,}" for i, c in zip(unique, counts)])
        ax.text(0.02, 0.98, size_text, transform=ax.transAxes,
               verticalalignment='top',
               bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
               fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def generate_all_plots(self, X, output_dir='results'):
        """
        Generate all diagnostic plots.
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*80)
        print("GENERATING DIAGNOSTIC PLOTS")
        print("="*80)
        
        self.plot_selection_metrics(
            save_path=output_path / 'gbtm_selection_metrics.png'
        )
        plt.close()
        
        self.plot_tsne(
            X,
            save_path=output_path / 'gbtm_tsne_phenotypes.png'
        )
        plt.close()
        
        print(f"\nAll plots saved to: {output_path}")


def data_loader_duckdb(db_path, table_name):
    """
    Load data from DuckDB database into a pandas DataFrame.
    
    Args:
        db_path: Path to DuckDB database file
        table_name: Name of the table to load
    Returns:
        df: Loaded DataFrame
    """
    conn = duckdb.connect(db_path, read_only=True)
    query = f"SELECT * FROM {table_name}"
    df = conn.execute(query).fetchdf()
    conn.close()
    return df


def run_gbtm_pipeline(df, exclude_cols, db_name, k_range=range(2, 7), 
                      n_init=10, max_iter=200, random_state=42,
                      output_dir='backend/output_dir/GBTM'):
    """
    Main function to run the complete GBTM pipeline.
    
    Args:
        df: Input DataFrame with clinical variables
        exclude_cols: List of columns to exclude from clustering
        db_name: Name of database (for labeling output files)
        k_range: Range of k values to test
        n_init: Number of random initializations
        max_iter: Maximum EM iterations
        random_state: Random seed
        output_dir: Directory to save results
        
    Returns:
        dict: Dictionary containing:
            - optimal_k: Selected number of classes
            - labels: Class assignments
            - probabilities: Class membership probabilities
    """
    print("\n" + "="*80)
    print("GROUP-BASED TRAJECTORY MODELING (GBTM) PIPELINE")
    print("="*80)
    print(f"Database: {db_name}")
    print(f"Output directory: {output_dir}")
    print(f"Input data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Initialize model
    model = GroupBasedTrajectoryModel(
        k_range=k_range,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Preprocess data
    X_scaled, trajectory_cols = model.preprocess_data(df=df, exclude_cols=exclude_cols)
    
    # Train models
    model.train_models(X_scaled)
    
    # Compute metrics
    model.compute_metrics(X_scaled)
    
    # Compute composite scores
    model.compute_composite_scores()
    
    # Print selection criteria
    model.print_selection_criteria()
    
    # Select optimal k
    optimal_k = model.select_optimal_k(X_scaled)
    
    # Add phenotype labels to DataFrame
    df_with_phenotypes = df.copy()
    df_with_phenotypes['phenotype'] = model.labels
    
    # Generate all diagnostic plots
    model.generate_all_plots(X_scaled, output_dir=output_dir)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PHENOTYPE SUMMARY")
    print("="*80)
    print(f"\nOptimal k: {optimal_k}")
    print(f"\nClass sizes:")
    class_sizes = df_with_phenotypes['phenotype'].value_counts().sort_index()
    for class_id, count in class_sizes.items():
        pct = count / len(df_with_phenotypes) * 100
        print(f"  Class {class_id}: {count:,} ({pct:.1f}%)")
    
    # Save summary
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    summary_path = output_path / f'gbtm_summary_{db_name}.txt'
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GROUP-BASED TRAJECTORY MODELING SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Database: {db_name}\n")
        f.write(f"Optimal number of classes: {optimal_k}\n")
        f.write(f"Selection method: composite (6 metrics)\n\n")
        f.write(f"Class sizes:\n")
        for class_id, count in class_sizes.items():
            pct = count / len(df_with_phenotypes) * 100
            f.write(f"  Class {class_id}: {count:,} ({pct:.1f}%)\n")
        f.write("\n\nModel metrics:\n\n")
        for k in sorted(model.bic_scores.keys()):
            f.write(f"k={k}:\n")
            f.write(f"  bic: {model.bic_scores[k]:.1f}\n")
            f.write(f"  aic: {model.aic_scores[k]:.1f}\n")
            f.write(f"  silhouette: {model.silhouette_scores[k]:.4f}\n")
            f.write(f"  composite_score: {model.composite_scores[k]:.4f}\n\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    # Get posterior probabilities
    probabilities = model.optimal_model.predict_proba(X_scaled)
    
    # Return results
    results = {
        'optimal_k': optimal_k,
        'labels': model.labels,
        'probabilities': probabilities
    }
    
    print("\n" + "="*80)
    print("GBTM PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nReturned objects:")
    print("  - optimal_k: Selected number of classes")
    print("  - labels: Class assignments")
    print("  - probabilities: Class membership probabilities")
    
    return results


# Example usage
if __name__ == "__main__":
    """
    Example of how to use the GBTM pipeline with both datasets
    """
    
    print("\n" + "="*80)
    print("TESTING GBTM PIPELINE WITH BOTH DATASETS")
    print("="*80)
    
    # ============================================================
    # Dataset 1: fixed_hour_length_issue.duckdb
    # Expected optimal k: 5
    # ============================================================
    print("\n" + "="*80)
    print("DATASET 1: fixed_hour_length_issue.duckdb")
    print("="*80)
    
    df1 = data_loader_duckdb('data/fixed_hour_length_issue.duckdb', 'clipped_brits_saits')
    
    # Filter to ≤24h patients and aggregate
    patient_id_column = 'stay_id'
    patient_hour_counts = df1.groupby(patient_id_column).size().reset_index(name='n_hours')
    patients_24h_or_less = patient_hour_counts[patient_hour_counts['n_hours'] <= 24][patient_id_column]
    df_temp = df1[df1[patient_id_column].isin(patients_24h_or_less)].copy()
    
    # Aggregate to one row per patient
    agg_dict = {}
    for col in df_temp.columns:
        if col == patient_id_column:
            continue
        elif col == 'hour':
            agg_dict[col] = 'max'
        elif col in ['age_at_admission', 'sofa_total', 'phenotype', 'sofa_cat']:
            agg_dict[col] = 'first'
        elif df_temp[col].dtype in ['float64', 'int64']:
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'
    
    df1 = df_temp.groupby(patient_id_column).agg(agg_dict).reset_index()
    
    # Run GBTM pipeline
    results1 = run_gbtm_pipeline(
        df=df1,
        exclude_cols=['stay_id', 'hour', 'sofa_total', 'age_at_admission', 'model', 
                      'all_icd_codes', 'all_icd_versions', 'is_sepsis', 'phenotype', 'sofa_cat'],
        db_name='fixed_hour_length',
        k_range=range(2, 7),
        n_init=10,
        max_iter=200,
        random_state=42,
        output_dir='backend/output_dir/GBTM/fixed_hour_length'
    )
    
    print(f"\n✓ Dataset 1 Result: k={results1['optimal_k']} (Expected: k=5)")
    
    # ============================================================
    # Dataset 2: consensus_clustering_results_updated.duckdb
    # Expected optimal k: 4
    # ============================================================
    print("\n\n" + "="*80)
    print("DATASET 2: consensus_clustering_results_updated.duckdb")
    print("="*80)
    
    df2 = data_loader_duckdb('data/consensus_clustering_results_updated.duckdb', 'df_subset')
    
    # Filter to ≤24h patients and aggregate
    patient_hour_counts = df2.groupby(patient_id_column).size().reset_index(name='n_hours')
    patients_24h_or_less = patient_hour_counts[patient_hour_counts['n_hours'] <= 24][patient_id_column]
    df_temp = df2[df2[patient_id_column].isin(patients_24h_or_less)].copy()
    
    # Aggregate to one row per patient
    agg_dict = {}
    for col in df_temp.columns:
        if col == patient_id_column:
            continue
        elif col == 'hour' or col == 'chart_hour':
            agg_dict[col] = 'max'
        elif col in ['age_at_admission', 'sofa_total', 'phenotype', 'sofa_cat']:
            agg_dict[col] = 'first'
        elif df_temp[col].dtype in ['float64', 'int64']:
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = 'first'
    
    df2 = df_temp.groupby(patient_id_column).agg(agg_dict).reset_index()
    
    # Run GBTM pipeline
    results2 = run_gbtm_pipeline(
        df=df2,
        exclude_cols=['stay_id', 'hour', 'chart_hour', 'sofa_total', 'age_at_admission', 'model',
                      'all_icd_codes', 'all_icd_versions', 'is_sepsis', 'phenotype', 'sofa_cat'],
        db_name='consensus_clustering',
        k_range=range(2, 7),
        n_init=10,
        max_iter=200,
        random_state=42,
        output_dir='backend/output_dir/GBTM/consensus_clustering'
    )
    
    print(f"\n✓ Dataset 2 Result: k={results2['optimal_k']} (Expected: k=4)")
    
    # ============================================================
    # Final Summary
    # ============================================================
    print("\n\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"\nDataset 1 (fixed_hour_length): k={results1['optimal_k']} {'✓ CORRECT' if results1['optimal_k'] == 5 else '✗ INCORRECT (expected k=5)'}")
    print(f"Dataset 2 (consensus_clustering): k={results2['optimal_k']} {'✓ CORRECT' if results2['optimal_k'] == 4 else '✗ INCORRECT (expected k=4)'}")
