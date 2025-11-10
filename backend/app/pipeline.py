"""
Consensus K-Means Clustering Pipeline
Based on Seymour et al. methodology for phenotype discovery
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class ConsensusKMeans:
    """
    Consensus K-Means clustering implementation
    """
    
    def __init__(self, k_range=range(2, 7), n_iterations=100, 
                 subsample_fraction=0.8, random_state=42):
        """
        Initialize consensus clustering parameters.
        
        Args:
            k_range: Range of cluster numbers to test
            n_iterations: Number of subsampling iterations
            subsample_fraction: Fraction of data to subsample each iteration
            random_state: Random seed for reproducibility
        """
        self.k_range = k_range
        self.n_iterations = n_iterations
        self.subsample_fraction = subsample_fraction
        self.random_state = random_state
        self.consensus_matrices = {}
        self.consensus_scores = {}
        self.pairwise_consensus = {}
        self.optimal_k = None
        self.labels = None
        
    def preprocess_data(self, df, exclude_cols, correlation_threshold=0.8,
                       log_transform=False, log_transform_skew_threshold=1.0):
        """
        Preprocess clinical data following Seymour et al. methodology.
        
        Args:
            df: Input DataFrame
            exclude_cols: Columns to exclude from clustering
            correlation_threshold: Threshold for removing correlated variables
            log_transform: Whether to apply log transformation
            log_transform_skew_threshold: Skewness threshold for log transform
            
        Returns:
            X_processed: Standardized feature matrix
            df_transformed: Transformed DataFrame with excluded columns
            feature_names: List of final feature names
        """
        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE")
        print("="*80)
        
        # Separate excluded columns
        df_transformed = df.drop(columns=list(exclude_cols), errors='ignore').copy()
        df_excluded = df[exclude_cols].copy() if exclude_cols else None
        
        # Optional: Log transformation
        if log_transform:
            print("\nApplying log transformation to skewed variables...")
            numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns
            transformed_vars = []
            
            for col in numeric_cols:
                skewness = stats.skew(df_transformed[col].dropna())
                
                if abs(skewness) > log_transform_skew_threshold:
                    min_val = df_transformed[col].min()
                    
                    if min_val <= 0:
                        df_transformed[col] = np.log1p(df_transformed[col] - min_val + 1)
                    else:
                        df_transformed[col] = np.log(df_transformed[col])
                    
                    transformed_vars.append(col)
            
            print(f"  Log-transformed {len(transformed_vars)} variables")
        
        # Remove highly correlated variables
        print(f"\nRemoving highly correlated variables (r > {correlation_threshold})...")
        corr_matrix = df_transformed.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = set()
        for column in upper_tri.columns:
            high_corr = upper_tri[column][upper_tri[column] > correlation_threshold]
            if len(high_corr) > 0:
                mean_corr_col = corr_matrix[column].mean()
                for corr_var in high_corr.index:
                    mean_corr_var = corr_matrix[corr_var].mean()
                    if mean_corr_col > mean_corr_var:
                        to_drop.add(column)
                    else:
                        to_drop.add(corr_var)
        
        df_transformed = df_transformed.drop(columns=list(to_drop))
        print(f"  Removed {len(to_drop)} correlated variables")
        print(f"  Final number of variables: {len(df_transformed.columns)}")
        
        # Standardize
        print("\nStandardizing features...")
        scaler = StandardScaler()
        X_processed = scaler.fit_transform(df_transformed).astype(np.float32)
        
        # Combine back with excluded columns
        if df_excluded is not None:
            df_transformed = pd.concat([df_transformed, df_excluded], axis=1)
        
        feature_names = list(df_transformed.columns)
        
        print("\n" + "="*80)
        print(f"PREPROCESSING COMPLETE: {X_processed.shape[1]} features, {X_processed.shape[0]} samples")
        print("="*80)
        
        return X_processed, df_transformed, feature_names
    
    def run_consensus_clustering(self, X):
        """
        Perform consensus k-means clustering using vectorized operations.
        
        Args:
            X: Preprocessed feature matrix
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        subsample_size = int(n_samples * self.subsample_fraction)
        
        print("\n" + "="*80)
        print("CONSENSUS CLUSTERING")
        print("="*80)
        
        for k in self.k_range:
            print(f"\nProcessing k={k}...")
            consensus_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
            indicator_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
            
            for iter_idx in tqdm(range(self.n_iterations), desc=f"k={k}", ncols=80):
                # Random subsample
                sample_indices = np.random.choice(n_samples, size=subsample_size, 
                                                 replace=False)
                X_subsample = X[sample_indices]
                
                # K-means clustering
                kmeans = KMeans(n_clusters=k, n_init=10, random_state=iter_idx)
                labels_subsample = kmeans.fit_predict(X_subsample)
                
                # Vectorized co-clustering update
                same_cluster = (labels_subsample[:, None] == 
                              labels_subsample[None, :]).astype(np.float32)
                consensus_matrix[np.ix_(sample_indices, sample_indices)] += same_cluster
                indicator_matrix[np.ix_(sample_indices, sample_indices)] += 1
            
            # Normalize to get consensus probabilities
            with np.errstate(divide='ignore', invalid='ignore'):
                consensus_matrix = np.divide(
                    consensus_matrix,
                    indicator_matrix,
                    out=np.zeros_like(consensus_matrix),
                    where=indicator_matrix != 0
                )
            
            self.consensus_matrices[k] = consensus_matrix
    
    def get_cluster_labels(self, consensus_matrix, k):
        """
        Get final cluster assignments using hierarchical clustering.
        
        Args:
            consensus_matrix: Consensus matrix for given k
            k: Number of clusters
            
        Returns:
            labels: Cluster assignments
        """
        distance_matrix = 1 - consensus_matrix
        condensed_dist = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        linkage_matrix = linkage(condensed_dist, method='average')
        labels = fcluster(linkage_matrix, k, criterion='maxclust') - 1
        return labels
    
    def compute_consensus_metrics(self):
        """
        Compute consensus score (AUC) and pairwise consensus for each k.
        """
        print("\n" + "="*80)
        print("COMPUTING CONSENSUS METRICS")
        print("="*80)
        
        for k, consensus_matrix in self.consensus_matrices.items():
            # Consensus score (area under CDF)
            upper_triangle = consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)]
            hist, _ = np.histogram(upper_triangle, bins=100, range=(0, 1))
            cdf = np.cumsum(hist) / np.sum(hist)
            auc = np.trapz(cdf, dx=1/100)
            self.consensus_scores[k] = auc
            
            # Pairwise consensus (within-cluster)
            labels = self.get_cluster_labels(consensus_matrix, k)
            within_cluster_consensus = []
            
            for cluster_id in range(k):
                cluster_mask = labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) > 1:
                    cluster_consensus = consensus_matrix[np.ix_(cluster_indices, 
                                                                cluster_indices)]
                    upper_tri = cluster_consensus[np.triu_indices_from(cluster_consensus, k=1)]
                    within_cluster_consensus.append(np.mean(upper_tri))
                else:
                    within_cluster_consensus.append(1.0)
            
            self.pairwise_consensus[k] = {
                'mean': np.mean(within_cluster_consensus),
                'min': np.min(within_cluster_consensus),
                'per_cluster': within_cluster_consensus
            }
    
    def print_selection_criteria(self):
        """
        Print comprehensive selection criteria for optimal k.
        """
        print("\n" + "="*80)
        print("PHENOTYPE SELECTION CRITERIA")
        print("="*80)
        
        header = f"\n{'k':<5} {'Size Range':<20} {'Consensus Score':<18} {'Min Pairwise':<15} {'Mean Pairwise':<15} {'Meets Criterion'}"
        print(header)
        print("-" * 100)
        
        for k in sorted(self.consensus_matrices.keys()):
            labels = self.get_cluster_labels(self.consensus_matrices[k], k)
            sizes = [np.sum(labels == i) for i in range(k)]
            size_range = f"{min(sizes)}-{max(sizes)}"
            
            consensus_score = self.consensus_scores[k]
            min_pairwise = self.pairwise_consensus[k]['min']
            mean_pairwise = self.pairwise_consensus[k]['mean']
            
            criterion_met = "✓" if min_pairwise > 0.8 else "✗"
            
            print(f"{k:<5} {size_range:<20} {consensus_score:<18.4f} "
                  f"{min_pairwise:<15.4f} {mean_pairwise:<15.4f} {criterion_met}")
        
        print("\nSelection Guidelines (from Seymour et al.):")
        print("1. Pairwise consensus >0.8 (KEY CRITERION)")
        print("2. Clear separation in consensus matrix heatmap")
        print("3. Adequate phenotype sizes (balanced distribution)")
        print("4. Steep CDF curves (values near 0 or 1)")
    
    def select_optimal_k(self, method='auto', manual_k=None):
        """
        Select optimal k based on consensus metrics.
        
        Args:
            method: 'auto' (use pairwise consensus) or 'manual'
            manual_k: Manually specified k value
            
        Returns:
            optimal_k: Selected number of clusters
        """
        if method == 'manual' and manual_k is not None:
            self.optimal_k = manual_k
            print(f"\nManually selected k = {self.optimal_k}")
        else:
            # Auto-select based on pairwise consensus > 0.8
            valid_k = [k for k, metrics in self.pairwise_consensus.items() 
                      if metrics['min'] > 0.8]
            
            if valid_k:
                self.optimal_k = min(valid_k)  # Choose smallest k meeting criterion
                print(f"\nAuto-selected k = {self.optimal_k} (min pairwise consensus > 0.8)")
            else:
                # Fallback to highest pairwise consensus
                self.optimal_k = max(self.pairwise_consensus.items(), 
                                   key=lambda x: x[1]['min'])[0]
                print(f"\nNo k met criterion, selected k = {self.optimal_k} "
                      f"(highest min pairwise consensus)")
        
        # Get final labels
        self.labels = self.get_cluster_labels(
            self.consensus_matrices[self.optimal_k], 
            self.optimal_k
        )
        
        return self.optimal_k
    
    def plot_consensus_matrices(self, figsize=(20, 5), save_path=None):
        """
        Plot consensus matrices for all k values with cluster color bars.
        """
        from matplotlib.colors import ListedColormap
        
        k_values = sorted(self.consensus_matrices.keys())
        n_k = len(k_values)
        
        cluster_colors = ['#B3E5FC', '#7CB342', '#5C6BC0', '#FFB74D', '#EC407A', 
                         '#26A69A', '#AB47BC', '#FFA726', '#66BB6A']
        
        fig, axes = plt.subplots(1, n_k, figsize=figsize)
        if n_k == 1:
            axes = [axes]
        
        for idx, k in enumerate(k_values):
            labels = self.get_cluster_labels(self.consensus_matrices[k], k)
            sorted_idx = np.argsort(labels)
            sorted_matrix = self.consensus_matrices[k][np.ix_(sorted_idx, sorted_idx)]
            sorted_labels = labels[sorted_idx]
            
            im = axes[idx].imshow(sorted_matrix, cmap='Blues', vmin=0, vmax=1, 
                                 aspect='auto')
            
            n_samples = len(sorted_labels)
            
            # Top color bar
            for i, label in enumerate(sorted_labels):
                axes[idx].add_patch(plt.Rectangle(
                    (i-0.5, -0.05*n_samples), 1, 0.05*n_samples,
                    facecolor=cluster_colors[label], 
                    edgecolor='none', clip_on=False
                ))
            
            # Right color bar
            for i, label in enumerate(sorted_labels):
                axes[idx].add_patch(plt.Rectangle(
                    (n_samples, i-0.5), 0.05*n_samples, 1,
                    facecolor=cluster_colors[label], 
                    edgecolor='none', clip_on=False
                ))
            
            axes[idx].set_title(f'Consensus matrix k = {k}', 
                              fontsize=12, fontweight='bold', pad=20)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            
            for spine in axes[idx].spines.values():
                spine.set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_cdf_curves(self, figsize=(10, 6), max_points=10000, save_path=None):
        """
        Plot consensus CDF curves for all k values.
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        for k in sorted(self.consensus_matrices.keys()):
            consensus_matrix = self.consensus_matrices[k]
            upper_triangle = consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)]
            
            sorted_values = np.sort(upper_triangle)
            
            # Downsample if too many points
            if len(sorted_values) > max_points:
                indices = np.linspace(0, len(sorted_values)-1, max_points, dtype=int)
                sorted_values = sorted_values[indices]
            
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            
            criterion_met = self.pairwise_consensus[k]['min'] > 0.8
            linestyle = '-' if criterion_met else '--'
            ax.plot(sorted_values, cdf, label=f'k={k}', linewidth=2, 
                   linestyle=linestyle)
        
        ax.set_xlabel('Consensus Index', fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title('Consensus CDF Curves (solid = pairwise >0.8)', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_selection_metrics(self, figsize=(12, 5), save_path=None):
        """
        Plot consensus scores and pairwise consensus metrics.
        """
        k_values = sorted(self.consensus_matrices.keys())
        scores = [self.consensus_scores[k] for k in k_values]
        min_pairwise = [self.pairwise_consensus[k]['min'] for k in k_values]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Consensus scores
        ax1.plot(k_values, scores, 'o-', linewidth=2, markersize=10, color='blue')
        ax1.set_xlabel('Number of Phenotypes (k)', fontsize=12)
        ax1.set_ylabel('Consensus Score (AUC)', fontsize=12)
        ax1.set_title('Consensus Score vs k', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Pairwise consensus with threshold
        ax2.plot(k_values, min_pairwise, 's-', linewidth=2, markersize=10, 
                color='green')
        ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, 
                   label='Threshold (0.8)', alpha=0.7)
        ax2.set_xlabel('Number of Phenotypes (k)', fontsize=12)
        ax2.set_ylabel('Min Pairwise Consensus', fontsize=12)
        ax2.set_title('Pairwise Consensus vs k', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_delta_area(self, figsize=(8, 6), save_path=None):
        """
        Plot relative change in area under CDF curve with increasing k.
        """
        k_values = sorted(self.consensus_scores.keys())
        scores = [self.consensus_scores[k] for k in k_values]
        
        delta_scores = []
        delta_k = []
        for i in range(1, len(scores)):
            delta = scores[i] - scores[i-1]
            delta_scores.append(delta)
            delta_k.append(k_values[i])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        ax.plot(delta_k, delta_scores, marker='o', color='steelblue', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        for k, delta in zip(delta_k, delta_scores):
            ax.scatter(k, delta, color='coral' if delta < 0 else 'steelblue', 
                      s=80, edgecolors='black', zorder=3)
        
        ax.set_xlabel('Number of Clusters (k)', fontsize=12)
        ax.set_ylabel('Δ Area Under CDF Curve', fontsize=12)
        ax.set_title('Relative Change in Consensus with Increasing k', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(k_values)
        ax.grid(True, alpha=0.3, axis='y')
        
        ax.text(0.98, 0.02, 'Little change → optimal k reached', 
                transform=ax.transAxes, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_cluster_consensus(self, figsize=(10, 6), save_path=None):
        """
        Plot mean pairwise consensus for each cluster.
        """
        k_values = sorted(self.consensus_matrices.keys())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        bar_width = 0.8 / len(k_values)
        x = np.arange(max([k for k in k_values]))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(k_values)))
        
        for idx, k in enumerate(k_values):
            per_cluster_consensus = self.pairwise_consensus[k]['per_cluster']
            x_pos = np.arange(k) + idx * bar_width
            
            bars = ax.bar(x_pos, per_cluster_consensus, bar_width, 
                         label=f'k={k}', color=colors[idx], alpha=0.8, 
                         edgecolor='black')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, 
                  label='Threshold (0.8)', alpha=0.7)
        
        ax.set_xlabel('Cluster ID', fontsize=12)
        ax.set_ylabel('Mean Pairwise Consensus', fontsize=12)
        ax.set_title('Cluster Consensus: Mean of Pairwise Consensus Within Each Cluster', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x + bar_width * (len(k_values)-1) / 2)
        ax.set_xticklabels([f'{i}' for i in range(len(x))])
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_tsne(self, X, perplexity=30, figsize=(10, 8), max_samples=10000, 
                  save_path=None):
        """
        Create t-SNE visualization of phenotypes.
        """
        from matplotlib.colors import ListedColormap
        
        print(f"\nComputing t-SNE for k={self.optimal_k}...")
        
        # Sample data if too large
        if X.shape[0] > max_samples:
            print(f"  Sampling {max_samples} points for visualization...")
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[sample_idx]
            labels_sample = self.labels[sample_idx]
        else:
            X_sample = X
            labels_sample = self.labels
        
        tsne = TSNE(n_components=2, perplexity=perplexity, 
                   random_state=self.random_state)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Create discrete colormap
        tab10_colors = plt.cm.tab10.colors
        colors = [tab10_colors[i] for i in range(self.optimal_k)]
        cmap = ListedColormap(colors)
        
        fig, ax = plt.subplots(figsize=figsize)
        scatter = ax.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=labels_sample, 
            cmap=cmap,
            alpha=0.6, 
            s=20,
            vmin=-0.5,
            vmax=self.optimal_k-0.5
        )
        
        ax.set_xlabel('t-SNE 1', fontsize=12)
        ax.set_ylabel('t-SNE 2', fontsize=12)
        ax.set_title(f't-SNE Visualization (k={self.optimal_k})', 
                    fontsize=14, fontweight='bold')
        
        cbar = plt.colorbar(scatter, ax=ax, ticks=range(self.optimal_k), 
                           boundaries=np.arange(-0.5, self.optimal_k, 1))
        cbar.set_label('Phenotype', fontsize=12)
        cbar.ax.set_yticklabels([f'Phenotype {i}' for i in range(self.optimal_k)])
        
        # Add cluster sizes
        unique, counts = np.unique(labels_sample, return_counts=True)
        size_text = "\n".join([f"Phenotype {i}: n={c:,}" 
                              for i, c in zip(unique, counts)])
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
        
        Args:
            X: Preprocessed feature matrix
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("\n" + "="*80)
        print("GENERATING DIAGNOSTIC PLOTS")
        print("="*80)
        
        self.plot_consensus_matrices(
            save_path=output_path / 'consensus_matrices.png'
        )
        plt.close()
        
        self.plot_cdf_curves(
            save_path=output_path / 'consensus_cdf.png'
        )
        plt.close()
        
        self.plot_selection_metrics(
            save_path=output_path / 'selection_metrics.png'
        )
        plt.close()
        
        self.plot_delta_area(
            save_path=output_path / 'delta_area.png'
        )
        plt.close()
        
        self.plot_cluster_consensus(
            save_path=output_path / 'cluster_consensus.png'
        )
        plt.close()
        
        self.plot_tsne(
            X,
            save_path=output_path / 'tsne_phenotypes.png'
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

    conn = duckdb.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = conn.execute(query).fetchdf()
    conn.close()
    return df

def run_consensus_pipeline(df, exclude_cols, k_range=range(2, 7), 
                           n_iterations=100, subsample_fraction=0.8,
                           correlation_threshold=0.8, log_transform=False,
                           subsample_data=None, output_dir='results',
                           manual_k=None, random_state=42, ):
    """
    Main function to run the complete consensus k-means pipeline.
    
    Args:
        df: Input DataFrame with clinical variables
        exclude_cols: List of columns to exclude from clustering
        k_range: Range of k values to test
        n_iterations: Number of consensus iterations
        subsample_fraction: Fraction to subsample in each iteration
        correlation_threshold: Threshold for removing correlated variables
        log_transform: Whether to apply log transformation
        subsample_data: Fraction of data to use (None = use all data)
        output_dir: Directory to save results
        manual_k: Manually specify optimal k (None = auto-select)
        random_state: Random seed
        
    Returns:
        dict: Dictionary containing:
            - model: Fitted ConsensusKMeans object
            - df_with_phenotypes: DataFrame with phenotype assignments
            - optimal_k: Selected optimal k
            - labels: Cluster assignments
            - X_processed: Preprocessed feature matrix
            - feature_names: List of feature names used
    """
    print("\n" + "="*80)
    print("CONSENSUS K-MEANS CLUSTERING PIPELINE")
    print("="*80)
    print(f"folder path / output dir: {output_dir}")
    print(f"Input data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Subsample data if requested
    if subsample_data is not None:
        print(f"\nSubsampling {subsample_data*100:.1f}% of data...")
        df = df.sample(frac=subsample_data, random_state=random_state)
        print(f"Subsampled data: {df.shape[0]:,} rows")
    
    # Initialize model
    model = ConsensusKMeans(
        k_range=k_range,
        n_iterations=n_iterations,
        subsample_fraction=subsample_fraction,
        random_state=random_state
    )
    
    # Preprocess data
    X_processed, df_transformed, feature_names = model.preprocess_data(
        df=df,
        exclude_cols=exclude_cols,
        correlation_threshold=correlation_threshold,
        log_transform=log_transform
    )
    
    # Run consensus clustering
    model.run_consensus_clustering(X_processed)
    
    # Compute metrics
    model.compute_consensus_metrics()
    
    # Print selection criteria
    model.print_selection_criteria()
    
    # Select optimal k
    optimal_k = model.select_optimal_k(
        method='manual' if manual_k else 'auto',
        manual_k=manual_k
    )
    
    # Add phenotype labels to DataFrame
    df_with_phenotypes = df_transformed.copy()
    df_with_phenotypes['phenotype'] = model.labels
    
    # Generate all diagnostic plots
    model.generate_all_plots(X_processed, output_dir=output_dir)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PHENOTYPE SUMMARY")
    print("="*80)
    print(f"\nOptimal k: {optimal_k}")
    print(f"\nPhenotype sizes:")
    print(df_with_phenotypes['phenotype'].value_counts().sort_index())
    
    # Return results
    results = {
        'model': model,
        'df_with_phenotypes': df_with_phenotypes,
        'optimal_k': optimal_k,
        'labels': model.labels,
        'X_processed': X_processed,
        'feature_names': [f for f in feature_names if f not in exclude_cols],
        'consensus_matrices': model.consensus_matrices,
        'consensus_scores': model.consensus_scores,
        'pairwise_consensus': model.pairwise_consensus
    }
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}/")
    print("\nReturned objects:")
    print("  - model: ConsensusKMeans object")
    print("  - df_with_phenotypes: DataFrame with 'phenotype' column")
    print("  - optimal_k: Selected number of clusters")
    print("  - labels: Cluster assignments")
    print("  - X_processed: Preprocessed feature matrix")
    print("  - feature_names: List of features used")
    print("  - consensus_matrices: Dict of consensus matrices by k")
    print("  - consensus_scores: Dict of consensus scores by k")
    print("  - pairwise_consensus: Dict of pairwise consensus metrics by k")
    
    return results



# Example usage
if __name__ == "__main__":
    """
    Example of how to use the pipeline
    """
    
    # Example 1: Basic usage with DuckDB
    # import duckdb
    df = data_loader_duckdb('../../data/final_imputed_output_26102025.duckdb', 'final_imputed_output_26102025')

    # Example 2: Using the pipeline
    results = run_consensus_pipeline(
        df=df,
        exclude_cols=['stay_id', 'hour', 'model', 'all_icd_codes', 
                      'all_icd_versions', 'is_sepsis', 'sofa_total'],
        # folder_path='your_folder_path',
        k_range=range(2, 7),
        n_iterations=100,
        correlation_threshold=0.8,
        log_transform=False,
        subsample_fraction=0.8,
        subsample_data=0.02,  # Use 2% of data
        output_dir='testing',
        manual_k=None,
        random_state=42
    )
    
    # Example 3: Access results
    model = results['model']
    df_phenotypes = results['df_with_phenotypes']
    optimal_k = results['optimal_k']
    labels = results['labels']
    
    # Example 4: Generate additional analysis
    feature_names = results['feature_names']
    phenotype_profiles = df_phenotypes.groupby('phenotype')[feature_names].mean()
    print(phenotype_profiles)
    pass