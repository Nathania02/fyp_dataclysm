"""
Latent Class Analysis (LCA) Pipeline
Model-based clustering using StepMix with 7-metric composite selection
Based on TAME - less than 24H.ipynb methodology
"""
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from stepmix.stepmix import StepMix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy import stats
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class LatentClassAnalysis:
    """
    Latent Class Analysis using StepMix with 7-metric composite selection
    """
    
    def __init__(self, k_range=range(2, 7), n_init=10, max_iter=200, random_state=42):
        """
        Initialize LCA parameters.
        
        Args:
            k_range: Range of class numbers to test
            n_init: Number of initializations
            max_iter: Maximum iterations for EM algorithm
            random_state: Random seed for reproducibility
        """
        self.k_range = k_range
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.models = {}
        self.metrics = {}
        self.optimal_k = None
        self.optimal_model = None
        self.labels = None
        self.probabilities = None
        self.composite_scores = {}
        
    def preprocess_data(self, df, exclude_cols, correlation_threshold=0.8,
                       log_transform=False, log_transform_skew_threshold=1.0):
        """
        Preprocess clinical data following notebook methodology.
        
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
        
        # Filter exclude_cols to only include columns that exist in the dataframe
        if exclude_cols:
            existing_exclude_cols = [col for col in exclude_cols if col in df.columns]
            missing_cols = [col for col in exclude_cols if col not in df.columns]
            if missing_cols:
                print(f"\nNote: {len(missing_cols)} exclude columns not found in dataset: {missing_cols}")
            exclude_cols = existing_exclude_cols
        
        # Separate excluded columns
        df_transformed = df.drop(columns=list(exclude_cols), errors='ignore').copy()
        df_excluded = df[list(exclude_cols)].copy() if exclude_cols else None
        
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
        
        # Store feature names before combining
        feature_names_only = list(df_transformed.columns)
        
        # Combine back with excluded columns
        if df_excluded is not None:
            df_transformed = pd.concat([df_transformed, df_excluded], axis=1)
        
        print("\n" + "="*80)
        print(f"PREPROCESSING COMPLETE: {X_processed.shape[1]} features, {X_processed.shape[0]} samples")
        print("="*80)
        
        return X_processed, df_transformed, feature_names_only
    
    def fit_models(self, X):
        """
        Fit StepMix LCA models for all k values and calculate 7 metrics.
        
        Args:
            X: Preprocessed feature matrix
        """
        print("\n" + "="*80)
        print("FITTING LCA MODELS WITH STEPMIX")
        print("="*80)
        
        # Initialize metric storage
        log_likelihood_scores = []
        bic_scores = []
        aic_scores = []
        inertia_scores = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        actual_n_classes_list = []
        
        print(f"\nTraining {len(self.k_range)} models...")
        print(f"Dataset: {X.shape[0]:,} samples × {X.shape[1]} features\n")
        
        for k in tqdm(self.k_range, desc="Fitting LCA models", ncols=80):
            # Fit StepMix LCA model
            lca_model = StepMix(
                n_components=k,
                measurement='continuous',
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=0
            )
            lca_model.fit(X)
            self.models[k] = lca_model
            actual_n_classes_list.append(k)
            
            # Get predictions
            labels_full = lca_model.predict(X)
            n_unique_labels = len(np.unique(labels_full))
            
            # ============================================================
            # Metric 1: Log-likelihood (higher is better)
            # ============================================================
            log_lik = lca_model.score(X)
            log_likelihood_scores.append(log_lik)
            
            # ============================================================
            # Metric 2 & 3: BIC and AIC (lower is better)
            # ============================================================
            n_features = X.shape[1]
            n_samples = X.shape[0]
            
            # Parameters: (k-1) class probs + k*n_features means + k*n_features variances
            n_params = (k - 1) + (k * n_features) + (k * n_features)
            
            bic = -2 * log_lik + n_params * np.log(n_samples)
            aic = -2 * log_lik + 2 * n_params
            
            bic_scores.append(bic)
            aic_scores.append(aic)
            
            # ============================================================
            # Metric 4: Inertia (lower is better)
            # ============================================================
            try:
                class_means = []
                for class_id in range(k):
                    class_mask = labels_full == class_id
                    if class_mask.sum() > 0:
                        class_mean = X[class_mask].mean(axis=0)
                        class_means.append(class_mean)
                
                if len(class_means) == k:
                    inertia = 0
                    for i in range(len(X)):
                        assigned_class = labels_full[i]
                        distance = np.sum((X[i] - class_means[assigned_class])**2)
                        inertia += distance
                else:
                    inertia = np.inf
            except:
                inertia = np.inf
            
            inertia_scores.append(inertia)
            
            # ============================================================
            # Metrics 5-7: Silhouette, Davies-Bouldin, Calinski-Harabasz
            # ============================================================
            # Match notebook: use full dataset for small samples (< 50000), 
            # use sampling only for very large datasets
            use_sampling = n_samples > 50000
            
            if n_unique_labels > 1 and n_samples >= k * 2:
                try:
                    if use_sampling:
                        # Sample for large datasets
                        sil = silhouette_score(X, labels_full, sample_size=min(50000, n_samples))
                    else:
                        # Use full dataset for small samples (MATCHES NOTEBOOK)
                        sil = silhouette_score(X, labels_full)
                except:
                    sil = -1
                
                try:
                    db = davies_bouldin_score(X, labels_full)
                except:
                    db = np.inf
                
                try:
                    ch = calinski_harabasz_score(X, labels_full)
                except:
                    ch = 0
            else:
                sil = -1
                db = np.inf
                ch = 0
            
            silhouette_scores.append(sil)
            davies_bouldin_scores.append(db)
            calinski_harabasz_scores.append(ch)
            
            # Store all metrics
            self.metrics[k] = {
                'log_likelihood': log_lik,
                'bic': bic,
                'aic': aic,
                'inertia': inertia,
                'silhouette': sil,
                'davies_bouldin': db,
                'calinski_harabasz': ch
            }
        
        print(f"\n✓ Fitted {len(self.models)} models (k={min(self.k_range)} to {max(self.k_range)})")
        
        # Store for composite scoring
        self.metrics_df = pd.DataFrame({
            'n_classes': actual_n_classes_list,
            'Log_Likelihood': log_likelihood_scores,
            'BIC': bic_scores,
            'AIC': aic_scores,
            'Inertia': inertia_scores,
            'Silhouette': silhouette_scores,
            'Davies_Bouldin': davies_bouldin_scores,
            'Calinski_Harabasz': calinski_harabasz_scores
        })
    
    def select_optimal_k_composite(self):
        """
        Select optimal k using 7-metric composite scoring with MINMAX NORMALIZATION.
        This matches the notebook methodology exactly.
        
        Returns:
            optimal_k: Selected number of classes
        """
        print("\n" + "="*80)
        print("OPTIMAL LCA MODEL SELECTION")
        print("="*80)
        
        # Normalize metrics using MinMaxScaler (0-1 scale, where 0 is best)
        # This matches the notebook methodology exactly
        scaler_metric = MinMaxScaler()
        
        # For metrics where LOWER is better - direct normalization (0 = best)
        self.metrics_df['BIC_norm'] = scaler_metric.fit_transform(self.metrics_df[['BIC']])
        self.metrics_df['AIC_norm'] = scaler_metric.fit_transform(self.metrics_df[['AIC']])
        self.metrics_df['Inertia_norm'] = scaler_metric.fit_transform(self.metrics_df[['Inertia']])
        self.metrics_df['Davies_Bouldin_norm'] = scaler_metric.fit_transform(self.metrics_df[['Davies_Bouldin']])
        
        # For metrics where HIGHER is better - invert (1 - normalized, so 0 = best)
        self.metrics_df['Log_Likelihood_norm'] = 1 - scaler_metric.fit_transform(self.metrics_df[['Log_Likelihood']])
        self.metrics_df['Silhouette_norm'] = 1 - scaler_metric.fit_transform(self.metrics_df[['Silhouette']])
        self.metrics_df['Calinski_Harabasz_norm'] = 1 - scaler_metric.fit_transform(self.metrics_df[['Calinski_Harabasz']])
        
        # Composite score = average of normalized metrics (lower is better, 0 = perfect)
        self.metrics_df['Composite_Score'] = self.metrics_df[[
            'Log_Likelihood_norm', 'BIC_norm', 'AIC_norm', 'Inertia_norm', 
            'Silhouette_norm', 'Davies_Bouldin_norm', 'Calinski_Harabasz_norm'
        ]].mean(axis=1)
        
        # Also create rankings for display purposes
        self.metrics_df['BIC_rank'] = self.metrics_df['BIC'].rank().astype(float)
        self.metrics_df['AIC_rank'] = self.metrics_df['AIC'].rank().astype(float)
        self.metrics_df['Inertia_rank'] = self.metrics_df['Inertia'].rank().astype(float)
        self.metrics_df['Davies_Bouldin_rank'] = self.metrics_df['Davies_Bouldin'].rank().astype(float)
        self.metrics_df['Log_Likelihood_rank'] = self.metrics_df['Log_Likelihood'].rank(ascending=False).astype(float)
        self.metrics_df['Silhouette_rank'] = self.metrics_df['Silhouette'].rank(ascending=False).astype(float)
        self.metrics_df['Calinski_Harabasz_rank'] = self.metrics_df['Calinski_Harabasz'].rank(ascending=False).astype(float)
        
        # Create rankings dataframe for display
        rank_df = pd.DataFrame({
            'n_classes': self.metrics_df['n_classes'].tolist(),
            'Log_Likelihood_rank': self.metrics_df['Log_Likelihood_rank'].astype(int),
            'BIC_rank': self.metrics_df['BIC_rank'].astype(int),
            'AIC_rank': self.metrics_df['AIC_rank'].astype(int),
            'Inertia_rank': self.metrics_df['Inertia_rank'].astype(int),
            'Silhouette_rank': self.metrics_df['Silhouette_rank'].astype(int),
            'Davies_Bouldin_rank': self.metrics_df['Davies_Bouldin_rank'].astype(int),
            'Calinski_Harabasz_rank': self.metrics_df['Calinski_Harabasz_rank'].astype(int),
            'Composite_rank': self.metrics_df['Composite_Score'].rank().astype(int)
        })
        
        # Display all composite scores before selection (DEBUG)
        print("\nComposite Scores for all k values (lower = better):")
        for idx, row in self.metrics_df.iterrows():
            print(f"  k={int(row['n_classes'])}: {row['Composite_Score']:.3f} " +
                  f"(Sil_norm={row['Silhouette_norm']:.3f}, BIC_norm={row['BIC_norm']:.3f}, AIC_norm={row['AIC_norm']:.3f})")
        
        # Select optimal model (lowest composite score = best)
        optimal_idx = self.metrics_df['Composite_Score'].idxmin()
        self.optimal_k = int(self.metrics_df.loc[optimal_idx, 'n_classes'])
        
        # BIC-only selection for comparison
        bic_optimal_idx = self.metrics_df['BIC'].idxmin()
        bic_optimal_k = int(self.metrics_df.loc[bic_optimal_idx, 'n_classes'])
        
        # Print results
        print(f"\n✓ OPTIMAL NUMBER OF CLASSES: {self.optimal_k}")
        print(f"  Composite Score: {self.metrics_df.loc[optimal_idx, 'Composite_Score']:.3f}")
        print(f"\nDetailed Metrics:")
        print(f"  Log-likelihood: {self.metrics_df.loc[optimal_idx, 'Log_Likelihood']:.2f} (rank: {rank_df.loc[optimal_idx, 'Log_Likelihood_rank']})")
        print(f"  BIC: {int(self.metrics_df.loc[optimal_idx, 'BIC'])} (rank: {rank_df.loc[optimal_idx, 'BIC_rank']})")
        print(f"  AIC: {int(self.metrics_df.loc[optimal_idx, 'AIC'])} (rank: {rank_df.loc[optimal_idx, 'AIC_rank']})")
        print(f"  Silhouette: {self.metrics_df.loc[optimal_idx, 'Silhouette']:.3f} (rank: {rank_df.loc[optimal_idx, 'Silhouette_rank']})")
        
        if bic_optimal_k != self.optimal_k:
            print(f"\n⚠ Note: BIC alone would select {bic_optimal_k} classes")
            print(f"  Multi-metric approach selects {self.optimal_k} classes instead")
        
        # Store composite scores and rankings
        self.composite_scores = dict(zip(self.metrics_df['n_classes'].tolist(), 
                                        self.metrics_df['Composite_Score'].tolist()))
        self.rank_df = rank_df
        
        # Get optimal model
        self.optimal_model = self.models[self.optimal_k]
        
        return self.optimal_k
    
    def select_optimal_k(self, method='composite', manual_k=None):
        """
        Select optimal k.
        
        Args:
            method: 'composite' (7-metric ranking), 'bic', 'aic', or 'manual'
            manual_k: Manually specified k value
            
        Returns:
            optimal_k: Selected number of classes
        """
        if method == 'manual' and manual_k is not None:
            self.optimal_k = manual_k
            self.optimal_model = self.models[self.optimal_k]
            print(f"\n✓ Manually selected k = {self.optimal_k}")
        elif method == 'composite':
            self.select_optimal_k_composite()
        elif method == 'bic':
            k_values = sorted(self.metrics.keys())
            bics = [self.metrics[k]['bic'] for k in k_values]
            self.optimal_k = k_values[np.argmin(bics)]
            self.optimal_model = self.models[self.optimal_k]
            print(f"\n✓ BIC-selected k = {self.optimal_k}")
        elif method == 'aic':
            k_values = sorted(self.metrics.keys())
            aics = [self.metrics[k]['aic'] for k in k_values]
            self.optimal_k = k_values[np.argmin(aics)]
            self.optimal_model = self.models[self.optimal_k]
            print(f"\n✓ AIC-selected k = {self.optimal_k}")
        
        return self.optimal_k
    
    def predict(self, X):
        """
        Get class assignments and probabilities for optimal model.
        
        Args:
            X: Preprocessed feature matrix
            
        Returns:
            labels: Class assignments
            probabilities: Class membership probabilities
        """
        self.labels = self.optimal_model.predict(X)
        self.probabilities = self.optimal_model.predict_proba(X)
        return self.labels, self.probabilities
    
    def plot_model_selection(self, figsize=(18, 14), save_path=None):
        """
        Plot all 7 metrics + composite score (3x3 grid).
        """
        k_values = self.metrics_df['n_classes'].tolist()
        
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        axes = axes.flatten()
        
        metric_configs = [
            ('Log-Likelihood', self.metrics_df['Log_Likelihood'].tolist(), 'purple', 'higher'),
            ('BIC', self.metrics_df['BIC'].tolist(), 'steelblue', 'lower'),
            ('AIC', self.metrics_df['AIC'].tolist(), 'orange', 'lower'),
            ('Inertia', self.metrics_df['Inertia'].tolist(), 'brown', 'lower'),
            ('Silhouette', self.metrics_df['Silhouette'].tolist(), 'green', 'higher'),
            ('Davies-Bouldin', self.metrics_df['Davies_Bouldin'].tolist(), 'crimson', 'lower'),
            ('Calinski-Harabasz', self.metrics_df['Calinski_Harabasz'].tolist(), 'teal', 'higher'),
            ('Composite Score', self.metrics_df['Composite_Score'].tolist(), 'black', 'lower')
        ]
        
        for idx, (name, scores, color, direction) in enumerate(metric_configs):
            ax = axes[idx]
            ax.plot(k_values, scores, marker='o', linewidth=2, markersize=8, color=color)
            ax.axvline(self.optimal_k, color='red', linestyle='--', linewidth=2, label='Optimal')
            ax.set_xlabel('Number of Classes', fontsize=11)
            ax.set_ylabel(f'{name} ({direction} is better)', fontsize=11)
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplot
        axes[8].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_class_probabilities(self, figsize=(12, 6), save_path=None):
        """
        Plot distribution of class membership probabilities.
        """
        fig, axes = plt.subplots(1, self.optimal_k, figsize=figsize, sharey=True)
        if self.optimal_k == 1:
            axes = [axes]
        
        for i in range(self.optimal_k):
            axes[i].hist(self.probabilities[:, i], bins=50, color=f'C{i}', alpha=0.7, edgecolor='black')
            axes[i].set_xlabel('Probability', fontsize=11)
            axes[i].set_title(f'Class {i}', fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3, axis='y')
            
            if i == 0:
                axes[i].set_ylabel('Frequency', fontsize=11)
        
        fig.suptitle('Distribution of Class Membership Probabilities', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_tsne(self, X, perplexity=30, figsize=(12, 5), max_samples=5000, save_path=None):
        """
        Create t-SNE visualization of latent classes.
        """
        print(f"\nComputing t-SNE for k={self.optimal_k}...")
        
        # Sample data if too large
        if X.shape[0] > max_samples:
            print(f"  Sampling {max_samples} points for visualization...")
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(X.shape[0], max_samples, replace=False)
            X_sample = X[sample_idx]
            labels_sample = self.labels[sample_idx]
            probs_sample = self.probabilities[sample_idx]
        else:
            X_sample = X
            labels_sample = self.labels
            probs_sample = self.probabilities
        
        # Use faster t-SNE settings for large datasets
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            random_state=self.random_state,
            n_iter=300,  # Reduced from default 1000 for speed
            learning_rate='auto',
            init='pca',  # PCA initialization is faster than random
            n_jobs=-1  # Use all CPU cores
        )
        X_tsne = tsne.fit_transform(X_sample)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Colored by class
        scatter1 = ax1.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=labels_sample,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        ax1.set_xlabel('t-SNE 1', fontsize=11)
        ax1.set_ylabel('t-SNE 2', fontsize=11)
        ax1.set_title(f't-SNE Visualization (k={self.optimal_k})', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Class', fontsize=11)
        
        # Add class sizes
        unique, counts = np.unique(labels_sample, return_counts=True)
        size_text = "\n".join([f"Class {i}: n={c:,}" for i, c in zip(unique, counts)])
        ax1.text(0.02, 0.98, size_text, transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
                fontsize=9)
        
        # Plot 2: Colored by max probability
        max_probs = np.max(probs_sample, axis=1)
        scatter2 = ax2.scatter(
            X_tsne[:, 0], X_tsne[:, 1],
            c=max_probs,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        ax2.set_xlabel('t-SNE 1', fontsize=11)
        ax2.set_ylabel('t-SNE 2', fontsize=11)
        ax2.set_title('Colored by Max Class Probability', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Max Probability', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_pca(self, X, figsize=(12, 5), save_path=None):
        """
        Create PCA visualization of latent classes.
        """
        print(f"\nComputing PCA for k={self.optimal_k}...")
        
        pca = PCA(n_components=2, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Colored by class
        scatter1 = ax1.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=self.labels,
            cmap='tab10',
            alpha=0.6,
            s=20
        )
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax1.set_title(f'PCA Visualization (k={self.optimal_k})', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Class', fontsize=11)
        
        # Add class sizes
        unique, counts = np.unique(self.labels, return_counts=True)
        size_text = "\n".join([f"Class {i}: n={c:,}" for i, c in zip(unique, counts)])
        ax1.text(0.02, 0.98, size_text, transform=ax1.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', alpha=0.8, facecolor='white'),
                fontsize=9)
        
        # Plot 2: Colored by max probability
        max_probs = np.max(self.probabilities, axis=1)
        scatter2 = ax2.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=max_probs,
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        ax2.set_title('Colored by Max Class Probability', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Max Probability', fontsize=11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_class_sizes(self, figsize=(8, 6), save_path=None):
        """
        Plot distribution of class sizes.
        """
        unique, counts = np.unique(self.labels, return_counts=True)
        percentages = counts / len(self.labels) * 100
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = [f'C{i}' for i in range(self.optimal_k)]
        bars = ax.bar(unique, counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add count and percentage labels
        for bar, count, pct in zip(bars, counts, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count:,}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Class Sizes (k={self.optimal_k})', fontsize=14, fontweight='bold')
        ax.set_xticks(unique)
        ax.set_xticklabels([f'Class {i}' for i in unique])
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_clinical_profiles_heatmap(self, X, feature_names, figsize=(14, 10), save_path=None):
        """
        Generate heatmap showing mean feature values for each class (clinical profiles).
        """
        # Calculate mean values for each class
        class_profiles = []
        for k in range(self.optimal_k):
            class_mask = self.labels == k
            class_mean = X[class_mask].mean(axis=0)
            class_profiles.append(class_mean)
        
        class_profiles = np.array(class_profiles)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(class_profiles.T, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(np.arange(self.optimal_k))
        ax.set_yticks(np.arange(len(feature_names)))
        ax.set_xticklabels([f'Class {i}\n(n={np.sum(self.labels==i)})' for i in range(self.optimal_k)])
        ax.set_yticklabels(feature_names, fontsize=9)
        
        # Rotate the tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Standardized Mean Value', rotation=270, labelpad=20, fontsize=11)
        
        # Add title
        ax.set_title('Clinical Profiles Heatmap - Mean Feature Values by Class', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Clinical Features', fontsize=12)
        
        # Add grid
        ax.set_xticks(np.arange(self.optimal_k + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(len(feature_names) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_feature_distributions(self, X, feature_names, top_n=10, figsize=(16, 12), save_path=None):
        """
        Plot distributions of top N features across classes using violin plots.
        """
        # Calculate variance for each feature across classes
        feature_variance = []
        for i in range(X.shape[1]):
            class_means = [X[self.labels == k, i].mean() for k in range(self.optimal_k)]
            feature_variance.append(np.var(class_means))
        
        # Get top N features with highest variance
        top_indices = np.argsort(feature_variance)[-top_n:][::-1]
        
        # Create subplots
        n_cols = 2
        n_rows = (top_n + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()
        
        for idx, feat_idx in enumerate(top_indices):
            ax = axes[idx]
            
            # Prepare data for violin plot
            data_by_class = [X[self.labels == k, feat_idx] for k in range(self.optimal_k)]
            
            # Create violin plot
            parts = ax.violinplot(data_by_class, positions=range(self.optimal_k),
                                  showmeans=True, showmedians=True)
            
            # Color the violins
            for pc, color_idx in zip(parts['bodies'], range(self.optimal_k)):
                pc.set_facecolor(f'C{color_idx}')
                pc.set_alpha(0.7)
            
            ax.set_xlabel('Class', fontsize=10)
            ax.set_ylabel('Standardized Value', fontsize=10)
            ax.set_title(feature_names[feat_idx], fontsize=11, fontweight='bold')
            ax.set_xticks(range(self.optimal_k))
            ax.set_xticklabels([f'{i}' for i in range(self.optimal_k)])
            ax.grid(True, alpha=0.3, axis='y')
        
        # Hide unused subplots
        for idx in range(top_n, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Top {top_n} Features with Highest Class Variance', 
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_feature_importance_bars(self, X, feature_names, top_n=15, figsize=(12, 8), save_path=None):
        """
        Plot feature importance based on variance across classes.
        """
        # Calculate variance for each feature across classes
        feature_importance = []
        for i in range(X.shape[1]):
            class_means = [X[self.labels == k, i].mean() for k in range(self.optimal_k)]
            feature_importance.append(np.var(class_means))
        
        # Get top N features
        top_indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_names[i] for i in top_indices]
        top_scores = [feature_importance[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
        bars = ax.barh(range(top_n), top_scores, color=colors, edgecolor='black', linewidth=0.8)
        
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_features, fontsize=10)
        ax.set_xlabel('Variance Across Classes', fontsize=12)
        ax.set_title(f'Top {top_n} Features by Class Separation', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, top_scores)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2., f'{score:.3f}',
                   ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def plot_class_statistics_table(self, X, feature_names, save_path=None):
        """
        Create a visual table showing statistics for each class.
        """
        fig, ax = plt.subplots(figsize=(14, max(8, self.optimal_k * 1.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Calculate statistics for each class
        table_data = []
        headers = ['Class', 'Size', '%', 'Top 3 High Features', 'Top 3 Low Features']
        
        for k in range(self.optimal_k):
            class_mask = self.labels == k
            class_size = np.sum(class_mask)
            class_pct = class_size / len(self.labels) * 100
            
            # Get class means
            class_means = X[class_mask].mean(axis=0)
            
            # Top 3 highest features
            top_high_idx = np.argsort(class_means)[-3:][::-1]
            top_high = ', '.join([f"{feature_names[i][:20]}" for i in top_high_idx])
            
            # Top 3 lowest features
            top_low_idx = np.argsort(class_means)[:3]
            top_low = ', '.join([f"{feature_names[i][:20]}" for i in top_low_idx])
            
            table_data.append([
                f'Class {k}',
                f'{class_size:,}',
                f'{class_pct:.1f}%',
                top_high,
                top_low
            ])
        
        table = ax.table(cellText=table_data, colLabels=headers,
                        cellLoc='left', loc='center',
                        colWidths=[0.08, 0.10, 0.08, 0.37, 0.37])
        
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(len(headers)):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white', fontsize=10)
        
        # Style rows
        for i in range(1, len(table_data) + 1):
            for j in range(len(headers)):
                cell = table[(i, j)]
                if i % 2 == 0:
                    cell.set_facecolor('#f0f0f0')
                cell.set_edgecolor('gray')
        
        plt.title('Class Statistics and Dominant Features', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig
    
    def generate_all_plots(self, X, feature_names=None, output_dir='results', include_clinical_profiles=False):
        """
        Generate all diagnostic plots.
        
        Args:
            X: Preprocessed feature matrix
            feature_names: List of feature names (required if include_clinical_profiles=True)
            output_dir: Directory to save plots
            include_clinical_profiles: Whether to generate clinical profile analysis plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*80)
        if include_clinical_profiles:
            print("GENERATING DIAGNOSTIC PLOTS AND CLINICAL PROFILE ANALYSIS")
        else:
            print("GENERATING DIAGNOSTIC PLOTS")
        print("="*80)
        
        # Only plot model selection if multiple models were fitted
        if len(self.models) > 1:
            self.plot_model_selection(
                save_path=output_path / 'lca_model_selection_metrics.png'
            )
            plt.close()
        else:
            print("Skipping model selection plot (only one k fitted)")
        
        self.plot_class_sizes(
            save_path=output_path / 'lca_class_sizes.png'
        )
        plt.close()
        
        self.plot_class_probabilities(
            save_path=output_path / 'lca_class_probabilities.png'
        )
        plt.close()
        
        self.plot_pca(
            X,
            save_path=output_path / 'lca_pca_visualization.png'
        )
        plt.close()
        
        self.plot_tsne(
            X,
            save_path=output_path / 'lca_tsne_visualization.png'
        )
        plt.close()
        
        # Generate clinical profile analysis if requested
        if include_clinical_profiles and feature_names is not None:
            self.plot_clinical_profiles_heatmap(
                X, feature_names,
                save_path=output_path / 'lca_clinical_profiles_heatmap.png'
            )
            plt.close()
            
            self.plot_feature_distributions(
                X, feature_names, top_n=10,
                save_path=output_path / 'lca_feature_distributions.png'
            )
            plt.close()
            
            self.plot_feature_importance_bars(
                X, feature_names, top_n=15,
                save_path=output_path / 'lca_feature_importance.png'
            )
            plt.close()
            
            self.plot_class_statistics_table(
                X, feature_names,
                save_path=output_path / 'lca_class_statistics.png'
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
    # Use read_only=True to allow concurrent access
    conn = duckdb.connect(db_path, read_only=True)
    query = f"SELECT * FROM {table_name}"
    df = conn.execute(query).fetchdf()
    conn.close()
    return df


def run_lca_pipeline(df, exclude_cols, k_range=range(2, 7),
                     n_init=50, max_iter=200,
                     correlation_threshold=0.8, log_transform=False,
                     subsample_data=None, output_dir='backend/output_dir/LCA',
                     manual_k=None, selection_method='composite', 
                     random_state=None):
    """
    Main function to run the complete Latent Class Analysis pipeline using StepMix.
    
    Args:
        df: Input DataFrame with clinical variables
        exclude_cols: List of columns to exclude from clustering
        k_range: Range of k values to test
        n_init: Number of initializations for StepMix (default=50 for robust convergence)
        max_iter: Maximum iterations for EM algorithm
        correlation_threshold: Threshold for removing correlated variables
        log_transform: Whether to apply log transformation
        subsample_data: Fraction of data to use (None = use all data)
        output_dir: Directory to save results
        manual_k: Manually specify optimal k (None = auto-select)
        selection_method: Method for selecting k ('composite', 'bic', 'aic', or 'manual')
        random_state: Random seed (None = try multiple random initializations, recommended)
        
    Returns:
        dict: Dictionary containing:
            - optimal_k: Selected optimal k
            - labels: Class assignments
            - probabilities: Class membership probabilities
    """
    print("\n" + "="*80)
    print("LATENT CLASS ANALYSIS (LCA) PIPELINE - STEPMIX")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Input data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    
    # Subsample data if requested
    if subsample_data is not None:
        print(f"\nSubsampling {subsample_data*100:.1f}% of data...")
        df = df.sample(frac=subsample_data, random_state=random_state)
        print(f"Subsampled data: {df.shape[0]:,} rows")
    
    # Initialize model
    model = LatentClassAnalysis(
        k_range=k_range,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    
    # Preprocess data
    X_processed, df_transformed, feature_names = model.preprocess_data(
        df=df,
        exclude_cols=exclude_cols,
        correlation_threshold=correlation_threshold,
        log_transform=log_transform
    )
    
    # Fit models for all k values
    if manual_k is None:
        model.fit_models(X_processed)
        
        # Select optimal k using specified method
        optimal_k = model.select_optimal_k(
            method=selection_method,
            manual_k=manual_k
        )
    else:
        # Only fit the manual k
        print(f"\nFitting single model with k={manual_k}...")
        model.k_range = [manual_k]
        model.fit_models(X_processed)
        optimal_k = model.select_optimal_k(method='manual', manual_k=manual_k)
    
    # Get predictions
    labels, probabilities = model.predict(X_processed)
    
    # Add class labels to DataFrame
    df_with_classes = df_transformed.copy()
    df_with_classes['lca_class'] = labels
    
    # Add class probabilities
    for i in range(optimal_k):
        df_with_classes[f'prob_class_{i}'] = probabilities[:, i]
    
    # Generate all diagnostic plots
    model.generate_all_plots(X_processed, output_dir=output_dir)
    
    # If optimal k is not 4, generate k=4 results as well
    if optimal_k != 4 and 4 in model.models:
        print("\n" + "="*80)
        print("GENERATING ADDITIONAL RESULTS FOR k=4")
        print("="*80)
        print(f"Note: Optimal k={optimal_k}, but also generating k=4 results as requested")
        
        # Temporarily switch to k=4 model
        original_optimal_k = model.optimal_k
        original_optimal_model = model.optimal_model
        original_labels = model.labels
        original_probabilities = model.probabilities
        
        model.optimal_k = 4
        model.optimal_model = model.models[4]
        labels_k4, probabilities_k4 = model.predict(X_processed)
        
        # Create separate output directory for k=4
        output_path_k4 = Path(output_dir) / 'k4_results'
        output_path_k4.mkdir(parents=True, exist_ok=True)
        
        # Generate plots for k=4 with clinical profile analysis
        model.generate_all_plots(X_processed, feature_names=feature_names, 
                                output_dir=str(output_path_k4), 
                                include_clinical_profiles=True)
        
        # Save k=4 results
        df_with_classes_k4 = df_transformed.copy()
        df_with_classes_k4['lca_class'] = labels_k4
        for i in range(4):
            df_with_classes_k4[f'prob_class_{i}'] = probabilities_k4[:, i]
        
        results_file_k4 = output_path_k4 / 'lca_results_k4.csv'
        df_with_classes_k4.to_csv(results_file_k4, index=False)
        print(f"k=4 results saved to: {results_file_k4}")
        
        # Save k=4 summary
        summary_file_k4 = output_path_k4 / 'lca_summary_k4.txt'
        with open(summary_file_k4, 'w') as f:
            f.write("="*80 + "\n")
            f.write("LATENT CLASS ANALYSIS SUMMARY (k=4 RESULTS)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Number of classes: 4 (forced, optimal was {original_optimal_k})\n")
            f.write(f"Selection method: {selection_method}\n\n")
            f.write("Class sizes:\n")
            f.write(df_with_classes_k4['lca_class'].value_counts().sort_index().to_string())
            f.write("\n\n")
            f.write("Class proportions:\n")
            f.write((df_with_classes_k4['lca_class'].value_counts(normalize=True).sort_index() * 100).to_string())
            f.write("\n\n")
            if 4 in model.metrics:
                f.write("Model metrics for k=4:\n")
                for metric, value in model.metrics[4].items():
                    f.write(f"  {metric}: {value:.4f}\n")
        
        print(f"k=4 summary saved to: {summary_file_k4}")
        print(f"\nk=4 class sizes:")
        print(df_with_classes_k4['lca_class'].value_counts().sort_index())
        print(f"\nk=4 class proportions (%):")
        print((df_with_classes_k4['lca_class'].value_counts(normalize=True).sort_index() * 100).round(2))
        
        # Restore original optimal model
        model.optimal_k = original_optimal_k
        model.optimal_model = original_optimal_model
        model.labels = original_labels
        model.probabilities = original_probabilities
        
        print(f"\n✓ Generated both optimal k={original_optimal_k} and k=4 results")
    elif optimal_k == 4:
        print("\n✓ Optimal k is already 4, no additional results needed")
    
    # Save results to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'lca_results.csv'
    df_with_classes.to_csv(results_file, index=False)
    print(f"\nResults saved to: {results_file}")
    
    # Save summary statistics
    summary_file = output_path / 'lca_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LATENT CLASS ANALYSIS SUMMARY (STEPMIX)\n")
        f.write("="*80 + "\n\n")
        f.write(f"Optimal number of classes: {optimal_k}\n")
        f.write(f"Selection method: {selection_method}\n\n")
        f.write("Class sizes:\n")
        f.write(df_with_classes['lca_class'].value_counts().sort_index().to_string())
        f.write("\n\n")
        f.write("Class proportions:\n")
        f.write((df_with_classes['lca_class'].value_counts(normalize=True).sort_index() * 100).to_string())
        f.write("\n\n")
        f.write("Model metrics:\n")
        for k in sorted(model.metrics.keys()):
            f.write(f"\nk={k}:\n")
            for metric, value in model.metrics[k].items():
                f.write(f"  {metric}: {value:.4f}\n")
    
    print(f"Summary saved to: {summary_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("LATENT CLASS SUMMARY")
    print("="*80)
    print(f"\nOptimal k: {optimal_k}")
    print(f"\nClass sizes:")
    print(df_with_classes['lca_class'].value_counts().sort_index())
    print(f"\nClass proportions (%):")
    print((df_with_classes['lca_class'].value_counts(normalize=True).sort_index() * 100).round(2))
    
    # Return results
    results = {
        'optimal_k': optimal_k,
        'labels': labels,
        'probabilities': probabilities,
        'lca_object': model,  # Include LCA object for debugging
    }
    
    print("\n" + "="*80)
    print("LCA PIPELINE COMPLETE")
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
    Example of how to use the LCA pipeline
    """
    
    # Load data from DuckDB
    df = data_loader_duckdb(
        '../../data/fixed_hour_length_issue_BRITSSAITS_10112025.duckdb',
        'clipped_brits_saits'
    )
    
    # Define columns to exclude (from TAME - less than 24H.ipynb notebook)
    exclude_cols = [
        'stay_id', 'hour', 'chart_hour', 'sofa_total', 'sofa_cat', 
        'age_at_admission', 'phenotype', 'model', 
        'all_icd_codes', 'all_icd_versions', 'is_sepsis'
    ]
    
    # Run the LCA pipeline
    results = run_lca_pipeline(
        df=df,
        exclude_cols=exclude_cols,
        k_range=range(2, 7),
        n_init=10,
        max_iter=200,
        correlation_threshold=0.8,
        log_transform=False,
        subsample_data=None,  # Use full dataset
        output_dir='backend/output_dir/LCA_fixed_hour_length',
        manual_k=None,  # Set to specific k value to skip automatic selection
        selection_method='composite',  # Use 7-metric composite scoring
        random_state=42
    )
    
    # Access results
    print(f"\nOptimal k: {results['optimal_k']}")
    print(f"Labels shape: {results['labels'].shape}")
    print(f"Probabilities shape: {results['probabilities'].shape}")
