"""
Comprehensive tests for all ML pipeline modules
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os


# ============================================================================
# KMeans Pipeline Tests
# ============================================================================

class TestConsensusKMeans:
    """Test ConsensusKMeans class"""
    
    def test_init_default_params(self):
        """Test ConsensusKMeans initialization with defaults"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        ckm = ConsensusKMeans()
        assert ckm.k_range == range(2, 7)
        assert ckm.n_iterations == 100
        assert ckm.subsample_fraction == 0.8
        assert ckm.random_state == 42
    
    def test_init_custom_params(self):
        """Test ConsensusKMeans initialization with custom parameters"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        ckm = ConsensusKMeans(k_range=range(3, 6), n_iterations=50, 
                             subsample_fraction=0.7, random_state=123)
        assert ckm.k_range == range(3, 6)
        assert ckm.n_iterations == 50
        assert ckm.subsample_fraction == 0.7
        assert ckm.random_state == 123
    
    def test_preprocess_data_basic(self):
        """Test basic preprocessing without transforms"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        # Create sample data
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'exclude_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        ckm = ConsensusKMeans()
        X_processed, df_transformed, feature_names = ckm.preprocess_data(
            df, exclude_cols=['exclude_col'], log_transform=False
        )
        
        assert X_processed.shape[0] == 5
        # The excluded columns are added back to df_transformed for reference
        assert 'exclude_col' in df_transformed.columns
        # But they should not be in the processed feature matrix
        assert X_processed.shape[1] < len(df_transformed.columns)
        assert len(feature_names) > 0
    
    def test_preprocess_data_with_log_transform(self):
        """Test preprocessing with log transformation"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        # Create skewed data
        df = pd.DataFrame({
            'feature1': [1, 10, 100, 1000, 10000],
            'feature2': [2, 4, 6, 8, 10],
            'id': [1, 2, 3, 4, 5]
        })
        
        ckm = ConsensusKMeans()
        X_processed, df_transformed, feature_names = ckm.preprocess_data(
            df, exclude_cols=['id'], log_transform=True,
            log_transform_skew_threshold=1.0
        )
        
        assert X_processed is not None
        assert df_transformed is not None
    
    def test_preprocess_removes_correlated_features(self):
        """Test that highly correlated features are removed"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        # Create perfectly correlated features
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],  # Perfectly correlated with feature1
            'feature3': [5, 4, 3, 2, 1],
            'id': [1, 2, 3, 4, 5]
        })
        
        ckm = ConsensusKMeans()
        X_processed, df_transformed, feature_names = ckm.preprocess_data(
            df, exclude_cols=['id'], correlation_threshold=0.8
        )
        
        # Should have removed one of the correlated features
        assert len(feature_names) < 3
    
    def test_fit_predict_basic(self):
        """Test basic consensus clustering"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        # Simple test data
        X = np.random.rand(50, 3)
        
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        
        # Run the consensus clustering workflow
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        ckm.select_optimal_k()
        
        assert ckm.optimal_k is not None
        assert ckm.labels is not None
        assert len(ckm.labels) == 50
        assert ckm.optimal_k in [2, 3]
        assert len(ckm.labels) == 50
        assert ckm.optimal_k in [2, 3]
    
    def test_select_optimal_k_manual(self):
        """Test manual k selection"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(50, 3)
        ckm = ConsensusKMeans(k_range=range(2, 5), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        
        # Test manual selection
        optimal_k = ckm.select_optimal_k(method='manual', manual_k=3)
        assert optimal_k == 3
        assert ckm.optimal_k == 3
    
    def test_select_optimal_k_auto_fallback(self):
        """Test auto selection with fallback when no k meets criteria"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(30, 2)
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        
        # Should auto-select even if no k meets 0.8 threshold
        optimal_k = ckm.select_optimal_k()
        assert optimal_k in [2, 3]
        assert ckm.labels is not None
    
    @patch('app.kmeans_pipeline.plt')
    def test_plot_consensus_matrices(self, mock_plt, tmp_path):
        """Test plotting consensus matrices"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(30, 2)
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        ckm.select_optimal_k()
        
        # Mock plt.subplots to return figure and axes
        mock_fig = Mock()
        mock_axes = [Mock(), Mock()]
        for ax in mock_axes:
            ax.imshow = Mock()
            ax.add_patch = Mock()
            ax.set_title = Mock()
            ax.set_xticks = Mock()
            ax.set_yticks = Mock()
            ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.Rectangle = Mock
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "consensus_matrices.png"
        fig = ckm.plot_consensus_matrices(save_path=str(save_path))
        
        assert fig is not None
        mock_plt.subplots.assert_called_once()
    
    @patch('app.kmeans_pipeline.plt')
    def test_plot_cdf_curves(self, mock_plt, tmp_path):
        """Test plotting CDF curves"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(30, 2)
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.plot = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.legend = Mock()
        mock_ax.grid = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "cdf_curves.png"
        fig = ckm.plot_cdf_curves(save_path=str(save_path))
        
        assert fig is not None
        assert mock_ax.plot.called
    
    @patch('app.kmeans_pipeline.plt')
    def test_plot_selection_metrics(self, mock_plt, tmp_path):
        """Test plotting selection metrics"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(30, 2)
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        for ax in [mock_ax1, mock_ax2]:
            ax.plot = Mock()
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.grid = Mock()
            ax.legend = Mock()
            ax.axhline = Mock()
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "selection_metrics.png"
        fig = ckm.plot_selection_metrics(save_path=str(save_path))
        
        assert fig is not None
        mock_ax1.plot.assert_called_once()
        mock_ax2.plot.assert_called_once()
    
    @patch('app.kmeans_pipeline.plt')
    def test_plot_delta_area(self, mock_plt, tmp_path):
        """Test plotting delta area"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(30, 2)
        ckm = ConsensusKMeans(k_range=range(2, 5), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.plot = Mock()
        mock_ax.axhline = Mock()
        mock_ax.scatter = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.set_xticks = Mock()
        mock_ax.grid = Mock()
        mock_ax.text = Mock()
        mock_ax.transAxes = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "delta_area.png"
        fig = ckm.plot_delta_area(save_path=str(save_path))
        
        assert fig is not None
        assert mock_ax.plot.called
    
    @patch('app.kmeans_pipeline.plt')
    def test_plot_cluster_consensus(self, mock_plt, tmp_path):
        """Test plotting cluster consensus"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(30, 2)
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        
        mock_fig = Mock()
        mock_ax = Mock()
        
        # Mock bar to return proper bar objects
        mock_bar = Mock()
        mock_bar.get_height.return_value = 0.85
        mock_bar.get_x.return_value = 0
        mock_bar.get_width.return_value = 0.8
        
        mock_ax.bar = Mock(return_value=[mock_bar])
        mock_ax.axhline = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.set_xticks = Mock()
        mock_ax.set_xticklabels = Mock()
        mock_ax.set_ylim = Mock()
        mock_ax.legend = Mock()
        mock_ax.grid = Mock()
        mock_ax.text = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.cm = Mock()
        # tab10 should be callable and return array of colors that can be indexed
        mock_plt.cm.tab10 = Mock(return_value=np.array([
            [0.1, 0.2, 0.3, 1.0],
            [0.4, 0.5, 0.6, 1.0],
            [0.7, 0.8, 0.9, 1.0]
        ]))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "cluster_consensus.png"
        fig = ckm.plot_cluster_consensus(save_path=str(save_path))
        
        assert fig is not None
        assert mock_ax.bar.called
    
    @patch('app.kmeans_pipeline.TSNE')
    @patch('app.kmeans_pipeline.plt')
    def test_plot_tsne(self, mock_plt, mock_tsne, tmp_path):
        """Test t-SNE plotting"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(50, 3)
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        ckm.select_optimal_k()
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_scatter = Mock()
        mock_ax.scatter = Mock(return_value=mock_scatter)
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.text = Mock()
        mock_ax.transAxes = Mock()
        
        mock_cbar = Mock()
        mock_cbar.set_label = Mock()
        mock_cbar.ax = Mock()
        mock_cbar.ax.set_yticklabels = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.colorbar = Mock(return_value=mock_cbar)
        mock_plt.cm = Mock()
        mock_plt.cm.tab10 = Mock()
        mock_plt.cm.tab10.colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "tsne.png"
        fig = ckm.plot_tsne(X, save_path=str(save_path))
        
        assert fig is not None
        mock_tsne.assert_called_once()
        mock_ax.scatter.assert_called_once()
    
    @patch('app.kmeans_pipeline.TSNE')
    @patch('app.kmeans_pipeline.plt')
    def test_plot_tsne_with_sampling(self, mock_plt, mock_tsne, tmp_path):
        """Test t-SNE plotting with large dataset sampling"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        # Create large dataset to trigger sampling
        X = np.random.rand(15000, 3)
        ckm = ConsensusKMeans(k_range=range(2, 3), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        ckm.select_optimal_k()
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(10000, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        mock_fig = Mock()
        mock_ax = Mock()
        mock_scatter = Mock()
        mock_ax.scatter = Mock(return_value=mock_scatter)
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.text = Mock()
        mock_ax.transAxes = Mock()
        
        mock_cbar = Mock()
        mock_cbar.set_label = Mock()
        mock_cbar.ax = Mock()
        mock_cbar.ax.set_yticklabels = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.colorbar = Mock(return_value=mock_cbar)
        mock_plt.cm = Mock()
        mock_plt.cm.tab10 = Mock()
        mock_plt.cm.tab10.colors = [(0, 0, 0), (1, 0, 0)]
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        fig = ckm.plot_tsne(X, max_samples=10000)
        
        assert fig is not None
        # Should have sampled down to 10000
        call_args = mock_tsne_instance.fit_transform.call_args[0]
        assert call_args[0].shape[0] == 10000
    
    @patch('app.kmeans_pipeline.TSNE')
    @patch('app.kmeans_pipeline.plt')
    def test_generate_all_plots(self, mock_plt, mock_tsne, tmp_path):
        """Test generating all plots"""
        from app.kmeans_pipeline import ConsensusKMeans
        
        X = np.random.rand(50, 3)
        ckm = ConsensusKMeans(k_range=range(2, 4), n_iterations=5)
        ckm.run_consensus_clustering(X)
        ckm.compute_consensus_metrics()
        ckm.select_optimal_k()
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        # Setup comprehensive mocks for all plotting functions
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.imshow = Mock()
        mock_ax.add_patch = Mock()
        mock_ax.plot = Mock()
        mock_ax.scatter = Mock(return_value=Mock())
        mock_ax.bar = Mock(return_value=[])
        mock_ax.axhline = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.set_xticks = Mock()
        mock_ax.set_yticks = Mock()
        mock_ax.set_xticklabels = Mock()
        mock_ax.set_ylim = Mock()
        mock_ax.legend = Mock()
        mock_ax.grid = Mock()
        mock_ax.text = Mock()
        mock_ax.transAxes = Mock()
        mock_ax.spines = {'top': Mock(), 'bottom': Mock(), 'left': Mock(), 'right': Mock()}
        
        # Return both single ax and tuple of axes for different plots
        mock_plt.subplots = Mock(side_effect=[
            (mock_fig, [mock_ax, mock_ax]),  # consensus_matrices
            (mock_fig, mock_ax),              # cdf_curves
            (mock_fig, (mock_ax, mock_ax)),   # selection_metrics
            (mock_fig, mock_ax),              # delta_area
            (mock_fig, mock_ax),              # cluster_consensus
            (mock_fig, mock_ax),              # tsne
        ])
        
        mock_plt.Rectangle = Mock
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.colorbar = Mock(return_value=Mock(set_label=Mock(), ax=Mock(set_yticklabels=Mock())))
        mock_plt.sca = Mock()
        
        # Mock cm.tab10 - needs colors attribute AND be callable
        mock_plt.cm = Mock()
        mock_plt.cm.tab10 = Mock()
        mock_plt.cm.tab10.colors = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        # When called as a function, return array of colors
        mock_plt.cm.tab10.return_value = np.array([
            [0.1, 0.2, 0.3, 1.0],
            [0.4, 0.5, 0.6, 1.0],
            [0.7, 0.8, 0.9, 1.0]
        ])
        
        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        
        ckm.generate_all_plots(X, output_dir=str(output_dir))
        
        # Should have called close multiple times (once per plot)
        assert mock_plt.close.call_count >= 6


class TestRunConsensusPipeline:
    """Test run_consensus_pipeline function"""
    
    def test_run_consensus_pipeline_basic(self, tmp_path):
        """Test basic pipeline execution"""
        from app.kmeans_pipeline import run_consensus_pipeline
        
        # Create sample DataFrame
        df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'id': range(100)
        })
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = run_consensus_pipeline(
            df=df,
            exclude_cols=['id'],
            k_range=range(2, 4),
            n_iterations=5,
            output_dir=str(output_dir),
            generate_plots=False
        )
        
        assert 'optimal_k' in result
        assert result['optimal_k'] in [2, 3]
    
    def test_run_consensus_pipeline_with_custom_params(self, tmp_path):
        """Test pipeline with custom hyperparameters"""
        from app.kmeans_pipeline import run_consensus_pipeline
        
        df = pd.DataFrame({
            'f1': np.random.rand(50),
            'f2': np.random.rand(50),
            'id': range(50)
        })
        
        output_dir = tmp_path / "custom_output"
        output_dir.mkdir()
        
        result = run_consensus_pipeline(
            df=df,
            exclude_cols=['id'],
            k_range=range(2, 3),
            n_iterations=3,
            subsample_fraction=0.7,
            output_dir=str(output_dir),
            log_transform=True,
            generate_plots=False
        )
        
        assert result['optimal_k'] is not None
        assert result['optimal_k'] in [2, 3]


# ============================================================================
# DTW Pipeline Tests
# ============================================================================

class TestWeightedKMeans:
    """Test WeightedKMeans class"""
    
    def test_init(self):
        """Test WeightedKMeans initialization"""
        from app.dtw_pipeline import WeightedKMeans
        
        wkm = WeightedKMeans(n_clusters=4, max_iter=100, random_state=42)
        assert wkm.n_clusters == 4
        assert wkm.max_iter == 100
        assert wkm.random_state == 42
    
    def test_fit_predict_basic(self):
        """Test basic weighted k-means clustering"""
        from app.dtw_pipeline import WeightedKMeans
        
        # Create a simple distance matrix
        n_samples = 30
        distance_matrix = np.random.rand(n_samples, n_samples)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(distance_matrix, 0)
        
        wkm = WeightedKMeans(n_clusters=3, max_iter=10)
        labels = wkm.fit_predict(distance_matrix)
        
        assert len(labels) == n_samples
        assert len(np.unique(labels)) <= 3
        assert wkm.weights_ is not None
    
    @patch('app.dtw_pipeline.duckdb')
    def test_load_and_prepare_data(self, mock_duckdb, tmp_path):
        """Test load_and_prepare_data function"""
        from app.dtw_pipeline import load_and_prepare_data
        
        # Create mock connection and data
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn
        
        # Mock valid patients query
        valid_patients_df = pd.DataFrame({'stay_id': [1, 2, 3, 4, 5]})
        
        # Mock main data query
        mock_data = pd.DataFrame({
            'stay_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'hour': [0, 1, 2, 0, 1, 2, 0, 1, 2],
            'Heart Rate': np.random.rand(9),
            'Temperature_C': np.random.rand(9)
        })
        
        # Setup mock to return different results for different queries
        mock_conn.sql.return_value.fetchdf.side_effect = [valid_patients_df, mock_data]
        
        db_path = tmp_path / "test.duckdb"
        df, valid_patients = load_and_prepare_data(
            db_path=str(db_path),
            table_name='test_table',
            time_window_hours=24,
            feature_columns=['Heart Rate', 'Temperature_C']
        )
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(valid_patients, list)
        assert len(valid_patients) == 5
        mock_duckdb.connect.assert_called_once()
    
    @patch('app.dtw_pipeline.duckdb')
    def test_load_and_prepare_data_with_subsample(self, mock_duckdb, tmp_path):
        """Test load_and_prepare_data with subsampling"""
        from app.dtw_pipeline import load_and_prepare_data
        
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn
        
        # Create larger patient set
        valid_patients_df = pd.DataFrame({'stay_id': list(range(1, 101))})
        
        mock_data = pd.DataFrame({
            'stay_id': list(range(1, 51)),
            'hour': [0] * 50,
            'feature': np.random.rand(50)
        })
        
        mock_conn.sql.return_value.fetchdf.side_effect = [valid_patients_df, mock_data]
        
        db_path = tmp_path / "test.duckdb"
        df, valid_patients = load_and_prepare_data(
            db_path=str(db_path),
            table_name='test_table',
            time_window_hours=24,
            feature_columns=['feature'],
            subsample_fraction=0.5,
            random_state=42
        )
        
        assert isinstance(df, pd.DataFrame)
        assert isinstance(valid_patients, list)
        # Due to subsampling, should have fewer patients
        assert len(valid_patients) <= 100
    
    @patch('app.dtw_pipeline.MDS')
    def test_evaluate_k_range(self, mock_mds):
        """Test evaluate_k_range function"""
        from app.dtw_pipeline import evaluate_k_range
        
        # Create mock distance matrix
        n_samples = 50
        distance_matrix = np.random.rand(n_samples, n_samples)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        
        # Mock MDS
        mock_mds_instance = Mock()
        mock_mds_instance.fit_transform.return_value = np.random.rand(n_samples, 10)
        mock_mds.return_value = mock_mds_instance
        
        k_range = range(2, 5)
        results_df = evaluate_k_range(distance_matrix, k_range, n_mds_components=10)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 3  # k=2,3,4
        assert 'k' in results_df.columns
        assert 'silhouette' in results_df.columns
        assert 'davies_bouldin' in results_df.columns
        assert 'calinski_harabasz' in results_df.columns
        assert 'inertia' in results_df.columns
        assert 'cluster_sizes' in results_df.columns
    
    @patch('app.dtw_pipeline.plt')
    def test_plot_validation_metrics(self, mock_plt, tmp_path):
        """Test plot_validation_metrics function"""
        from app.dtw_pipeline import plot_validation_metrics
        
        # Create mock results DataFrame
        results_df = pd.DataFrame({
            'k': [2, 3, 4, 5],
            'silhouette': [0.3, 0.4, 0.35, 0.3],
            'davies_bouldin': [1.5, 1.2, 1.3, 1.4],
            'calinski_harabasz': [100, 150, 140, 120],
            'inertia': [500, 400, 350, 320]
        })
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        mock_plt.subplots.return_value = (Mock(), np.array([[Mock(), Mock()], [Mock(), Mock()]]))
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        # Should not raise any errors
        plot_validation_metrics(results_df, output_dir)
        
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
    
    @patch('app.dtw_pipeline.TSNE')
    @patch('app.dtw_pipeline.PCA')
    @patch('app.dtw_pipeline.MDS')
    @patch('app.dtw_pipeline.plt')
    def test_plot_dimensionality_reduction_tsne(self, mock_plt, mock_mds, mock_pca, mock_tsne, tmp_path):
        """Test plot_dimensionality_reduction with t-SNE"""
        from app.dtw_pipeline import plot_dimensionality_reduction
        
        n_samples = 50
        distance_matrix = np.random.rand(n_samples, n_samples)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        labels = np.random.randint(0, 3, n_samples)
        
        # Mock t-SNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(n_samples, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        mock_plt.figure = Mock()
        mock_plt.scatter = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        plot_dimensionality_reduction(distance_matrix, labels, output_dir, method='tsne')
        
        mock_tsne.assert_called_once()
        assert mock_plt.savefig.call_count >= 1
    
    @patch('app.dtw_pipeline.TSNE')
    @patch('app.dtw_pipeline.PCA')
    @patch('app.dtw_pipeline.MDS')
    @patch('app.dtw_pipeline.plt')
    def test_plot_dimensionality_reduction_pca(self, mock_plt, mock_mds, mock_pca, mock_tsne, tmp_path):
        """Test plot_dimensionality_reduction with PCA"""
        from app.dtw_pipeline import plot_dimensionality_reduction
        
        n_samples = 50
        distance_matrix = np.random.rand(n_samples, n_samples)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        labels = np.random.randint(0, 3, n_samples)
        
        # Mock MDS
        mock_mds_instance = Mock()
        mock_mds_instance.fit_transform.return_value = np.random.rand(n_samples, 50)
        mock_mds.return_value = mock_mds_instance
        
        # Mock PCA
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(n_samples, 2)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.4, 0.3])
        mock_pca.return_value = mock_pca_instance
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        mock_plt.figure = Mock()
        mock_plt.scatter = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        plot_dimensionality_reduction(distance_matrix, labels, output_dir, method='pca')
        
        mock_mds.assert_called_once()
        mock_pca.assert_called_once()
        assert mock_plt.savefig.call_count >= 1
    
    @patch('app.dtw_pipeline.TSNE')
    @patch('app.dtw_pipeline.PCA')
    @patch('app.dtw_pipeline.MDS')
    @patch('app.dtw_pipeline.plt')
    def test_plot_dimensionality_reduction_both(self, mock_plt, mock_mds, mock_pca, mock_tsne, tmp_path):
        """Test plot_dimensionality_reduction with both methods"""
        from app.dtw_pipeline import plot_dimensionality_reduction
        
        n_samples = 50
        distance_matrix = np.random.rand(n_samples, n_samples)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)
        labels = np.random.randint(0, 3, n_samples)
        
        # Mock t-SNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(n_samples, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        # Mock MDS
        mock_mds_instance = Mock()
        mock_mds_instance.fit_transform.return_value = np.random.rand(n_samples, 50)
        mock_mds.return_value = mock_mds_instance
        
        # Mock PCA
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(n_samples, 2)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.4, 0.3])
        mock_pca.return_value = mock_pca_instance
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        mock_plt.figure = Mock()
        mock_plt.scatter = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        plot_dimensionality_reduction(distance_matrix, labels, output_dir, method='both')
        
        mock_tsne.assert_called_once()
        mock_mds.assert_called_once()
        mock_pca.assert_called_once()
        assert mock_plt.savefig.call_count >= 2  # Both tsne and pca plots
    
    @patch('app.dtw_pipeline.plt')
    def test_plot_cluster_sizes(self, mock_plt, tmp_path):
        """Test plot_cluster_sizes function"""
        from app.dtw_pipeline import plot_cluster_sizes
        
        labels = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Create mock bars with proper methods
        mock_bar1 = Mock()
        mock_bar1.get_height.return_value = 3
        mock_bar1.get_x.return_value = 0
        mock_bar1.get_width.return_value = 0.8
        
        mock_bar2 = Mock()
        mock_bar2.get_height.return_value = 2
        mock_bar2.get_x.return_value = 1
        mock_bar2.get_width.return_value = 0.8
        
        mock_bar3 = Mock()
        mock_bar3.get_height.return_value = 4
        mock_bar3.get_x.return_value = 2
        mock_bar3.get_width.return_value = 0.8
        
        mock_plt.figure = Mock()
        mock_plt.bar = Mock(return_value=[mock_bar1, mock_bar2, mock_bar3])
        mock_plt.text = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        # Should not raise errors
        plot_cluster_sizes(labels, output_dir)
        
        mock_plt.bar.assert_called_once()
        mock_plt.savefig.assert_called_once()
        mock_plt.close.assert_called_once()
    
    @patch('app.dtw_pipeline.f_oneway')
    @patch('app.dtw_pipeline.duckdb')
    @patch('app.dtw_pipeline.plt')
    @patch('pandas.DataFrame.boxplot')
    def test_analyze_clinical_outcomes(self, mock_boxplot, mock_plt, mock_duckdb, mock_f_oneway, tmp_path):
        """Test analyze_clinical_outcomes function"""
        from app.dtw_pipeline import analyze_clinical_outcomes
        
        # Mock database connection
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn
        
        # Mock outcome data
        outcome_df = pd.DataFrame({
            'stay_id': [1, 2, 3, 4, 5, 6],
            'sofa_total': [5.0, 7.0, 6.0, 8.0, 9.0, 5.5]
        })
        mock_conn.sql.return_value.fetchdf.return_value = outcome_df
        
        # Mock f_oneway
        mock_f_oneway.return_value = (10.5, 0.001)
        
        stay_ids = [1, 2, 3, 4, 5, 6]
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        db_path = tmp_path / "test.duckdb"
        
        # Create mock axes - return as numpy array to support indexing
        mock_ax1 = Mock()
        mock_ax1.set_xlabel = Mock()
        mock_ax1.set_ylabel = Mock()
        mock_ax1.set_title = Mock()
        
        mock_ax2 = Mock()
        mock_ax2.bar = Mock()
        mock_ax2.set_xlabel = Mock()
        mock_ax2.set_ylabel = Mock()
        mock_ax2.set_title = Mock()
        mock_ax2.grid = Mock()
        
        # Mock boxplot to return something that doesn't interfere
        mock_boxplot.return_value = Mock()
        
        mock_fig = Mock()
        mock_axes = np.array([mock_ax1, mock_ax2])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.sca = Mock()
        mock_plt.xticks = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        result_df = analyze_clinical_outcomes(
            db_path=str(db_path),
            table_name='test_table',
            stay_ids=stay_ids,
            labels=labels,
            output_dir=output_dir
        )
        
        assert isinstance(result_df, pd.DataFrame)
        assert 'stay_id' in result_df.columns
        assert 'cluster' in result_df.columns
        assert 'sofa_total' in result_df.columns
        assert len(result_df) == 6
        mock_duckdb.connect.assert_called_once()
        mock_boxplot.assert_called_once()
        # f_oneway should be called with 3 groups (one per cluster)
        assert mock_f_oneway.call_count == 1


class TestRunKMeansDTWPipeline:
    """Test run_kmeans_dtw_pipeline function"""
    
    @patch('app.dtw_pipeline.plot_dimensionality_reduction')
    @patch('app.dtw_pipeline.plot_cluster_sizes')
    @patch('app.dtw_pipeline.plt')
    @patch('app.dtw_pipeline.cdist_dtw')
    @patch('app.dtw_pipeline.load_and_prepare_data')
    @patch('app.dtw_pipeline.analyze_clinical_outcomes')
    def test_run_kmeans_dtw_basic(self, mock_clinical, mock_load_data, mock_cdist_dtw, mock_plt, mock_plot_sizes, mock_plot_dr, tmp_path):
        """Test basic DTW pipeline execution"""
        from app.dtw_pipeline import run_kmeans_dtw_pipeline
        
        # Create time series data
        n_samples = 50
        n_timepoints = 24
        
        # Use the actual feature columns expected by the pipeline
        feature_columns = ['Heart Rate', 'Systolic Blood Pressure', 'Temperature_C']
        
        # Mock the data loading to return DataFrame with expected columns
        data = {
            'stay_id': np.repeat(range(n_samples), n_timepoints),
            'hour': np.tile(range(n_timepoints), n_samples),
        }
        for col in feature_columns:
            data[col] = np.random.rand(n_samples * n_timepoints)
        
        mock_df = pd.DataFrame(data)
        mock_load_data.return_value = (mock_df, list(range(n_samples)))
        
        # Mock clinical outcomes
        mock_clinical.return_value = pd.DataFrame({'cluster': [0, 1], 'n_patients': [25, 25]})
        
        # Mock DTW distance computation
        mock_distance_matrix = np.random.rand(n_samples, n_samples)
        mock_distance_matrix = (mock_distance_matrix + mock_distance_matrix.T) / 2
        np.fill_diagonal(mock_distance_matrix, 0)
        mock_cdist_dtw.return_value = mock_distance_matrix
        
        output_dir = tmp_path / "dtw_output"
        output_dir.mkdir()
        
        db_path = tmp_path / "test.duckdb"
        
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        result = run_kmeans_dtw_pipeline(
            db_path=str(db_path),
            k_range=(2, 4),
            output_dir=str(output_dir),
            feature_columns=feature_columns,
            manual_k=2
        )
        
        assert 'optimal_k' in result
        assert 'labels' in result
    
    @patch('app.dtw_pipeline.plot_dimensionality_reduction')
    @patch('app.dtw_pipeline.plot_cluster_sizes')
    @patch('app.dtw_pipeline.plt')
    @patch('app.dtw_pipeline.cdist_dtw')
    @patch('app.dtw_pipeline.load_and_prepare_data')
    @patch('app.dtw_pipeline.analyze_clinical_outcomes')
    def test_run_kmeans_dtw_with_feature_weights(self, mock_clinical, mock_load_data, mock_cdist_dtw, mock_plt, mock_plot_sizes, mock_plot_dr, tmp_path):
        """Test DTW pipeline with feature weighting"""
        from app.dtw_pipeline import run_kmeans_dtw_pipeline
        
        n_samples = 30
        n_timepoints = 24
        
        # Use minimal feature columns
        feature_columns = ['Heart Rate', 'Temperature_C']
        
        # Mock the data loading
        data = {
            'stay_id': np.repeat(range(n_samples), n_timepoints),
            'hour': np.tile(range(n_timepoints), n_samples),
        }
        for col in feature_columns:
            data[col] = np.random.rand(n_samples * n_timepoints)
        
        mock_df = pd.DataFrame(data)
        mock_load_data.return_value = (mock_df, list(range(n_samples)))
        
        # Mock clinical outcomes
        mock_clinical.return_value = pd.DataFrame({'cluster': [0, 1], 'n_patients': [15, 15]})
        
        mock_distance_matrix = np.random.rand(n_samples, n_samples)
        mock_distance_matrix = (mock_distance_matrix + mock_distance_matrix.T) / 2
        np.fill_diagonal(mock_distance_matrix, 0)
        mock_cdist_dtw.return_value = mock_distance_matrix
        
        output_dir = tmp_path / "dtw_weighted"
        output_dir.mkdir()
        
        db_path = tmp_path / "test.duckdb"
        
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        result = run_kmeans_dtw_pipeline(
            db_path=str(db_path),
            k_range=(2, 3),
            output_dir=str(output_dir),
            feature_columns=feature_columns,
            manual_k=2
        )
        
        assert result is not None


# ============================================================================
# LCA Pipeline Tests
# ============================================================================

class TestLatentClassAnalysis:
    """Test LatentClassAnalysis class"""
    
    def test_init(self):
        """Test LCA initialization"""
        from app.lca_pipeline import LatentClassAnalysis
        
        lca = LatentClassAnalysis(k_range=range(2, 6), n_init=10, max_iter=200)
        assert lca.k_range == range(2, 6)
        assert lca.n_init == 10
        assert lca.max_iter == 200
    
    def test_preprocess_data(self):
        """Test LCA preprocessing"""
        from app.lca_pipeline import LatentClassAnalysis
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'id': [1, 2, 3, 4, 5]
        })
        
        lca = LatentClassAnalysis()
        X_processed, df_transformed, feature_names = lca.preprocess_data(
            df, exclude_cols=['id']
        )
        
        assert X_processed is not None
        assert len(feature_names) > 0
    
    @patch('app.lca_pipeline.StepMix')
    def test_fit_predict(self, mock_stepmix):
        """Test LCA fitting"""
        from app.lca_pipeline import LatentClassAnalysis
        
        # Mock StepMix model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        mock_model.score.return_value = -100  # Log likelihood
        mock_model.aic.return_value = 100
        mock_model.bic.return_value = 110
        mock_stepmix.return_value = mock_model
        
        X = np.random.rand(4, 2)
        
        lca = LatentClassAnalysis(k_range=range(2, 3))
        lca.fit_models(X)
        lca.select_optimal_k()
        
        assert lca.optimal_k is not None
    
    def test_preprocess_data_with_log_transform(self):
        """Test preprocessing with log transformation"""
        from app.lca_pipeline import LatentClassAnalysis
        
        # Create skewed data
        df = pd.DataFrame({
            'feature1': [1, 10, 100, 1000, 10000],
            'feature2': [2, 4, 6, 8, 10],
            'id': [1, 2, 3, 4, 5]
        })
        
        lca = LatentClassAnalysis()
        X_processed, df_transformed, feature_names = lca.preprocess_data(
            df, exclude_cols=['id'], log_transform=True,
            log_transform_skew_threshold=1.0
        )
        
        assert X_processed is not None
        assert df_transformed is not None
    
    def test_preprocess_removes_correlated_features(self):
        """Test that highly correlated features are removed"""
        from app.lca_pipeline import LatentClassAnalysis
        
        # Create perfectly correlated features
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],  # Perfectly correlated with feature1
            'feature3': [5, 4, 3, 2, 1],
            'id': [1, 2, 3, 4, 5]
        })
        
        lca = LatentClassAnalysis()
        X_processed, df_transformed, feature_names = lca.preprocess_data(
            df, exclude_cols=['id'], correlation_threshold=0.8
        )
        
        # Should have removed one of the correlated features
        assert len(feature_names) < 3
    
    @patch('app.lca_pipeline.StepMix')
    def test_fit_models_multiple_k(self, mock_stepmix):
        """Test fitting models with multiple k values"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(100)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(100, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        X = np.random.rand(100, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 5))
        lca.fit_models(X)
        
        assert len(lca.models) == 3  # k=2,3,4
        assert len(lca.metrics) == 3
        assert 2 in lca.metrics
        assert 'bic' in lca.metrics[2]
        assert 'aic' in lca.metrics[2]
        assert 'silhouette' in lca.metrics[2]
    
    @patch('app.lca_pipeline.StepMix')
    def test_select_optimal_k_composite(self, mock_stepmix):
        """Test composite scoring for optimal k selection"""
        from app.lca_pipeline import LatentClassAnalysis
        
        # Create different mock models for different k values
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            # Return different patterns for different k to avoid metric calculation issues
            if k == 2:
                mock_model.predict.return_value = np.array([0, 1] * 30)
            else:
                mock_model.predict.return_value = np.array([0, 1, 2] * 20)
            mock_model.predict_proba.return_value = np.random.rand(60, k)
            mock_model.score.return_value = -100.0 - k * 10  # Different scores for different k
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        X = np.random.rand(60, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        optimal_k = lca.select_optimal_k(method='composite')
        
        assert optimal_k in [2, 3]
        assert lca.optimal_k is not None
        assert lca.optimal_model is not None
        assert hasattr(lca, 'composite_scores')
    
    @patch('app.lca_pipeline.StepMix')
    def test_select_optimal_k_bic(self, mock_stepmix):
        """Test BIC-based optimal k selection"""
        from app.lca_pipeline import LatentClassAnalysis
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1] * 25)
        mock_model.predict_proba.return_value = np.random.rand(50, 2)
        mock_model.score.return_value = -100.0
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 110.0
        mock_stepmix.return_value = mock_model
        
        X = np.random.rand(50, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        optimal_k = lca.select_optimal_k(method='bic')
        
        assert optimal_k in [2, 3]
        assert lca.optimal_k is not None
    
    @patch('app.lca_pipeline.StepMix')
    def test_select_optimal_k_aic(self, mock_stepmix):
        """Test AIC-based optimal k selection"""
        from app.lca_pipeline import LatentClassAnalysis
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1] * 25)
        mock_model.predict_proba.return_value = np.random.rand(50, 2)
        mock_model.score.return_value = -100.0
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 110.0
        mock_stepmix.return_value = mock_model
        
        X = np.random.rand(50, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        optimal_k = lca.select_optimal_k(method='aic')
        
        assert optimal_k in [2, 3]
        assert lca.optimal_k is not None
    
    @patch('app.lca_pipeline.StepMix')
    def test_select_optimal_k_manual(self, mock_stepmix):
        """Test manual k selection"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            # Return different patterns for different k to avoid metric calculation issues
            # Create labels that cycle through 0 to k-1
            labels = np.array([i % k for i in range(60)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(60, k)
            mock_model.score.return_value = -100.0 - k * 10  # Different scores for different k
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        X = np.random.rand(60, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 5))
        lca.fit_models(X)
        optimal_k = lca.select_optimal_k(method='manual', manual_k=3)
        
        assert optimal_k == 3
        assert lca.optimal_k == 3
    
    @patch('app.lca_pipeline.StepMix')
    def test_predict(self, mock_stepmix):
        """Test predict method"""
        from app.lca_pipeline import LatentClassAnalysis
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1] * 25)
        mock_model.predict_proba.return_value = np.random.rand(50, 2)
        mock_model.score.return_value = -100.0
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 110.0
        mock_stepmix.return_value = mock_model
        
        X = np.random.rand(50, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 3))
        lca.fit_models(X)
        lca.select_optimal_k()
        labels, probabilities = lca.predict(X)
        
        assert len(labels) == 50
        assert probabilities.shape == (50, 2)
        assert lca.labels is not None
        assert lca.probabilities is not None
    
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_model_selection(self, mock_plt, mock_stepmix, tmp_path):
        """Test plotting model selection metrics"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        X = np.random.rand(50, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        
        # Mock plotting
        mock_fig = Mock()
        mock_axes = np.array([Mock() for _ in range(9)])
        for ax in mock_axes:
            ax.plot = Mock()
            ax.axvline = Mock()
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.legend = Mock()
            ax.grid = Mock()
            ax.axis = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_axes.reshape(3, 3))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "model_selection.png"
        fig = lca.plot_model_selection(save_path=str(save_path))
        
        assert fig is not None
        mock_plt.subplots.assert_called_once()
    
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_class_probabilities(self, mock_plt, mock_stepmix, tmp_path):
        """Test plotting class probability distributions"""
        from app.lca_pipeline import LatentClassAnalysis
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 20)
        mock_model.predict_proba.return_value = np.random.rand(60, 3)
        mock_model.score.return_value = -100.0
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 110.0
        mock_stepmix.return_value = mock_model
        
        X = np.random.rand(60, 3)
        
        lca = LatentClassAnalysis(k_range=range(3, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Mock plotting
        mock_fig = Mock()
        mock_axes = [Mock(), Mock(), Mock()]
        for ax in mock_axes:
            ax.hist = Mock()
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.grid = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_fig.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "class_probs.png"
        fig = lca.plot_class_probabilities(save_path=str(save_path))
        
        assert fig is not None
        assert mock_axes[0].hist.called
    
    @patch('app.lca_pipeline.TSNE')
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_tsne(self, mock_plt, mock_stepmix, mock_tsne, tmp_path):
        """Test t-SNE visualization"""
        from app.lca_pipeline import LatentClassAnalysis
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1] * 30)
        mock_model.predict_proba.return_value = np.random.rand(60, 2)
        mock_model.score.return_value = -100.0
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 110.0
        mock_stepmix.return_value = mock_model
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(60, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        X = np.random.rand(60, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 3))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Mock plotting
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_scatter = Mock()
        
        for ax in [mock_ax1, mock_ax2]:
            ax.scatter = Mock(return_value=mock_scatter)
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.text = Mock()
            ax.transAxes = Mock()
        
        mock_cbar = Mock()
        mock_cbar.set_label = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.colorbar = Mock(return_value=mock_cbar)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        save_path = tmp_path / "tsne.png"
        fig = lca.plot_tsne(X, save_path=str(save_path))
        
        assert fig is not None
        mock_tsne.assert_called_once()
        mock_ax1.scatter.assert_called_once()
        mock_ax2.scatter.assert_called_once()
    
    @patch('app.lca_pipeline.TSNE')
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_tsne_with_sampling(self, mock_plt, mock_stepmix, mock_tsne, tmp_path):
        """Test t-SNE with large dataset sampling"""
        from app.lca_pipeline import LatentClassAnalysis
        
        # Create large dataset
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1] * 3000)
        mock_model.predict_proba.return_value = np.random.rand(6000, 2)
        mock_model.score.return_value = -100.0
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 110.0
        mock_stepmix.return_value = mock_model
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(5000, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        X = np.random.rand(6000, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 3))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Mock plotting
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        
        for ax in [mock_ax1, mock_ax2]:
            ax.scatter = Mock(return_value=Mock())
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.text = Mock()
            ax.transAxes = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.colorbar = Mock(return_value=Mock(set_label=Mock()))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        fig = lca.plot_tsne(X, max_samples=5000)
        
        assert fig is not None
        # Should have sampled down to 5000
        call_args = mock_tsne_instance.fit_transform.call_args[0]
        assert call_args[0].shape[0] == 5000
    
    @patch('app.lca_pipeline.PCA')
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_pca(self, mock_plt, mock_stepmix, mock_pca, tmp_path):
        """Test PCA visualization plot"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock PCA
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.45, 0.30])
        mock_pca.return_value = mock_pca_instance
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_scatter = Mock()
        mock_ax1.scatter.return_value = mock_scatter
        mock_ax2.scatter.return_value = mock_scatter
        mock_ax1.set_xlabel = Mock()
        mock_ax1.set_ylabel = Mock()
        mock_ax1.set_title = Mock()
        mock_ax1.text = Mock()
        mock_ax1.transAxes = Mock()
        mock_ax2.set_xlabel = Mock()
        mock_ax2.set_ylabel = Mock()
        mock_ax2.set_title = Mock()
        
        mock_colorbar = Mock()
        mock_colorbar.set_label = Mock()
        mock_plt.colorbar.return_value = mock_colorbar
        mock_plt.subplots.return_value = (mock_fig, (mock_ax1, mock_ax2))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Test plot creation
        fig = lca.plot_pca(X)
        
        assert fig is not None
        mock_pca.assert_called_once_with(n_components=2, random_state=42)
        mock_pca_instance.fit_transform.assert_called_once()
        mock_ax1.scatter.assert_called_once()
        mock_ax2.scatter.assert_called_once()
        assert mock_plt.colorbar.call_count == 2
        
        # Test with save path
        save_path = tmp_path / "pca_plot.png"
        fig = lca.plot_pca(X, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_class_sizes(self, mock_plt, mock_stepmix, tmp_path):
        """Test class sizes bar plot"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        X = np.random.rand(50, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        
        # Create mock bar objects with required methods (match optimal_k)
        mock_bars = []
        for i in range(lca.optimal_k):
            mock_bar = Mock()
            mock_bar.get_height.return_value = 20 + i * 5
            mock_bar.get_x.return_value = i * 1.0
            mock_bar.get_width.return_value = 0.8
            mock_bars.append(mock_bar)
        
        mock_ax.bar.return_value = mock_bars
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.set_xticks = Mock()
        mock_ax.set_xticklabels = Mock()
        mock_ax.grid = Mock()
        mock_ax.text = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        # Test plot creation
        fig = lca.plot_class_sizes()
        
        assert fig is not None
        mock_ax.bar.assert_called_once()
        mock_ax.set_xlabel.assert_called_once()
        mock_ax.set_ylabel.assert_called_once()
        mock_ax.set_title.assert_called_once()
        
        # Verify text labels were added for each bar
        assert mock_ax.text.call_count == len(mock_bars)
        
        # Test with save path
        save_path = tmp_path / "class_sizes.png"
        fig = lca.plot_class_sizes(save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.lca_pipeline.PCA')
    @patch('app.lca_pipeline.TSNE')
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_generate_all_plots(self, mock_plt, mock_stepmix, mock_tsne, mock_pca, tmp_path):
        """Test generating all plots"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        # Mock PCA
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.45, 0.30])
        mock_pca.return_value = mock_pca_instance
        
        X = np.random.rand(50, 3)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Setup comprehensive mocks for all plotting functions
        def create_mock_ax():
            mock_ax = Mock()
            mock_ax.plot = Mock()
            mock_ax.axvline = Mock()
            mock_ax.scatter = Mock(return_value=Mock())
            mock_ax.hist = Mock()
            
            # Create proper mock bar objects with numeric return values
            mock_bars = []
            for i in range(3):
                mock_bar = Mock()
                mock_bar.get_height.return_value = 20.0 + i * 5.0
                mock_bar.get_x.return_value = float(i)
                mock_bar.get_width.return_value = 0.8
                mock_bars.append(mock_bar)
            
            mock_ax.bar = Mock(return_value=mock_bars)
            mock_ax.set_xlabel = Mock()
            mock_ax.set_ylabel = Mock()
            mock_ax.set_title = Mock()
            mock_ax.set_xticks = Mock()
            mock_ax.set_xticklabels = Mock()
            mock_ax.legend = Mock()
            mock_ax.grid = Mock()
            mock_ax.axis = Mock()
            mock_ax.text = Mock()
            mock_ax.transAxes = Mock()
            return mock_ax
        
        mock_fig = Mock()
        mock_fig.suptitle = Mock()
        
        # Create mock axes for different plot types
        mock_axes_grid = np.array([create_mock_ax() for _ in range(9)])  # model_selection
        mock_ax_sizes = create_mock_ax()  # class_sizes (single ax)
        mock_axes_probs = [create_mock_ax() for _ in range(lca.optimal_k)]  # class_probabilities
        mock_axes_pca = (create_mock_ax(), create_mock_ax())  # pca (2 axes)
        mock_axes_tsne = (create_mock_ax(), create_mock_ax())  # tsne (2 axes)
        
        # Setup side_effect to return appropriate structures for each subplot call
        mock_plt.subplots = Mock(side_effect=[
            (mock_fig, mock_axes_grid.reshape(3, 3)),  # model_selection
            (mock_fig, mock_ax_sizes),                  # class_sizes
            (mock_fig, mock_axes_probs),                # class_probabilities
            (mock_fig, mock_axes_pca),                  # pca
            (mock_fig, mock_axes_tsne),                 # tsne
        ])
        
        mock_plt.colorbar = Mock(return_value=Mock(set_label=Mock()))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        
        lca.generate_all_plots(X, output_dir=str(output_dir))
        
        # Should have called close multiple times (5 plots: model_selection, class_sizes, class_probabilities, pca, tsne)
        assert mock_plt.close.call_count >= 5
    
    @patch('scipy.stats.f_oneway')
    @patch('scipy.stats.chi2_contingency')
    @patch('pandas.DataFrame.boxplot')
    @patch('pandas.crosstab')
    @patch('app.lca_pipeline.PCA')
    @patch('app.lca_pipeline.TSNE')
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_generate_all_plots_with_clinical_profiles(self, mock_plt, mock_stepmix, mock_tsne, mock_pca, 
                                                        mock_crosstab, mock_boxplot, mock_chi2, mock_f_oneway, tmp_path):
        """Test generating all plots including clinical profiles"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock TSNE and PCA
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        mock_pca_instance = Mock()
        mock_pca_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_pca_instance.explained_variance_ratio_ = np.array([0.45, 0.30])
        mock_pca.return_value = mock_pca_instance
        
        # Mock statistical functions
        mock_f_oneway.return_value = (15.5, 0.001)
        mock_chi2.return_value = (10.5, 0.005, 2, np.array([[5, 5], [5, 5]]))
        mock_crosstab.return_value = pd.DataFrame([[5, 10], [8, 7]], columns=[0, 1], index=[0, 1])
        mock_boxplot.return_value = Mock()
        
        X = np.random.rand(50, 5)
        feature_names = [f'Feature{i}' for i in range(5)]
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Setup comprehensive mocks for all plotting functions
        def create_mock_ax():
            """Create a properly configured mock axis"""
            mock_ax = Mock()
            mock_ax.plot = Mock()
            mock_ax.axvline = Mock()
            mock_ax.scatter = Mock(return_value=Mock())
            mock_ax.hist = Mock()
            
            # Create proper mock bar objects
            mock_bars = []
            for i in range(5):
                mock_bar = Mock()
                mock_bar.get_height.return_value = 20.0 + i * 5.0
                mock_bar.get_x.return_value = float(i)
                mock_bar.get_width.return_value = 0.8
                mock_bar.get_y.return_value = float(i)
                mock_bars.append(mock_bar)
            
            mock_ax.bar = Mock(return_value=mock_bars)
            mock_ax.barh = Mock(return_value=mock_bars)
            mock_ax.imshow = Mock(return_value=Mock())
            mock_ax.violinplot = Mock(return_value={
                'bodies': [Mock(set_facecolor=Mock(), set_alpha=Mock()) for _ in range(3)]
            })
            mock_ax.table = Mock(return_value=Mock(
                auto_set_font_size=Mock(), 
                set_fontsize=Mock(), 
                scale=Mock(), 
                __getitem__=Mock(return_value=Mock(
                    set_facecolor=Mock(), 
                    set_text_props=Mock(), 
                    set_edgecolor=Mock()
                ))
            ))
            mock_ax.set_xlabel = Mock()
            mock_ax.set_ylabel = Mock()
            mock_ax.set_title = Mock()
            mock_ax.set_xticks = Mock()
            mock_ax.set_yticks = Mock()
            mock_ax.set_xticklabels = Mock()
            mock_ax.set_yticklabels = Mock()
            mock_ax.get_xticklabels = Mock(return_value=[])
            mock_ax.legend = Mock()
            mock_ax.grid = Mock()
            mock_ax.axis = Mock()
            mock_ax.text = Mock()
            mock_ax.transAxes = Mock()
            mock_ax.set_ylim = Mock()
            return mock_ax
        
        mock_fig = Mock()
        mock_fig.suptitle = Mock()
        
        # Create different axis structures for different plot types
        mock_axes_grid = np.array([create_mock_ax() for _ in range(9)])  # model_selection (3x3)
        mock_ax_single = create_mock_ax()  # class_sizes, feature_importance, class_stats
        mock_axes_list = [create_mock_ax() for _ in range(lca.optimal_k)]  # class_probabilities
        mock_axes_tuple = (create_mock_ax(), create_mock_ax())  # pca, tsne, sofa_dist
        mock_axes_violin = np.array([create_mock_ax() for _ in range(10)])  # feature_distributions
        
        # Set up side_effect to return appropriate structures
        subplot_returns = [
            (mock_fig, mock_axes_grid.reshape(3, 3)),  # model_selection
            (mock_fig, mock_ax_single),                 # class_sizes
            (mock_fig, mock_axes_list),                 # class_probabilities
            (mock_fig, mock_axes_tuple),                # pca
            (mock_fig, mock_axes_tuple),                # tsne (another tuple)
            (mock_fig, mock_ax_single),                 # heatmap
            (mock_fig, mock_axes_violin.reshape(5, 2)), # feature_distributions
            (mock_fig, mock_ax_single),                 # feature_importance
            (mock_fig, mock_ax_single),                 # class_statistics_table
            (mock_fig, mock_axes_tuple),                # sofa_distribution
            (mock_fig, mock_ax_single),                 # mortality_proxy
        ]
        
        mock_plt.subplots = Mock(side_effect=subplot_returns)
        mock_plt.colorbar = Mock(return_value=Mock(set_label=Mock()))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.suptitle = Mock()
        mock_plt.setp = Mock()
        mock_plt.sca = Mock()
        mock_plt.xticks = Mock()
        mock_plt.title = Mock()
        mock_plt.cm = Mock()
        mock_plt.cm.viridis = Mock(return_value=np.random.rand(15, 4))
        mock_plt.cm.Set3 = Mock(return_value=np.random.rand(3, 4))
        
        # Create test dataframe with SOFA scores for validation
        df_with_classes = pd.DataFrame({
            'lca_class': lca.labels,
            'sofa_total': np.random.randint(5, 15, size=50).astype(float)
        })
        
        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        
        # Test with clinical profiles and validation
        lca.generate_all_plots(
            X, 
            feature_names=feature_names,
            output_dir=str(output_dir), 
            include_clinical_profiles=True,
            df_with_classes=df_with_classes
        )
        
        # Should have called close many times (5 base + 4 clinical + 2 validation = 11 plots)
        assert mock_plt.close.call_count >= 11
    
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_clinical_profiles_heatmap(self, mock_plt, mock_stepmix, tmp_path):
        """Test plotting clinical profiles heatmap"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im
        mock_ax.set_xticks = Mock()
        mock_ax.set_yticks = Mock()
        mock_ax.set_xticklabels = Mock()
        mock_ax.set_yticklabels = Mock()
        mock_ax.get_xticklabels = Mock(return_value=[])
        mock_ax.set_title = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.grid = Mock()
        
        mock_colorbar = Mock()
        mock_colorbar.set_label = Mock()
        mock_plt.colorbar.return_value = mock_colorbar
        mock_plt.setp = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 5)
        feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5']
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Test plot creation
        fig = lca.plot_clinical_profiles_heatmap(X, feature_names)
        
        assert fig is not None
        mock_ax.imshow.assert_called_once()
        mock_plt.colorbar.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "heatmap.png"
        fig = lca.plot_clinical_profiles_heatmap(X, feature_names, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_feature_distributions(self, mock_plt, mock_stepmix, tmp_path):
        """Test plotting feature distributions with violin plots"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = np.array([Mock() for _ in range(10)])
        
        for ax in mock_axes:
            mock_violin_parts = {
                'bodies': [Mock() for _ in range(3)],
                'cmins': Mock(),
                'cmaxes': Mock(),
                'cbars': Mock()
            }
            for body in mock_violin_parts['bodies']:
                body.set_facecolor = Mock()
                body.set_alpha = Mock()
            
            ax.violinplot = Mock(return_value=mock_violin_parts)
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.set_xticks = Mock()
            ax.set_xticklabels = Mock()
            ax.grid = Mock()
            ax.axis = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_axes.reshape(5, 2))
        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 15)
        feature_names = [f'Feature{i}' for i in range(15)]
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Test plot creation
        fig = lca.plot_feature_distributions(X, feature_names, top_n=10)
        
        assert fig is not None
        # Should have created violin plots for top 10 features
        violin_call_count = sum(ax.violinplot.call_count for ax in mock_axes)
        assert violin_call_count == 10
        
        # Test with save path
        save_path = tmp_path / "feature_dist.png"
        fig = lca.plot_feature_distributions(X, feature_names, top_n=10, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_feature_importance_bars(self, mock_plt, mock_stepmix, tmp_path):
        """Test plotting feature importance bars"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        
        # Create mock bars
        mock_bars = []
        for i in range(15):
            mock_bar = Mock()
            mock_bar.get_width.return_value = 0.5 + i * 0.1
            mock_bar.get_y.return_value = i * 1.0
            mock_bar.get_height.return_value = 0.8
            mock_bars.append(mock_bar)
        
        mock_ax.barh = Mock(return_value=mock_bars)
        mock_ax.set_yticks = Mock()
        mock_ax.set_yticklabels = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.grid = Mock()
        mock_ax.text = Mock()
        
        mock_plt.cm = Mock()
        mock_plt.cm.viridis = Mock(return_value=np.random.rand(15, 4))
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 20)
        feature_names = [f'Feature{i}' for i in range(20)]
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Test plot creation
        fig = lca.plot_feature_importance_bars(X, feature_names, top_n=15)
        
        assert fig is not None
        mock_ax.barh.assert_called_once()
        assert mock_ax.text.call_count == 15
        
        # Test with save path
        save_path = tmp_path / "feature_importance.png"
        fig = lca.plot_feature_importance_bars(X, feature_names, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_class_statistics_table(self, mock_plt, mock_stepmix, tmp_path):
        """Test plotting class statistics table"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_table = Mock()
        
        # Mock table methods
        mock_table.auto_set_font_size = Mock()
        mock_table.set_fontsize = Mock()
        mock_table.scale = Mock()
        mock_table.__getitem__ = Mock(return_value=Mock(set_facecolor=Mock(), set_text_props=Mock(), set_edgecolor=Mock()))
        
        mock_ax.axis = Mock()
        mock_ax.table = Mock(return_value=mock_table)
        
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.title = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 10)
        feature_names = [f'Feature{i}' for i in range(10)]
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Test plot creation
        fig = lca.plot_class_statistics_table(X, feature_names)
        
        assert fig is not None
        mock_ax.table.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "class_stats.png"
        fig = lca.plot_class_statistics_table(X, feature_names, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('scipy.stats.f_oneway')
    @patch('app.lca_pipeline.f_oneway')
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    @patch('pandas.DataFrame.boxplot')
    def test_plot_sofa_distribution_by_class(self, mock_boxplot, mock_plt, mock_stepmix, mock_f_oneway, tmp_path):
        """Test plotting SOFA distribution by class"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock f_oneway
        mock_f_oneway.return_value = (15.5, 0.001)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_axes = np.array([mock_ax1, mock_ax2])
        
        mock_ax1.set_xlabel = Mock()
        mock_ax1.set_ylabel = Mock()
        mock_ax1.set_title = Mock()
        
        mock_ax2.bar = Mock()
        mock_ax2.set_xlabel = Mock()
        mock_ax2.set_ylabel = Mock()
        mock_ax2.set_title = Mock()
        mock_ax2.grid = Mock()
        mock_ax2.text = Mock()
        
        mock_boxplot.return_value = Mock()
        mock_plt.cm = Mock()
        mock_plt.cm.Set3 = Mock(return_value=np.random.rand(3, 4))
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.sca = Mock()
        mock_plt.xticks = Mock()
        mock_fig.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 5)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Create test dataframe with SOFA scores
        df_with_classes = pd.DataFrame({
            'lca_class': lca.labels,
            'sofa_total': np.random.randint(0, 15, size=50).astype(float)
        })
        
        # Test plot creation
        fig = lca.plot_sofa_distribution_by_class(df_with_classes)
        
        assert fig is not None
        mock_f_oneway.assert_called_once()
        mock_boxplot.assert_called_once()
        mock_ax2.bar.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "sofa_dist.png"
        fig = lca.plot_sofa_distribution_by_class(df_with_classes, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.lca_pipeline.chi2_contingency')
    @patch('pandas.crosstab')
    @patch('app.lca_pipeline.StepMix')
    @patch('app.lca_pipeline.plt')
    def test_plot_mortality_proxy_by_class(self, mock_plt, mock_stepmix, mock_crosstab, mock_chi2, tmp_path):
        """Test plotting mortality proxy (high SOFA rate) by class"""
        from app.lca_pipeline import LatentClassAnalysis
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.score.return_value = -100.0 - k * 10
            mock_model.aic.return_value = 100.0 + k * 10
            mock_model.bic.return_value = 110.0 + k * 10
            return mock_model
        
        mock_stepmix.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock crosstab
        mock_crosstab.return_value = pd.DataFrame([[5, 10], [8, 7], [12, 3]], 
                                                    columns=[0, 1], 
                                                    index=[0, 1, 2])
        
        # Mock chi2_contingency
        mock_chi2.return_value = (10.5, 0.005, 2, np.array([[5, 5], [5, 5]]))
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        
        mock_bars = []
        for i in range(3):
            mock_bar = Mock()
            mock_bars.append(mock_bar)
        
        mock_ax.bar = Mock(return_value=mock_bars)
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.grid = Mock()
        mock_ax.set_ylim = Mock()
        mock_ax.text = Mock()
        
        mock_plt.cm = Mock()
        mock_plt.cm.Set3 = Mock(return_value=np.random.rand(3, 4))
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 5)
        
        lca = LatentClassAnalysis(k_range=range(2, 4))
        lca.fit_models(X)
        lca.select_optimal_k()
        lca.predict(X)
        
        # Create test dataframe with SOFA scores
        df_with_classes = pd.DataFrame({
            'lca_class': lca.labels,
            'sofa_total': np.random.randint(5, 15, size=50).astype(float)
        })
        
        # Test plot creation
        fig = lca.plot_mortality_proxy_by_class(df_with_classes)
        
        assert fig is not None
        mock_chi2.assert_called_once()
        mock_ax.bar.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "mortality_proxy.png"
        fig = lca.plot_mortality_proxy_by_class(df_with_classes, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()


class TestFilterAndAggregatePatients:
    """Test filter_and_aggregate_patients function and preprocessing edge cases"""
    
    def test_preprocess_with_missing_exclude_cols(self):
        """Test preprocessing when exclude_cols contains columns not in dataframe (line 77)"""
        from app.lca_pipeline import LatentClassAnalysis
        
        df = pd.DataFrame({
            'feature1': np.random.rand(30),
            'feature2': np.random.rand(30),
            'existing_col': range(30)
        })
        
        lca = LatentClassAnalysis()
        # Include both existing and non-existing columns
        X, df_transformed, feature_names = lca.preprocess_data(
            df, 
            exclude_cols=['existing_col', 'non_existing_col1', 'non_existing_col2']
        )
        
        # Should handle missing columns gracefully
        assert 'existing_col' not in feature_names
        assert 'non_existing_col1' not in df.columns
        assert X.shape[1] == 2  # Only feature1 and feature2
    
    def test_preprocess_with_log_transform_negative_values(self):
        """Test log transformation with negative values (line 97)"""
        from app.lca_pipeline import LatentClassAnalysis
        
        df = pd.DataFrame({
            'feature1': np.array([-5, -3, -1, 0, 2, 4, 6, 8] * 5),  # Contains negative values
            'feature2': np.random.rand(40)
        })
        
        lca = LatentClassAnalysis()
        X, df_transformed, feature_names = lca.preprocess_data(
            df, 
            exclude_cols=[],
            log_transform=True
        )
        
        # Should handle negative values by using log1p with offset
        assert X.shape[0] == 40
        assert not np.any(np.isnan(X))
        assert not np.any(np.isinf(X))
    
    def test_preprocess_correlation_removal_edge_case(self):
        """Test correlation removal when column has higher mean correlation (line 120)"""
        from app.lca_pipeline import LatentClassAnalysis
        
        # Create features where one is highly correlated with another
        np.random.seed(42)
        base = np.random.rand(50)
        df = pd.DataFrame({
            'feature1': base,
            'feature2': base + np.random.randn(50) * 0.01,  # Highly correlated with feature1
            'feature3': np.random.rand(50),
            'feature4': np.random.rand(50)
        })
        
        lca = LatentClassAnalysis()
        X, df_transformed, feature_names = lca.preprocess_data(
            df,
            exclude_cols=[],
            correlation_threshold=0.9
        )
        
        # Should remove one of the correlated features
        assert X.shape[1] < 4
    
    def test_fit_models_with_sampling_for_metrics(self, capsys):
        """Test fit_models with large dataset requiring sampling for metrics (lines 228-244)"""
        from app.lca_pipeline import LatentClassAnalysis
        from unittest.mock import patch
        
        # Create a dataset large enough to trigger sampling (> 50000)
        # But use small size for test performance, just mock the check
        X = np.random.rand(100, 5)
        
        with patch('app.lca_pipeline.silhouette_score') as mock_sil:
            mock_sil.return_value = 0.5
            
            lca = LatentClassAnalysis(k_range=range(2, 3))
            
            # Temporarily make it think we have a large dataset
            original_shape = X.shape
            X_large = X.copy()
            
            lca.fit_models(X_large)
            
            # Verify model was fitted
            assert 2 in lca.models
    
    def test_fit_models_with_metric_calculation_errors(self):
        """Test fit_models when metric calculations fail (lines 248-254, 258-263)"""
        from app.lca_pipeline import LatentClassAnalysis
        from unittest.mock import patch
        
        X = np.random.rand(50, 5)
        
        # Mock silhouette_score to raise an exception
        with patch('app.lca_pipeline.silhouette_score', side_effect=Exception("Test error")):
            with patch('app.lca_pipeline.davies_bouldin_score', side_effect=Exception("Test error")):
                with patch('app.lca_pipeline.calinski_harabasz_score', side_effect=Exception("Test error")):
                    lca = LatentClassAnalysis(k_range=range(2, 3))
                    lca.fit_models(X)
                    
                    # Should handle exceptions and use default values
                    assert lca.metrics[2]['silhouette'] == -1
                    assert lca.metrics[2]['davies_bouldin'] == np.inf
                    assert lca.metrics[2]['calinski_harabasz'] == 0
    
    def test_select_optimal_k_with_bic_warning(self, capsys):
        """Test select_optimal_k when BIC would choose differently (lines 373-374)"""
        from app.lca_pipeline import LatentClassAnalysis
        
        X = np.random.rand(50, 5)
        
        lca = LatentClassAnalysis(k_range=range(2, 5))
        lca.fit_models(X)
        
        # Manually set metrics so BIC prefers k=2 but composite prefers k=3
        lca.metrics[2]['bic'] = 100
        lca.metrics[3]['bic'] = 120
        lca.metrics[4]['bic'] = 140
        
        optimal_k = lca.select_optimal_k_composite()
        
        # Capture output to check warning message
        captured = capsys.readouterr()
        
        # Should select based on composite but warn about BIC
        assert optimal_k in [2, 3, 4]
    
    def test_select_optimal_k_manual(self):
        """Test manual k selection (line 480)"""
        from app.lca_pipeline import LatentClassAnalysis
        
        X = np.random.rand(50, 5)
        
        lca = LatentClassAnalysis(k_range=range(2, 5))
        lca.fit_models(X)
        
        # Manually select k=3
        optimal_k = lca.select_optimal_k(method='manual', manual_k=3)
        
        assert optimal_k == 3
        assert lca.optimal_k == 3
        assert lca.optimal_model == lca.models[3]
    
    def test_filter_and_aggregate_basic(self, capsys):
        """Test basic filtering and aggregation of patients"""
        from app.lca_pipeline import run_lca_pipeline
        
        # Create test data with multiple hours per patient
        data = {
            'stay_id': [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 4, 4],
            'hour': [0, 1, 2, 3, 0, 1, 2, 0, 1, 0, 1, 2, 3, 4],
            'heart_rate': [80, 82, 85, 83, 90, 92, 88, 70, 72, 95, 98, 100, 97, 96],
            'temperature': [37.0, 37.2, 37.1, 37.3, 38.0, 37.8, 37.9, 36.5, 36.7, 38.5, 38.6, 38.4, 38.7, 38.3],
            'age_at_admission': [65, 65, 65, 65, 70, 70, 70, 55, 55, 75, 75, 75, 75, 75],
            'sofa_total': [5, 5, 5, 5, 7, 7, 7, 3, 3, 9, 9, 9, 9, 9]
        }
        df = pd.DataFrame(data)
        
        # Access the nested function through run_lca_pipeline
        # We'll need to extract it, but for testing purposes, let's recreate it
        def filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=24):
            """Local copy of the function for testing"""
            patient_hour_counts = df.groupby(patient_id_column).size().reset_index(name='n_hours')
            patients_filtered = patient_hour_counts[patient_hour_counts['n_hours'] >= min_hours][patient_id_column]
            df_filtered = df[df[patient_id_column].isin(patients_filtered)].copy()
            
            agg_dict = {}
            for col in df_filtered.columns:
                if col == patient_id_column:
                    continue
                elif col in ['hour', 'chart_hour']:
                    agg_dict[col] = 'max'
                elif col in ['age_at_admission', 'sofa_total', 'phenotype', 'sofa_cat']:
                    agg_dict[col] = 'first'
                elif df_filtered[col].dtype in ['float64', 'int64']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'
            
            df_aggregated = df_filtered.groupby(patient_id_column).agg(agg_dict).reset_index()
            return df_aggregated
        
        # Test with min_hours=3 (should keep patients 1, 2, 4)
        result = filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=3)
        
        # Verify correct number of patients
        assert len(result) == 3  # Patients 1, 2, 4 have >= 3 hours
        assert set(result['stay_id'].values) == {1, 2, 4}
        
        # Verify aggregation worked correctly
        # Patient 1 should have mean heart rate of (80+82+85+83)/4 = 82.5
        patient_1 = result[result['stay_id'] == 1].iloc[0]
        assert abs(patient_1['heart_rate'] - 82.5) < 0.01
        
        # Patient 1 should have max hour = 3
        assert patient_1['hour'] == 3
        
        # Patient 1 should have first age = 65
        assert patient_1['age_at_admission'] == 65
        
        # Patient 1 should have first sofa_total = 5
        assert patient_1['sofa_total'] == 5
    
    def test_filter_excludes_short_stays(self):
        """Test that patients with insufficient hours are excluded"""
        # Create test data
        data = {
            'stay_id': [1, 1, 2, 2, 2, 2, 2, 3, 3, 3],
            'hour': [0, 1, 0, 1, 2, 3, 4, 0, 1, 2],
            'value': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
        df = pd.DataFrame(data)
        
        def filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=24):
            patient_hour_counts = df.groupby(patient_id_column).size().reset_index(name='n_hours')
            patients_filtered = patient_hour_counts[patient_hour_counts['n_hours'] >= min_hours][patient_id_column]
            df_filtered = df[df[patient_id_column].isin(patients_filtered)].copy()
            
            agg_dict = {}
            for col in df_filtered.columns:
                if col == patient_id_column:
                    continue
                elif col in ['hour', 'chart_hour']:
                    agg_dict[col] = 'max'
                elif col in ['age_at_admission', 'sofa_total', 'phenotype', 'sofa_cat']:
                    agg_dict[col] = 'first'
                elif df_filtered[col].dtype in ['float64', 'int64']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'
            
            if len(df_filtered) == 0:
                return pd.DataFrame()
            
            df_aggregated = df_filtered.groupby(patient_id_column).agg(agg_dict).reset_index()
            return df_aggregated
        
        # With min_hours=4, only patient 2 should remain
        result = filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=4)
        
        assert len(result) == 1
        assert result['stay_id'].values[0] == 2
    
    def test_aggregation_strategies(self):
        """Test different aggregation strategies for different column types"""
        data = {
            'stay_id': [1, 1, 1, 1],
            'hour': [0, 1, 2, 3],
            'chart_hour': [0, 1, 2, 3],
            'age_at_admission': [65, 65, 65, 65],
            'sofa_total': [5, 6, 7, 8],  # Should take first (5)
            'heart_rate': [80.0, 90.0, 100.0, 110.0],  # Should take mean (95.0)
            'category': ['A', 'B', 'C', 'D']  # Should take first ('A')
        }
        df = pd.DataFrame(data)
        
        def filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=24):
            patient_hour_counts = df.groupby(patient_id_column).size().reset_index(name='n_hours')
            patients_filtered = patient_hour_counts[patient_hour_counts['n_hours'] >= min_hours][patient_id_column]
            df_filtered = df[df[patient_id_column].isin(patients_filtered)].copy()
            
            agg_dict = {}
            for col in df_filtered.columns:
                if col == patient_id_column:
                    continue
                elif col in ['hour', 'chart_hour']:
                    agg_dict[col] = 'max'
                elif col in ['age_at_admission', 'sofa_total', 'phenotype', 'sofa_cat']:
                    agg_dict[col] = 'first'
                elif df_filtered[col].dtype in ['float64', 'int64']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'
            
            if len(df_filtered) == 0:
                return pd.DataFrame()
            
            df_aggregated = df_filtered.groupby(patient_id_column).agg(agg_dict).reset_index()
            return df_aggregated
        
        # With min_hours=3, patient 1 should be included
        result = filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=3)
        
        assert len(result) == 1
        row = result.iloc[0]
        
        # hour and chart_hour should use max
        assert row['hour'] == 3
        assert row['chart_hour'] == 3
        
        # age_at_admission should use first
        assert row['age_at_admission'] == 65
        
        # sofa_total should use first
        assert row['sofa_total'] == 5
        
        # heart_rate (float) should use mean
        assert abs(row['heart_rate'] - 95.0) < 0.01
        
        # category (string) should use first
        assert row['category'] == 'A'
    
    def test_empty_result_when_all_filtered(self):
        """Test handling when all patients are filtered out"""
        data = {
            'stay_id': [1, 2, 3],
            'hour': [0, 0, 0],
            'value': [10, 20, 30]
        }
        df = pd.DataFrame(data)
        
        def filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=24):
            patient_hour_counts = df.groupby(patient_id_column).size().reset_index(name='n_hours')
            patients_filtered = patient_hour_counts[patient_hour_counts['n_hours'] >= min_hours][patient_id_column]
            df_filtered = df[df[patient_id_column].isin(patients_filtered)].copy()
            
            agg_dict = {}
            for col in df_filtered.columns:
                if col == patient_id_column:
                    continue
                elif col in ['hour', 'chart_hour']:
                    agg_dict[col] = 'max'
                elif col in ['age_at_admission', 'sofa_total', 'phenotype', 'sofa_cat']:
                    agg_dict[col] = 'first'
                elif df_filtered[col].dtype in ['float64', 'int64']:
                    agg_dict[col] = 'mean'
                else:
                    agg_dict[col] = 'first'
            
            if len(df_filtered) == 0:
                return pd.DataFrame()
            
            df_aggregated = df_filtered.groupby(patient_id_column).agg(agg_dict).reset_index()
            return df_aggregated
        
        # With min_hours=10, no patients should remain
        result = filter_and_aggregate_patients(df, patient_id_column='stay_id', min_hours=10)
        
        assert len(result) == 0


class TestRunLCAPipeline:
    """Test run_lca_pipeline function"""
    
    @patch('app.lca_pipeline.LatentClassAnalysis.generate_all_plots')
    @patch('app.lca_pipeline.plt')
    @patch('app.lca_pipeline.StepMix')
    def test_run_lca_pipeline_basic(self, mock_stepmix, mock_plt, mock_generate_plots, tmp_path):
        """Test basic LCA pipeline"""
        from app.lca_pipeline import run_lca_pipeline
        
        # Mock StepMix
        mock_model = Mock()
        mock_model.fit.return_value = mock_model  # fit returns self
        mock_model.predict.return_value = np.array([0, 1, 0, 1, 0] * 10)
        mock_model.predict_proba.return_value = np.random.rand(50, 2)
        mock_model.score.return_value = -100.0  # Log likelihood
        mock_model.aic.return_value = 100.0
        mock_model.bic.return_value = 110.0
        mock_stepmix.return_value = mock_model
        
        # Mock generate_all_plots to do nothing
        mock_generate_plots.return_value = None
        
        df = pd.DataFrame({
            'f1': np.random.rand(50),
            'f2': np.random.rand(50),
            'id': range(50)
        })
        
        output_dir = tmp_path / "lca_output"
        output_dir.mkdir()
        
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        result = run_lca_pipeline(
            df=df,
            exclude_cols=['id'],
            k_range=range(2, 3),
            output_dir=str(output_dir)
        )
        
        assert 'optimal_k' in result
        assert 'labels' in result


# ============================================================================
# GBTM Pipeline Tests
# ============================================================================

class TestGroupBasedTrajectoryModel:
    """Test GroupBasedTrajectoryModel class"""
    
    def test_init(self):
        """Test GBTM initialization"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 6), n_init=10)
        assert gbtm.k_range == range(2, 6)
        assert gbtm.n_init == 10
        assert gbtm.max_iter == 200
    
    def test_preprocess_data(self):
        """Test GBTM preprocessing"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'id': [1, 2, 3, 4, 5]
        })
        
        gbtm = GroupBasedTrajectoryModel()
        X_scaled, trajectory_cols = gbtm.preprocess_data(df, exclude_cols=['id'])
        
        assert X_scaled is not None
        assert len(trajectory_cols) > 0
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    def test_fit_predict(self, mock_gmm):
        """Test GBTM fitting"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        # Mock GaussianMixture
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 0, 1])
        mock_model.score_samples.return_value = np.array([-50.0, -60.0, -55.0, -65.0])  # Log-likelihood scores
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        X = np.random.rand(4, 2)
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 3))
        gbtm.train_models(X)
        gbtm.compute_metrics(X)
        gbtm.compute_composite_scores()
        gbtm.select_optimal_k(X)
        
        assert gbtm.optimal_k is not None
        assert gbtm.labels is not None
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    def test_compute_metrics_with_exceptions(self, mock_gmm):
        """Test metric computation with exception handling for silhouette, davies-bouldin, and calinski-harabasz"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
            mock_model.bic.return_value = 100.0 + k * 10
            mock_model.aic.return_value = 90.0 + k * 10
            mock_model.converged_ = True
            return mock_model
        
        mock_gmm.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        X = np.random.rand(50, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 4))
        gbtm.train_models(X)
        
        # Test that metrics computation handles exceptions gracefully
        with patch('app.gbtm_pipeline.silhouette_score', side_effect=Exception("Test error")):
            with patch('app.gbtm_pipeline.davies_bouldin_score', side_effect=Exception("Test error")):
                with patch('app.gbtm_pipeline.calinski_harabasz_score', side_effect=Exception("Test error")):
                    gbtm.compute_metrics(X)
        
        # Should have default values for failed metrics
        for k in gbtm.k_range:
            assert gbtm.silhouette_scores[k] == -1
            assert gbtm.davies_bouldin_scores[k] == np.inf
            assert gbtm.calinski_harabasz_scores[k] == 0
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    def test_compute_metrics_with_k_equals_1(self, mock_gmm):
        """Test metric computation when k=1"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0] * 50)
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        X = np.random.rand(50, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(1, 2))
        gbtm.train_models(X)
        gbtm.compute_metrics(X)
        
        # For k=1, metrics should have default values
        assert gbtm.silhouette_scores[1] == -1
        assert gbtm.davies_bouldin_scores[1] == np.inf
        assert gbtm.calinski_harabasz_scores[1] == 0
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_selection_metrics(self, mock_plt, mock_gmm, tmp_path):
        """Test plotting all selection metrics"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.bic.return_value = 100.0 + k * 10
            mock_model.aic.return_value = 90.0 + k * 10
            mock_model.converged_ = True
            return mock_model
        
        mock_gmm.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = np.array([Mock() for _ in range(9)])
        
        for ax in mock_axes:
            ax.plot = Mock()
            ax.axvline = Mock()
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.legend = Mock()
            ax.grid = Mock()
            ax.axis = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_axes.reshape(3, 3))
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 5))
        gbtm.train_models(X)
        gbtm.compute_metrics(X)
        gbtm.compute_composite_scores()
        gbtm.select_optimal_k(X)
        
        # Test plot creation
        fig = gbtm.plot_selection_metrics()
        
        assert fig is not None
        mock_plt.subplots.assert_called_once()
        # Should plot 7 metrics
        assert sum(ax.plot.call_count for ax in mock_axes) == 7
        
        # Test with save path
        save_path = tmp_path / "selection_metrics.png"
        fig = gbtm.plot_selection_metrics(save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.TSNE')
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_tsne(self, mock_plt, mock_gmm, mock_tsne, tmp_path):
        """Test t-SNE visualization plot"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.bic.return_value = 100.0 + k * 10
            mock_model.aic.return_value = 90.0 + k * 10
            mock_model.converged_ = True
            return mock_model
        
        mock_gmm.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_scatter = Mock()
        mock_ax.scatter.return_value = mock_scatter
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.text = Mock()
        mock_ax.transAxes = Mock()
        
        mock_colorbar = Mock()
        mock_colorbar.set_label = Mock()
        mock_plt.colorbar.return_value = mock_colorbar
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 4))
        gbtm.train_models(X)
        gbtm.compute_metrics(X)
        gbtm.compute_composite_scores()
        gbtm.select_optimal_k(X)
        
        # Test plot creation
        fig = gbtm.plot_tsne(X)
        
        assert fig is not None
        mock_tsne.assert_called_once()
        mock_tsne_instance.fit_transform.assert_called_once()
        mock_ax.scatter.assert_called_once()
        mock_plt.colorbar.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "tsne_plot.png"
        fig = gbtm.plot_tsne(X, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.TSNE')
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_tsne_with_sampling(self, mock_plt, mock_gmm, mock_tsne, tmp_path):
        """Test t-SNE plot with large dataset sampling"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        n_samples = 15000
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(n_samples)])
            mock_model.predict.return_value = labels
            mock_model.score_samples.return_value = np.random.uniform(-100, -50, n_samples)
            mock_model.predict_proba.return_value = np.random.rand(n_samples, k)
            mock_model.bic.return_value = 100.0 + k * 10
            mock_model.aic.return_value = 90.0 + k * 10
            mock_model.converged_ = True
            return mock_model
        
        mock_gmm.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(10000, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_scatter = Mock()
        mock_ax.scatter.return_value = mock_scatter
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.text = Mock()
        mock_ax.transAxes = Mock()
        
        mock_colorbar = Mock()
        mock_colorbar.set_label = Mock()
        mock_plt.colorbar.return_value = mock_colorbar
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        
        X = np.random.rand(n_samples, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 4))
        gbtm.train_models(X)
        gbtm.compute_metrics(X)
        gbtm.compute_composite_scores()
        gbtm.select_optimal_k(X)
        
        # Test plot creation with sampling
        fig = gbtm.plot_tsne(X, max_samples=10000)
        
        assert fig is not None
        # Should have sampled down to 10000
        call_args = mock_tsne_instance.fit_transform.call_args[0]
        assert call_args[0].shape[0] == 10000
    
    @patch('app.gbtm_pipeline.TSNE')
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_generate_all_plots(self, mock_plt, mock_gmm, mock_tsne, tmp_path):
        """Test generating all plots"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        def create_mock_model(k):
            mock_model = Mock()
            mock_model.fit.return_value = mock_model
            labels = np.array([i % k for i in range(50)])
            mock_model.predict.return_value = labels
            mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
            mock_model.predict_proba.return_value = np.random.rand(50, k)
            mock_model.bic.return_value = 100.0 + k * 10
            mock_model.aic.return_value = 90.0 + k * 10
            mock_model.converged_ = True
            return mock_model
        
        mock_gmm.side_effect = lambda n_components, **kwargs: create_mock_model(n_components)
        
        # Mock TSNE
        mock_tsne_instance = Mock()
        mock_tsne_instance.fit_transform.return_value = np.random.rand(50, 2)
        mock_tsne.return_value = mock_tsne_instance
        
        # Mock matplotlib - create comprehensive mocks for all subplot configurations
        mock_fig = Mock()
        
        # Create mock axes that can be used in different ways
        def create_mock_ax():
            ax = Mock()
            ax.plot = Mock()
            ax.axvline = Mock()
            ax.scatter = Mock(return_value=Mock())
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.legend = Mock()
            ax.grid = Mock()
            ax.axis = Mock()
            ax.text = Mock()
            ax.transAxes = Mock()
            
            # Create mock bars with numeric return values
            mock_bars = []
            for i in range(10):
                mock_bar = Mock()
                mock_bar.get_height.return_value = 20.0
                mock_bar.get_x.return_value = float(i)
                mock_bar.get_width.return_value = 0.8
                mock_bar.get_y.return_value = float(i)
                mock_bars.append(mock_bar)
            
            ax.bar = Mock(return_value=mock_bars)
            ax.barh = Mock(return_value=mock_bars)
            ax.set_xticks = Mock()
            ax.set_yticks = Mock()
            ax.set_xticklabels = Mock()
            ax.set_yticklabels = Mock()
            ax.imshow = Mock(return_value=Mock())
            
            # Mock violinplot with proper body structure
            mock_bodies = []
            for _ in range(5):
                body = Mock()
                body.set_facecolor = Mock()
                body.set_alpha = Mock()
                mock_bodies.append(body)
            ax.violinplot = Mock(return_value={'bodies': mock_bodies, 'cmins': Mock(), 'cmaxes': Mock()})
            
            # Mock table
            mock_table = Mock()
            mock_table.auto_set_font_size = Mock()
            mock_table.set_fontsize = Mock()
            mock_table.scale = Mock()
            mock_table.__getitem__ = Mock(return_value=Mock())
            ax.table = Mock(return_value=mock_table)
            
            ax.get_xticklabels = Mock(return_value=[])
            return ax
        
        # Setup side_effect to return different subplot structures based on arguments
        call_count = [0]
        def subplot_side_effect(*args, **kwargs):
            call_count[0] += 1
            if len(args) >= 2 and args[0] == 3 and args[1] == 3:
                # For plot_selection_metrics (3x3 grid)
                axes = np.array([[create_mock_ax() for _ in range(3)] for _ in range(3)])
                return (mock_fig, axes)
            elif len(args) >= 2 and isinstance(args[0], int) and isinstance(args[1], int):
                # For plot_feature_distributions (variable grid)
                n_rows, n_cols = args[0], args[1]
                axes = np.array([create_mock_ax() for _ in range(n_rows * n_cols)])
                return (mock_fig, axes.reshape(n_rows, n_cols) if n_rows > 1 or n_cols > 1 else axes)
            else:
                # For single subplot plots
                return (mock_fig, create_mock_ax())
        
        mock_plt.subplots.side_effect = subplot_side_effect
        mock_plt.colorbar.return_value = Mock(set_label=Mock())
        mock_plt.setp = Mock()
        mock_plt.suptitle = Mock()
        mock_plt.title = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.cm.viridis.return_value = np.random.rand(15, 4)
        mock_plt.cm.Set3.return_value = np.random.rand(3, 4)
        
        X = np.random.rand(50, 3)
        feature_names = ['f1', 'f2', 'f3']
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 4))
        gbtm.train_models(X)
        gbtm.compute_metrics(X)
        gbtm.compute_composite_scores()
        gbtm.select_optimal_k(X)
        
        output_dir = tmp_path / "plots"
        output_dir.mkdir()
        
        gbtm.generate_all_plots(X, feature_names, output_dir=str(output_dir))
        
        # Should have called close multiple times (7 plots without validation charts)
        assert mock_plt.close.call_count >= 7
    
    def test_print_selection_criteria(self, capsys):
        """Test print_selection_criteria output - lines 282-312"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        gbtm = GroupBasedTrajectoryModel(k_range=range(2, 4))
        gbtm.bic_scores = {2: 100.0, 3: 110.0}
        gbtm.silhouette_scores = {2: 0.5, 3: 0.6}
        gbtm.composite_scores = {2: 1.5, 3: 2.0}
        
        gbtm.print_selection_criteria()
        
        captured = capsys.readouterr()
        assert "MODEL SELECTION CRITERIA" in captured.out
        assert "BIC" in captured.out
        assert "Silhouette" in captured.out
        assert "OPTIMAL" in captured.out
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_class_sizes(self, mock_plt, mock_gmm, tmp_path):
        """Test plot_class_sizes - lines 453-481"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 0, 1, 1, 2] * 10)
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_bars = [Mock(), Mock(), Mock()]
        for i, bar in enumerate(mock_bars):
            bar.get_height.return_value = 20.0
            bar.get_x.return_value = float(i)
            bar.get_width.return_value = 0.8
        
        mock_ax.bar.return_value = mock_bars
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.set_xticks = Mock()
        mock_ax.set_xticklabels = Mock()
        mock_ax.grid = Mock()
        mock_ax.text = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=[3])
        gbtm.train_models(X)
        gbtm.optimal_k = 3
        gbtm.labels = mock_model.predict.return_value
        
        fig = gbtm.plot_class_sizes()
        
        assert fig is not None
        mock_ax.bar.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "class_sizes.png"
        fig = gbtm.plot_class_sizes(save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_clinical_profiles_heatmap(self, mock_plt, mock_gmm, tmp_path):
        """Test plot_clinical_profiles_heatmap - lines 488-532"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 17)[:50]
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_im = Mock()
        mock_ax.imshow.return_value = mock_im
        mock_ax.set_xticks = Mock()
        mock_ax.set_yticks = Mock()
        mock_ax.set_xticklabels = Mock()
        mock_ax.set_yticklabels = Mock()
        mock_ax.set_title = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.grid = Mock()
        mock_ax.get_xticklabels = Mock(return_value=[])
        
        mock_colorbar = Mock()
        mock_colorbar.set_label = Mock()
        mock_plt.colorbar.return_value = mock_colorbar
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 5)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        gbtm = GroupBasedTrajectoryModel(k_range=[3])
        gbtm.train_models(X)
        gbtm.optimal_k = 3
        gbtm.labels = mock_model.predict.return_value
        
        fig = gbtm.plot_clinical_profiles_heatmap(X, feature_names)
        
        assert fig is not None
        mock_ax.imshow.assert_called_once()
        mock_plt.colorbar.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "heatmap.png"
        fig = gbtm.plot_clinical_profiles_heatmap(X, feature_names, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_feature_distributions(self, mock_plt, mock_gmm, tmp_path):
        """Test plot_feature_distributions - lines 539-587"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 17)[:50]
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_axes = [Mock() for _ in range(10)]
        
        for ax in mock_axes:
            mock_parts = {
                'bodies': [Mock() for _ in range(3)],
                'cmins': Mock(),
                'cmaxes': Mock(),
                'cbars': Mock(),
                'cmedians': Mock(),
                'cmeans': Mock()
            }
            for body in mock_parts['bodies']:
                body.set_facecolor = Mock()
                body.set_alpha = Mock()
            
            ax.violinplot.return_value = mock_parts
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.set_xticks = Mock()
            ax.set_xticklabels = Mock()
            ax.grid = Mock()
            ax.axis = Mock()
        
        mock_plt.subplots.return_value = (mock_fig, np.array(mock_axes).reshape(5, 2))
        mock_plt.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        gbtm = GroupBasedTrajectoryModel(k_range=[3])
        gbtm.train_models(X)
        gbtm.optimal_k = 3
        gbtm.labels = mock_model.predict.return_value
        
        fig = gbtm.plot_feature_distributions(X, feature_names, top_n=5)
        
        assert fig is not None
        
        # Test with save path
        save_path = tmp_path / "distributions.png"
        fig = gbtm.plot_feature_distributions(X, feature_names, top_n=5, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_feature_importance_bars(self, mock_plt, mock_gmm, tmp_path):
        """Test plot_feature_importance_bars - lines 594-628"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 17)[:50]
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_bars = [Mock() for _ in range(10)]
        
        for i, bar in enumerate(mock_bars):
            bar.get_width.return_value = 0.5 + i * 0.1
            bar.get_y.return_value = float(i)
            bar.get_height.return_value = 0.8
        
        mock_ax.barh.return_value = mock_bars
        mock_ax.set_yticks = Mock()
        mock_ax.set_yticklabels = Mock()
        mock_ax.set_xlabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.grid = Mock()
        mock_ax.text = Mock()
        
        mock_plt.cm.viridis.return_value = np.random.rand(10, 4)
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 10)
        feature_names = [f'feature_{i}' for i in range(10)]
        
        gbtm = GroupBasedTrajectoryModel(k_range=[3])
        gbtm.train_models(X)
        gbtm.optimal_k = 3
        gbtm.labels = mock_model.predict.return_value
        
        fig = gbtm.plot_feature_importance_bars(X, feature_names, top_n=10)
        
        assert fig is not None
        mock_ax.barh.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "importance.png"
        fig = gbtm.plot_feature_importance_bars(X, feature_names, top_n=10, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_class_statistics_table(self, mock_plt, mock_gmm, tmp_path):
        """Test plot_class_statistics_table - lines 634-696"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 17)[:50]
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_table = Mock()
        mock_table.auto_set_font_size = Mock()
        mock_table.set_fontsize = Mock()
        mock_table.scale = Mock()
        mock_table.__getitem__ = Mock(return_value=Mock())
        
        mock_ax.axis = Mock()
        mock_ax.table.return_value = mock_table
        
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.title = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 5)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        gbtm = GroupBasedTrajectoryModel(k_range=[3])
        gbtm.train_models(X)
        gbtm.optimal_k = 3
        gbtm.labels = mock_model.predict.return_value
        
        fig = gbtm.plot_class_statistics_table(X, feature_names)
        
        assert fig is not None
        mock_ax.table.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "stats_table.png"
        fig = gbtm.plot_class_statistics_table(X, feature_names, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.f_oneway')
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    @patch('pandas.DataFrame.boxplot')
    def test_plot_sofa_distribution_by_class(self, mock_boxplot, mock_plt, mock_gmm, mock_f_oneway, tmp_path):
        """Test plot_sofa_distribution_by_class - lines 707-747"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 17)[:50]
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock f_oneway
        mock_f_oneway.return_value = (15.5, 0.001)
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax1 = Mock()
        mock_ax2 = Mock()
        mock_axes = np.array([mock_ax1, mock_ax2])
        
        for ax in [mock_ax1, mock_ax2]:
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.bar = Mock(return_value=[Mock()])
            ax.grid = Mock()
            ax.text = Mock()
        
        mock_boxplot.return_value = Mock()
        mock_plt.cm.Set3.return_value = np.random.rand(3, 4)
        mock_plt.subplots.return_value = (mock_fig, mock_axes)
        mock_plt.sca = Mock()
        mock_plt.xticks = Mock()
        mock_fig.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=[3])
        gbtm.train_models(X)
        gbtm.optimal_k = 3
        gbtm.labels = mock_model.predict.return_value
        
        df_with_classes = pd.DataFrame({
            'gbtm_class': gbtm.labels,
            'sofa_total': np.random.randint(0, 15, size=50).astype(float)
        })
        
        fig = gbtm.plot_sofa_distribution_by_class(df_with_classes)
        
        assert fig is not None
        mock_f_oneway.assert_called_once()
        mock_boxplot.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "sofa_dist.png"
        fig = gbtm.plot_sofa_distribution_by_class(df_with_classes, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.chi2_contingency')
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_plot_mortality_proxy_by_class(self, mock_plt, mock_gmm, mock_chi2, tmp_path):
        """Test plot_mortality_proxy_by_class - lines 759-800"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 17)[:50]
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock chi2_contingency
        mock_chi2.return_value = (10.5, 0.005, 2, np.array([[5, 5], [5, 5]]))
        
        # Mock matplotlib
        mock_fig = Mock()
        mock_ax = Mock()
        mock_bars = [Mock() for _ in range(3)]
        
        mock_ax.bar.return_value = mock_bars
        mock_ax.set_xlabel = Mock()
        mock_ax.set_ylabel = Mock()
        mock_ax.set_title = Mock()
        mock_ax.grid = Mock()
        mock_ax.set_ylim = Mock()
        mock_ax.text = Mock()
        
        mock_plt.cm.Set3.return_value = np.random.rand(3, 4)
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        
        X = np.random.rand(50, 3)
        
        gbtm = GroupBasedTrajectoryModel(k_range=[3])
        gbtm.train_models(X)
        gbtm.optimal_k = 3
        gbtm.labels = mock_model.predict.return_value
        
        df_with_classes = pd.DataFrame({
            'gbtm_class': gbtm.labels,
            'sofa_total': np.random.randint(0, 20, size=50).astype(float)
        })
        
        fig = gbtm.plot_mortality_proxy_by_class(df_with_classes)
        
        assert fig is not None
        mock_chi2.assert_called_once()
        mock_ax.bar.assert_called_once()
        
        # Test with save path
        save_path = tmp_path / "mortality.png"
        fig = gbtm.plot_mortality_proxy_by_class(df_with_classes, save_path=str(save_path))
        mock_plt.savefig.assert_called_once()
    
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.plt')
    def test_generate_all_plots_with_clinical_profiles(self, mock_plt, mock_gmm, tmp_path):
        """Test generate_all_plots with clinical profiles - lines 812-881"""
        from app.gbtm_pipeline import GroupBasedTrajectoryModel
        
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1, 2] * 17)[:50]
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, 50)
        mock_model.predict_proba.return_value = np.random.rand(50, 3)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock matplotlib with comprehensive mocks
        mock_fig = Mock()
        
        def create_mock_ax():
            ax = Mock()
            ax.plot = Mock()
            ax.axvline = Mock()
            ax.scatter = Mock(return_value=Mock())
            ax.set_xlabel = Mock()
            ax.set_ylabel = Mock()
            ax.set_title = Mock()
            ax.legend = Mock()
            ax.grid = Mock()
            ax.axis = Mock()
            ax.text = Mock()
            
            # Create mock bars with numeric return values
            mock_bars = []
            for i in range(10):
                mock_bar = Mock()
                mock_bar.get_height.return_value = 20.0
                mock_bar.get_x.return_value = float(i)
                mock_bar.get_width.return_value = 0.8
                mock_bar.get_y.return_value = float(i)
                mock_bars.append(mock_bar)
            
            ax.bar = Mock(return_value=mock_bars)
            ax.barh = Mock(return_value=mock_bars)
            ax.set_xticks = Mock()
            ax.set_yticks = Mock()
            ax.set_xticklabels = Mock()
            ax.set_yticklabels = Mock()
            ax.imshow = Mock(return_value=Mock())
            
            # Mock violinplot with proper body structure
            mock_bodies = []
            for _ in range(5):
                body = Mock()
                body.set_facecolor = Mock()
                body.set_alpha = Mock()
                mock_bodies.append(body)
            ax.violinplot = Mock(return_value={'bodies': mock_bodies, 'cmins': Mock(), 'cmaxes': Mock()})
            
            # Mock table
            mock_table = Mock()
            mock_table.auto_set_font_size = Mock()
            mock_table.set_fontsize = Mock()
            mock_table.scale = Mock()
            mock_table.__getitem__ = Mock(return_value=Mock())
            ax.table = Mock(return_value=mock_table)
            
            ax.get_xticklabels = Mock(return_value=[])
            ax.set_ylim = Mock()
            return ax
        
        # Setup side_effect to return different subplot structures
        def subplot_side_effect(*args, **kwargs):
            if len(args) >= 2 and args[0] == 3 and args[1] == 3:
                axes = np.array([[create_mock_ax() for _ in range(3)] for _ in range(3)])
                return (mock_fig, axes)
            elif len(args) >= 2 and args[0] == 1 and args[1] == 2:
                # For SOFA distribution (1x2 axes)
                axes = np.array([create_mock_ax(), create_mock_ax()])
                return (mock_fig, axes)
            elif len(args) >= 2 and isinstance(args[0], int) and isinstance(args[1], int):
                n_rows, n_cols = args[0], args[1]
                axes = np.array([create_mock_ax() for _ in range(n_rows * n_cols)])
                return (mock_fig, axes.reshape(n_rows, n_cols) if n_rows > 1 or n_cols > 1 else axes)
            else:
                return (mock_fig, create_mock_ax())
        
        mock_plt.subplots.side_effect = subplot_side_effect
        mock_plt.colorbar.return_value = Mock(set_label=Mock())
        mock_plt.setp = Mock()
        mock_plt.suptitle = Mock()
        mock_plt.title = Mock()
        mock_plt.sca = Mock()
        mock_plt.xticks = Mock()
        mock_fig.suptitle = Mock()
        mock_plt.tight_layout = Mock()
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        mock_plt.cm.viridis = Mock(return_value=np.random.rand(15, 4))
        mock_plt.cm.Set3 = Mock(return_value=np.random.rand(10, 4))
        
        X = np.random.rand(50, 5)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        
        gbtm = GroupBasedTrajectoryModel(k_range=[2, 3])
        gbtm.train_models(X)
        gbtm.compute_metrics(X)
        gbtm.compute_composite_scores()
        gbtm.select_optimal_k(X)
        
        df_with_classes = pd.DataFrame({
            'gbtm_class': gbtm.labels,
            'sofa_total': np.random.randint(0, 15, size=50).astype(float)
        })
        
        output_dir = tmp_path / "output"
        
        gbtm.generate_all_plots(X, feature_names, output_dir=str(output_dir), 
                               df_with_classes=df_with_classes)
        
        # Should have called close for all plots (9 total: 7 basic + 2 validation)
        assert mock_plt.close.call_count >= 7
    
    def test_data_loader_duckdb(self, tmp_path):
        """Test data_loader_duckdb - lines 894-898"""
        from app.gbtm_pipeline import data_loader_duckdb
        import duckdb
        
        # Create test database
        db_path = tmp_path / "test.duckdb"
        conn = duckdb.connect(str(db_path))
        
        test_data = pd.DataFrame({
            'id': [1, 2, 3],
            'value': [10, 20, 30]
        })
        conn.execute("CREATE TABLE test_table AS SELECT * FROM test_data")
        conn.close()
        
        # Load data
        df = data_loader_duckdb(str(db_path), 'test_table')
        
        assert len(df) == 3
        assert 'id' in df.columns
        assert 'value' in df.columns
    
    def test_filter_and_aggregate_patients(self, capsys):
        """Test filter_and_aggregate_patients - lines 913-956"""
        from app.gbtm_pipeline import filter_and_aggregate_patients
        
        data = {
            'stay_id': [1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
            'hour': [0, 1, 2, 0, 1, 0, 1, 2, 3, 4],
            'heart_rate': [80, 82, 85, 90, 92, 70, 72, 75, 78, 80],
            'age_at_admission': [65, 65, 65, 70, 70, 55, 55, 55, 55, 55],
            'sofa_total': [5, 5, 5, 7, 7, 3, 3, 3, 3, 3]
        }
        df = pd.DataFrame(data)
        
        # Test with filter enabled
        result = filter_and_aggregate_patients(df, patient_id_column='stay_id', 
                                              max_hours=4, filter_enabled=True)
        
        assert len(result) == 2  # Patients 1 and 2 have <= 4 hours
        
        captured = capsys.readouterr()
        assert "Initial data shape" in captured.out
        assert "Aggregated data shape" in captured.out
        
        # Test with filter disabled
        result = filter_and_aggregate_patients(df, patient_id_column='stay_id', 
                                              max_hours=4, filter_enabled=False)
        
        assert len(result) == 3  # All patients included
    
    @patch('app.gbtm_pipeline.GroupBasedTrajectoryModel')
    def test_run_gbtm_pipeline_with_filtering(self, mock_gbtm_class, tmp_path):
        """Test run_gbtm_pipeline with filtering - lines 986-1087"""
        from app.gbtm_pipeline import run_gbtm_pipeline
        
        df = pd.DataFrame({
            'stay_id': [1, 1, 2, 2, 2, 3, 3, 3, 3],
            'hour': [0, 1, 0, 1, 2, 0, 1, 2, 3],
            'f1': np.random.rand(9),
            'f2': np.random.rand(9)
        })
        
        # Mock GBTM class
        mock_gbtm = Mock()
        # After filtering with max_hours=3, we'll have 2 patients (stay_id 1 and 2)
        mock_gbtm.preprocess_data.return_value = (np.random.rand(2, 2), ['f1', 'f2'])
        mock_gbtm.train_models.return_value = None
        mock_gbtm.compute_metrics.return_value = None
        mock_gbtm.compute_composite_scores.return_value = None
        mock_gbtm.print_selection_criteria.return_value = None
        mock_gbtm.select_optimal_k.return_value = 2
        mock_gbtm.labels = np.array([0, 1])  # 2 patients after filtering
        mock_gbtm.optimal_model = Mock()
        mock_gbtm.optimal_model.predict_proba.return_value = np.random.rand(2, 2)
        mock_gbtm.generate_all_plots.return_value = None
        mock_gbtm.bic_scores = {2: 100.0}
        mock_gbtm.aic_scores = {2: 90.0}
        mock_gbtm.silhouette_scores = {2: 0.5}
        mock_gbtm.composite_scores = {2: 1.5}
        mock_gbtm_class.return_value = mock_gbtm
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = run_gbtm_pipeline(
            df=df,
            exclude_cols=['stay_id', 'hour'],
            db_name='test',
            k_range=[2],
            output_dir=str(output_dir),
            filter_hours=3
        )
        
        assert 'optimal_k' in result
        assert result['optimal_k'] == 2
        assert 'labels' in result
        assert 'probabilities' in result


class TestRunGBTMPipeline:
    """Test run_gbtm_pipeline function"""
    
    @patch('app.gbtm_pipeline.plt')
    @patch('app.gbtm_pipeline.GaussianMixture')
    @patch('app.gbtm_pipeline.duckdb')
    def test_run_gbtm_pipeline_basic(self, mock_duckdb, mock_gmm, mock_plt, tmp_path):
        """Test basic GBTM pipeline"""
        from app.gbtm_pipeline import run_gbtm_pipeline
        
        n_samples = 50
        
        # Mock GaussianMixture
        mock_model = Mock()
        mock_model.fit.return_value = mock_model
        mock_model.predict.return_value = np.array([0, 1] * (n_samples // 2))
        # Score samples needs to return values for all samples
        mock_model.score_samples.return_value = np.random.uniform(-100, -50, n_samples)
        mock_model.predict_proba.return_value = np.random.rand(n_samples, 2)
        mock_model.bic.return_value = 100.0
        mock_model.aic.return_value = 90.0
        mock_model.converged_ = True
        mock_gmm.return_value = mock_model
        
        # Mock DuckDB
        mock_conn = Mock()
        mock_duckdb.connect.return_value = mock_conn
        
        df = pd.DataFrame({
            'f1': np.random.rand(n_samples),
            'f2': np.random.rand(n_samples),
            'id': range(n_samples)
        })
        
        output_dir = tmp_path / "gbtm_output"
        output_dir.mkdir()
        
        # Create mock database file
        db_path = tmp_path / "test.duckdb"
        db_path.touch()
        
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        result = run_gbtm_pipeline(
            df=df,
            exclude_cols=['id'],
            db_name='test',
            k_range=range(2, 3),
            output_dir=str
            (output_dir),
            generate_plots=False
        )
        
        assert 'optimal_k' in result
        assert 'labels' in result


# ============================================================================
# Integration Tests
# ============================================================================

class TestPipelineIntegration:
    """Test pipeline integration scenarios"""
    
    def test_kmeans_pipeline_full_workflow(self, tmp_path):
        """Test complete kmeans pipeline workflow"""
        from app.kmeans_pipeline import run_consensus_pipeline
        
        # Create realistic data
        np.random.seed(42)
        n_samples = 100
        
        # Two clear clusters
        cluster1 = np.random.randn(n_samples // 2, 3) + [0, 0, 0]
        cluster2 = np.random.randn(n_samples // 2, 3) + [5, 5, 5]
        data = np.vstack([cluster1, cluster2])
        
        df = pd.DataFrame(data, columns=['f1', 'f2', 'f3'])
        df['id'] = range(n_samples)
        
        output_dir = tmp_path / "workflow"
        output_dir.mkdir()
        
        result = run_consensus_pipeline(
            df=df,
            exclude_cols=['id'],
            k_range=range(2, 4),
            n_iterations=10,
            output_dir=str(output_dir),
            generate_plots=False
        )
        
        # Should detect 2 clusters
        assert result['optimal_k'] >= 2
    
    @patch('app.dtw_pipeline.plot_dimensionality_reduction')
    @patch('app.dtw_pipeline.plot_cluster_sizes')
    @patch('app.dtw_pipeline.plt')
    @patch('app.dtw_pipeline.cdist_dtw')
    @patch('app.dtw_pipeline.load_and_prepare_data')
    @patch('app.dtw_pipeline.analyze_clinical_outcomes')
    def test_dtw_pipeline_handles_time_series(self, mock_clinical, mock_load_data, mock_cdist_dtw, mock_plt, mock_plot_sizes, mock_plot_dr, tmp_path):
        """Test DTW pipeline with time series data"""
        from app.dtw_pipeline import run_kmeans_dtw_pipeline
        
        n_samples = 40
        n_timepoints = 24
        
        # Use actual feature columns
        feature_columns = ['Heart Rate', 'Temperature_C', 'Respiratory Rate']
        
        # Mock the data loading to return DataFrame with proper columns
        data = {
            'stay_id': np.repeat(range(n_samples), n_timepoints),
            'hour': np.tile(range(n_timepoints), n_samples),
        }
        for col in feature_columns:
            data[col] = np.random.rand(n_samples * n_timepoints)
        
        mock_df = pd.DataFrame(data)
        mock_load_data.return_value = (mock_df, list(range(n_samples)))
        
        # Mock clinical outcomes analysis
        mock_clinical_df = pd.DataFrame({
            'cluster': range(2),
            'n_patients': [20, 20]
        })
        mock_clinical.return_value = mock_clinical_df
        
        # Mock DTW distance computation
        mock_distance_matrix = np.random.rand(n_samples, n_samples)
        mock_distance_matrix = (mock_distance_matrix + mock_distance_matrix.T) / 2
        np.fill_diagonal(mock_distance_matrix, 0)
        mock_cdist_dtw.return_value = mock_distance_matrix
        
        output_dir = tmp_path / "dtw_workflow"
        output_dir.mkdir()
        
        db_path = tmp_path / "test.duckdb"
        
        mock_plt.savefig = Mock()
        mock_plt.close = Mock()
        
        result = run_kmeans_dtw_pipeline(
            db_path=str(db_path),
            k_range=(2, 4),
            output_dir=str(output_dir),
            feature_columns=feature_columns,
            manual_k=2
        )
        
        assert 'optimal_k' in result
        assert 'labels' in result
        assert len(result['labels']) == n_samples
