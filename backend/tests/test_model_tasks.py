"""
Unit tests for model_tasks.py - Celery task functionality
"""
import pytest
import os
import json
import yaml
import tempfile
from unittest.mock import Mock, patch, MagicMock, mock_open
from datetime import datetime
from app.model_tasks import update_run_status, train_model


class TestUpdateRunStatus:
    """Test update_run_status function"""
    
    def test_update_run_status_success(self, test_storage_files):
        """Test successfully updating run status"""
        from app.storage import RunStorage
        from app.config import settings
        
        # Create a test run
        run = RunStorage.create({
            'user_id': 1,
            'user_email': 'test@example.com',
            'model_type': 'kmeans',
            'dataset_filename': 'test.duckdb',
            'dataset_name': 'test_table',
            'parameters_filename': 'params.yaml'
        })
        
        # Update status
        update_run_status(run['id'], {'status': 'completed', 'optimal_clusters': 3})
        
        # Verify update
        updated_run = RunStorage.get_by_id(run['id'])
        assert updated_run['status'] == 'completed'
        assert updated_run['optimal_clusters'] == 3
    
    def test_update_nonexistent_run(self, test_storage_files):
        """Test updating non-existent run (should not raise error)"""
        # Should not raise an exception
        update_run_status(99999, {'status': 'failed'})
    
    def test_update_multiple_fields(self, test_storage_files):
        """Test updating multiple fields at once"""
        from app.storage import RunStorage
        
        run = RunStorage.create({
            'user_id': 1,
            'user_email': 'test@example.com',
            'model_type': 'lca',
            'dataset_filename': 'test.duckdb',
            'dataset_name': 'test_table',
            'parameters_filename': 'params.yaml'
        })
        
        updates = {
            'status': 'completed',
            'optimal_clusters': 5,
            'completed_at': '2025-01-01T00:00:00'
        }
        
        update_run_status(run['id'], updates)
        
        updated_run = RunStorage.get_by_id(run['id'])
        assert updated_run['status'] == 'completed'
        assert updated_run['optimal_clusters'] == 5
        assert updated_run['completed_at'] == '2025-01-01T00:00:00'


class TestTrainModelTask:
    """Test train_model Celery task"""
    
    @pytest.fixture
    def mock_config_file(self, tmp_path):
        """Create a mock config YAML file"""
        config = {
            'range': {
                'k_min': 2,
                'k_max': 5
            },
            'columns_to_exclude': ['id', 'timestamp'],
            'hyperparameters': {
                'random_state': 42,
                'n_iterations': 10,
                'correlation_threshold': 0.8,
                'log_transform': False,
                'subsample_fraction': 0.8,
                'subsample_data': 0.1,
                'manual_k': None
            }
        }
        
        config_path = tmp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return str(config_path)
    
    @pytest.fixture
    def mock_duckdb_file(self, tmp_path):
        """Create a mock DuckDB file path"""
        db_path = tmp_path / "test.duckdb"
        return str(db_path)
    
    def test_train_model_file_not_found(self, tmp_path):
        """Test train_model with non-existent config file"""
        with pytest.raises(FileNotFoundError):
            # Call train_model without self parameter (Celery handles binding)
            train_model(
                run_id=1,
                model_type="kmeans",
                dataset_path="/nonexistent/path.duckdb",
                parameters_path="/nonexistent/config.yaml",
                dataset_name="test_table",
                folder_path=str(tmp_path)
            )
    
    def test_train_model_invalid_yaml(self, tmp_path):
        """Test train_model with invalid YAML file"""
        # Create invalid YAML file
        invalid_yaml = tmp_path / "invalid.yaml"
        with open(invalid_yaml, 'w') as f:
            f.write("invalid: yaml: content: [")
        
        with pytest.raises(yaml.YAMLError):
            # Call train_model without self parameter (Celery handles binding)
            train_model(
                run_id=1,
                model_type="kmeans",
                dataset_path="/path/to/data.duckdb",
                parameters_path=str(invalid_yaml),
                dataset_name="test_table",
                folder_path=str(tmp_path)
            )
    
    @patch('app.model_tasks.run_consensus_pipeline')
    @patch('app.model_tasks.duckdb.connect')
    def test_train_model_kmeans_success(self, mock_duckdb, mock_pipeline, tmp_path, mock_config_file):
        """Test successful kmeans model training"""
        # Mock DuckDB connection
        mock_con = Mock()
        mock_df = Mock()
        mock_con.sql.return_value.fetchdf.return_value = mock_df
        mock_duckdb.return_value = mock_con
        
        # Mock pipeline result
        mock_pipeline.return_value = {'optimal_k': 3, 'labels': [0, 1, 2]}
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = train_model(
            run_id=1,
            model_type="kmeans",
            dataset_path=str(tmp_path / "test.duckdb"),
            parameters_path=mock_config_file,
            dataset_name="test_table",
            folder_path=str(output_dir)
        )
        
        assert result['status'] == 'success'
        assert result['optimal_clusters'] == 3
        assert 'folder_path' in result
        
        # Verify pipeline was called
        mock_pipeline.assert_called_once()
    
    @patch('app.model_tasks.run_lca_pipeline')
    @patch('app.model_tasks.duckdb.connect')
    def test_train_model_lca_success(self, mock_duckdb, mock_pipeline, tmp_path):
        """Test successful LCA model training"""
        # Create LCA config
        config = {
            'range': {'k_min': 2, 'k_max': 4},
            'columns_to_exclude': ['id'],
            'hyperparameters': {
                'random_state': 42,
                'n_init': 10,
                'n_iterations': 100,
                'correlation_threshold': 0.8,
                'log_transform': False,
                'subsample_fraction': 0.8,
                'subsample_data': 0.1,
                'manual_k': None,
                'selection_method': 'bic'
            }
        }
        
        config_path = tmp_path / "lca_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Mock DuckDB
        mock_con = Mock()
        mock_df = Mock()
        mock_con.sql.return_value.fetchdf.return_value = mock_df
        mock_duckdb.return_value = mock_con
        
        # Mock pipeline
        mock_pipeline.return_value = {'optimal_k': 4, 'labels': [0, 1, 2, 3]}
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = train_model(
            run_id=2,
            model_type="lca",
            dataset_path=str(tmp_path / "test.duckdb"),
            parameters_path=str(config_path),
            dataset_name="test_table",
            folder_path=str(output_dir)
        )
        
        assert result['status'] == 'success'
        assert result['optimal_clusters'] == 4
        mock_pipeline.assert_called_once()
    
    @patch('app.model_tasks.run_kmeans_dtw_pipeline')
    def test_train_model_kmeans_dtw_success(self, mock_pipeline, tmp_path):
        """Test successful kmeans_dtw model training"""
        config = {
            'range': {'k_min': 2, 'k_max': 3},
            'columns_to_exclude': ['id'],
            'hyperparameters': {
                'random_state': 42,
                'n_init': 5,
                'time_window_hours': 24,
                'dtw_chunk_size': 100,
                'manual_k': None,
                'subsample_fraction': 0.5,
                'feature_columns': ['col1', 'col2']
            }
        }
        
        config_path = tmp_path / "dtw_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        mock_pipeline.return_value = {'optimal_k': 2, 'labels': [0, 1]}
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = train_model(
            run_id=3,
            model_type="kmeans_dtw",
            dataset_path=str(tmp_path / "test.duckdb"),
            parameters_path=str(config_path),
            dataset_name="test_table",
            folder_path=str(output_dir)
        )
        
        assert result['status'] == 'success'
        assert result['optimal_clusters'] == 2
        mock_pipeline.assert_called_once()
    
    @patch('app.model_tasks.run_gbtm_pipeline')
    @patch('app.model_tasks.duckdb.connect')
    def test_train_model_gbtm_success(self, mock_duckdb, mock_pipeline, tmp_path):
        """Test successful GBTM model training"""
        config = {
            'range': {'k_min': 2, 'k_max': 4},
            'columns_to_exclude': ['id'],
            'hyperparameters': {
                'random_state': 42,
                'n_init': 10,
                'n_iterations': 50
            }
        }
        
        config_path = tmp_path / "gbtm_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Mock DuckDB
        mock_con = Mock()
        mock_df = Mock()
        mock_con.sql.return_value.fetchdf.return_value = mock_df
        mock_duckdb.return_value = mock_con
        
        # Mock pipeline
        mock_pipeline.return_value = {'optimal_k': 3, 'labels': [0, 1, 2]}
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = train_model(
            run_id=4,
            model_type="gbtm",
            dataset_path=str(tmp_path / "test.duckdb"),
            parameters_path=str(config_path),
            dataset_name="test_table",
            folder_path=str(output_dir)
        )
        
        assert result['status'] == 'success'
        assert result['optimal_clusters'] == 3
        mock_pipeline.assert_called_once()
    
    @patch('app.model_tasks.run_consensus_pipeline')
    @patch('app.model_tasks.duckdb.connect')
    def test_train_model_creates_notes_file(self, mock_duckdb, mock_pipeline, tmp_path, mock_config_file):
        """Test that train_model creates notes file"""
        mock_con = Mock()
        mock_df = Mock()
        mock_con.sql.return_value.fetchdf.return_value = mock_df
        mock_duckdb.return_value = mock_con
        
        mock_pipeline.return_value = {'optimal_k': 3}
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        train_model(
            run_id=1,
            model_type="kmeans",
            dataset_path=str(tmp_path / "test.duckdb"),
            parameters_path=mock_config_file,
            dataset_name="test_table",
            folder_path=str(output_dir)
        )
        
        # Check notes file was created
        notes_file = output_dir / "notes_feedback.txt"
        assert notes_file.exists()
        
        # Check notes file content
        content = notes_file.read_text()
        assert "Model Training Results" in content
        assert "kmeans" in content
        assert "Optimal Clusters: 3" in content
    
    @patch('app.model_tasks.update_run_status')
    @patch('app.model_tasks.run_consensus_pipeline')
    @patch('app.model_tasks.duckdb.connect')
    def test_train_model_handles_exception(self, mock_duckdb, mock_pipeline, mock_update_status, tmp_path, mock_config_file):
        """Test train_model handles exceptions properly"""
        mock_duckdb.side_effect = Exception("Database connection failed")
        
        with pytest.raises(Exception):
            train_model(
                run_id=1,
                model_type="kmeans",
                dataset_path=str(tmp_path / "test.duckdb"),
                parameters_path=mock_config_file,
                dataset_name="test_table",
                folder_path=str(tmp_path / "output")
            )
        
        # Verify update_run_status was called with failed status
        mock_update_status.assert_called_once()
        call_args = mock_update_status.call_args[0]
        assert call_args[0] == 1
        assert call_args[1]['status'] == 'failed'
    
    @patch('app.model_tasks.run_consensus_pipeline')
    @patch('app.model_tasks.duckdb.connect')
    def test_train_model_with_manual_k(self, mock_duckdb, mock_pipeline, tmp_path):
        """Test training with manual_k specified"""
        config = {
            'range': {'k_min': 2, 'k_max': 5},
            'columns_to_exclude': ['id'],
            'hyperparameters': {
                'random_state': 42,
                'n_iterations': 10,
                'correlation_threshold': 0.8,
                'log_transform': False,
                'subsample_fraction': 0.8,
                'subsample_data': 0.1,
                'manual_k': 4
            }
        }
        
        config_path = tmp_path / "config_manual.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        mock_con = Mock()
        mock_df = Mock()
        mock_con.sql.return_value.fetchdf.return_value = mock_df
        mock_duckdb.return_value = mock_con
        
        mock_pipeline.return_value = {'optimal_k': 4}
        
        # Create output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = train_model(
            run_id=1,
            model_type="kmeans",
            dataset_path=str(tmp_path / "test.duckdb"),
            parameters_path=str(config_path),
            dataset_name="test_table",
            folder_path=str(output_dir)
        )
        
        # Verify manual_k was passed to pipeline
        call_kwargs = mock_pipeline.call_args[1]
        assert call_kwargs['manual_k'] == 4
    
    @patch('app.model_tasks.run_consensus_pipeline')
    @patch('app.model_tasks.duckdb.connect')
    def test_train_model_result_structure(self, mock_duckdb, mock_pipeline, tmp_path, mock_config_file):
        """Test that train_model returns correct result structure"""
        mock_con = Mock()
        mock_df = Mock()
        mock_con.sql.return_value.fetchdf.return_value = mock_df
        mock_duckdb.return_value = mock_con
        
        mock_pipeline.return_value = {
            'optimal_k': 5,
            'labels': [0, 1, 2, 3, 4],
            'extra_data': 'should not be in result'
        }
        
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        result = train_model(
            run_id=1,
            model_type="kmeans",
            dataset_path=str(tmp_path / "test.duckdb"),
            parameters_path=mock_config_file,
            dataset_name="test_table",
            folder_path=str(output_dir)
        )
        
        assert 'status' in result
        assert 'optimal_clusters' in result
        assert 'folder_path' in result
        assert result['status'] == 'success'
        assert result['optimal_clusters'] == 5
        assert result['folder_path'] == str(output_dir)
