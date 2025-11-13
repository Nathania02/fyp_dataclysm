import os
import json
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import duckdb
from app.celery_worker import celery_app
from app.kmeans_pipeline import run_consensus_pipeline
from app.lca_pipeline import run_lca_pipeline
from app.dtw_pipeline import run_kmeans_dtw_pipeline
from app.gbtm_pipeline import run_gbtm_pipeline

def update_run_status(run_id: int, updates: dict):
    """Update run status in JSON file"""
    from app.config import settings
    runs_file = settings.RUNS_FILE
    
    with open(runs_file, 'r') as f:
        runs = json.load(f)
    
    for i, run in enumerate(runs):
        if run['id'] == run_id:
            runs[i].update(updates)
            break
    
    with open(runs_file, 'w') as f:
        json.dump(runs, f, indent=2, default=str)

@celery_app.task(bind=True)
def train_model(self, run_id: int, model_type: str, dataset_path: str, parameters_path: str, dataset_name: str, folder_path: str):
    try:
        # Read parameters file to get parameters
        config_file_path = parameters_path
        try:
            with open(config_file_path, 'r') as f:
                config = yaml.safe_load(f)
            range_params = config['range']
            k_min = range_params['k_min']
            k_max = range_params['k_max']
            exclude_cols = config['columns_to_exclude']
            random_state = config['hyperparameters']['random_state']

        except FileNotFoundError:
            print(f"Error: The file '{config_file_path}' was not found.")
            raise
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            raise
        
        # Extract features (assuming all numeric columns except first one which might be ID)        
        results = {}

        if model_type == "kmeans":
            # Read and load duckdb file
            con = duckdb.connect(database=dataset_path) 
            df = con.sql(f"SELECT * FROM {dataset_name}").fetchdf()

            n_iterations = config['hyperparameters']['n_iterations']
            correlation_threshold = config['hyperparameters']['correlation_threshold']
            log_transform = config['hyperparameters']['log_transform']
            subsample_fraction = config['hyperparameters']['subsample_fraction']
            subsample_data = config['hyperparameters']['subsample_data']
            manual_k = config['hyperparameters']['manual_k']

            results = run_consensus_pipeline(df, exclude_cols, k_range=range(k_min, k_max+1), 
                           n_iterations=n_iterations, subsample_fraction=subsample_fraction,
                           correlation_threshold=correlation_threshold, log_transform=log_transform,
                           subsample_data=subsample_data, output_dir=folder_path,
                           manual_k=manual_k, random_state=random_state)
            print(results)
            print("Consensus clustering completed.")


        elif model_type == "lca":

            # Read and load duckdb file
            con = duckdb.connect(database=dataset_path) 
            df = con.sql(f"SELECT * FROM {dataset_name}").fetchdf()

            n_init = config['hyperparameters']['n_init']
            n_iterations = config['hyperparameters']['n_iterations']
            # init_params = config['hyperparameters']['init_params']
            correlation_threshold = config['hyperparameters']['correlation_threshold']
            log_transform = config['hyperparameters']['log_transform']
            subsample_fraction = config['hyperparameters']['subsample_fraction']
            subsample_data = config['hyperparameters']['subsample_data']
            manual_k = config['hyperparameters']['manual_k']
            selection_method = config['hyperparameters']['selection_method']

            # n_steps = config['hyperparameters']['n_steps']
            # abs_tol = config['hyperparameters']['abs_tol']
            # rel_tol = config['hyperparameters']['rel_tol']

            results = run_lca_pipeline(df, exclude_cols, k_range=range(k_min, k_max+1),
                           n_init=n_init, max_iter=n_iterations,
                           correlation_threshold=correlation_threshold, log_transform=log_transform,
                           subsample_data=subsample_data, output_dir=folder_path,
                           manual_k=manual_k, selection_method=selection_method, random_state=random_state)
            print(results)
            print("Consensus clustering completed.")

        elif model_type == "kmeans_dtw":
            n_init = config['hyperparameters']['n_init']
            time_window_hours = config['hyperparameters']['time_window_hours']
            dtw_chunk_size = config['hyperparameters']['dtw_chunk_size']
            manual_k = config['hyperparameters']['manual_k']
            subsample_fraction = config['hyperparameters']['subsample_fraction']
            feature_columns = config['hyperparameters']['feature_columns']

            results = run_kmeans_dtw_pipeline(dataset_path, table_name=dataset_name, time_window_hours=time_window_hours, 
                                               output_dir=folder_path, manual_k=manual_k, k_range=range(k_min, k_max+1),
                                               feature_columns=feature_columns,dtw_chunk_size=dtw_chunk_size, subsample_fraction=subsample_fraction, 
                                              random_state=random_state)

            print(results)
            print("DTW K-Means clustering completed.")

        elif model_type == "gbtm":
            
            # Read and load duckdb file
            con = duckdb.connect(database=dataset_path) 
            df = con.sql(f"SELECT * FROM {dataset_name}").fetchdf()

            n_init = config['hyperparameters']['n_init']
            n_iterations = config['hyperparameters']['n_iterations']
            results = run_gbtm_pipeline(df, exclude_cols, db_name=dataset_name, k_range=range(k_min, k_max+1),
                                        n_init=n_init, max_iter=n_iterations,random_state=random_state,
                                        output_dir=folder_path)

            print(results)
            print("GBTM clustering completed.")

        notes_file = os.path.join(folder_path, 'notes_feedback.txt')
        
        print(f"\nCreating notes file at: {notes_file}")
        
        with open(notes_file, 'w') as f:
            f.write(f"Model Training Results\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Optimal Clusters: {results.get('optimal_k', 'N/A')}\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"NOTES AND FEEDBACK\n")
            f.write(f"{'='*50}\n\n")
        
        print(f"Notes file created successfully at: {notes_file}")

        return {
            "status": "success",
            "optimal_clusters": results.get('optimal_k'),
            "folder_path": folder_path
        }
    
    except Exception as e:
        update_run_status(run_id, {
            'status': 'failed',
            'completed_at': datetime.utcnow().isoformat()
        })
        raise e
        
