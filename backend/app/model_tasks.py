import os
import json
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import duckdb
from app.celery_worker import celery_app
from app.pipeline import run_consensus_pipeline


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
        # Create folder for results
        os.makedirs(folder_path, exist_ok=True)

        # Read duckdb file
        con = duckdb.connect(database=dataset_path) 
        print(f"Connected to DuckDB database at {dataset_path}")
        print(dataset_name)
        con.sql("SHOW TABLES;").show()
        # Load dataset
        df = con.sql(f"SELECT * FROM {dataset_name}").fetchdf()

        # Read parameters file to get parameters
        config_file_path = parameters_path
        try:
            with open(config_file_path, 'r') as f:
                config = yaml.safe_load(f)
            range_params = config['range']
            k_min = range_params['k_min']
            k_max = range_params['k_max']
            exclude_cols = config['columns_to_exclude']
        except FileNotFoundError:
            print(f"Error: The file '{config_file_path}' was not found.")
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
        
        # Extract features (assuming all numeric columns except first one which might be ID)        
        results = {}

        if model_type == "kmeans":
            n_iterations = range_params['max-iter']
            results = run_consensus_pipeline(df, exclude_cols, k_range=range(k_min, k_max+1), 
                           n_iterations=n_iterations, subsample_fraction=0.8,
                           correlation_threshold=0.8, log_transform=False,
                           subsample_data=None, output_dir=folder_path,
                           manual_k=None, random_state=42)
        elif model_type == "lca":
            n_components = range_params['n_components']
            init_params = range_params['init_params']
            n_steps = range_params['n_steps']
            abs_tol = range_params['abs_tol']
            rel_tol = range_params['rel_tol']
            # results = train_kmeans_dtw(X, folder_path)
        elif model_type == "k_means_dtw":
            metric = range_params['metric']
            max_iter = range_params['max_iter']
            n_init = range_params['n_init']
            n_jobs = range_params['n_jobs']
            random_state = range_params['random_state']
            perplexity = range_params['perplexity']
            max_iter_tsne = range_params['max_iter_tsne']
            init = range_params['init']
            # results = train_lca(X, folder_path)
        elif model_type == "gbtm":

            # results = train_gbtm(X, folder_path)
            
            notes_file = os.path.join(folder_path, 'notes_feedback.txt')
            
            with open(notes_file, 'w') as f:
                f.write(f"Model Training Results\n")
                f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Model Type: {model_type}\n")
                f.write(f"Optimal Clusters: {results.get('optimal_clusters', 'N/A')}\n")
                f.write(f"\n{'='*50}\n")
                f.write(f"NOTES AND FEEDBACK\n")
                f.write(f"{'='*50}\n\n")

            return {
                "status": "success",
                "optimal_clusters": results.get('optimal_clusters'),
                "folder_path": folder_path
        }

    
    except Exception as e:
        update_run_status(run_id, {
            'status': 'failed',
            'completed_at': datetime.utcnow().isoformat()
        })
        raise e
        
