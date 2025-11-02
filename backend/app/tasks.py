import os
import json
from datetime import datetime
from celery import shared_task
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
import warnings
warnings.filterwarnings('ignore')

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

@shared_task
def train_model(run_id: int, model_type: str, dataset_path: str, folder_path: str):
    try:
        # Create folder for results
        os.makedirs(folder_path, exist_ok=True)
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Extract features (assuming all numeric columns except first one which might be ID)
        X = df.select_dtypes(include=[np.number]).values
        
        results = {}
        
        if model_type == "kmeans":
            results = train_kmeans(X, folder_path)
        elif model_type == "kmeans_dtw":
            results = train_kmeans_dtw(X, folder_path)
        elif model_type == "lca":
            results = train_lca(X, folder_path)
        
        # Save results to JSON
        with open(os.path.join(folder_path, 'results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create notes file
        notes_file = os.path.join(folder_path, 'notes_feedback.txt')
        with open(notes_file, 'w') as f:
            f.write(f"Model Training Results\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: {model_type}\n")
            f.write(f"Optimal Clusters: {results.get('optimal_clusters', 'N/A')}\n")
            f.write(f"\n{'='*50}\n")
            f.write(f"NOTES AND FEEDBACK\n")
            f.write(f"{'='*50}\n\n")
        
        # Update run status
        update_run_status(run_id, {
            'status': 'completed',
            'optimal_clusters': results.get('optimal_clusters'),
            'completed_at': datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "optimal_clusters": results.get('optimal_clusters'),
            "folder_path": folder_path
        }
    except Exception as e:
        # Update run status to failed
        update_run_status(run_id, {'status': 'failed'})
        return {
            "status": "failed",
            "error": str(e)
        }

def train_kmeans(X, folder_path):
    """Train K-means and generate consensus plots"""
    # Try different k values
    k_range = range(2, min(11, len(X)))
    inertias = []
    silhouettes = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        if k < len(X):
            silhouettes.append(silhouette_score(X, labels))
    
    # Find optimal k using elbow method
    optimal_k = k_range[silhouettes.index(max(silhouettes))]
    
    # Final model with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Generate consensus matrix plot
    plt.figure(figsize=(10, 8))
    # Create a simple consensus matrix based on cluster assignments
    n_samples = len(X)
    consensus_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if labels[i] == labels[j]:
                consensus_matrix[i, j] = 1
    
    sns.heatmap(consensus_matrix, cmap='YlOrRd', square=True, cbar_kws={'label': 'Consensus'})
    plt.title(f'Consensus Matrix (k={optimal_k})')
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'consensus_matrix.png'), dpi=150)
    plt.close()
    
    # Elbow plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'elbow_plot.png'), dpi=150)
    plt.close()
    
    # Silhouette plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouettes, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score For Different k')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'silhouette_plot.png'), dpi=150)
    plt.close()
    
    return {
        "optimal_clusters": int(optimal_k),
        "plots": ["consensus_matrix.png", "elbow_plot.png", "silhouette_plot.png"]
    }

def train_kmeans_dtw(X, folder_path):
    """Train K-means with DTW for temporal data"""
    k_range = range(2, min(11, len(X)))
    inertias = []
    silhouettes = []
    davies_bouldins = []
    
    for k in k_range:
        model = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42)
        labels = model.fit_predict(X)
        inertias.append(model.inertia_)
        if k < len(X):
            silhouettes.append(silhouette_score(X, labels))
            davies_bouldins.append(davies_bouldin_score(X, labels))
    
    # Find optimal k
    optimal_k = k_range[silhouettes.index(max(silhouettes))]
    
    # Final model
    model = TimeSeriesKMeans(n_clusters=optimal_k, metric="dtw", random_state=42)
    labels = model.fit_predict(X)
    
    # Silhouette plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouettes, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score (DTW)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'silhouette_dtw.png'), dpi=150)
    plt.close()
    
    # Inertia plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'go-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs Number of Clusters (DTW)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'inertia_dtw.png'), dpi=150)
    plt.close()
    
    # Davies-Bouldin plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, davies_bouldins, 'ro-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Davies-Bouldin Index')
    plt.title('Davies-Bouldin Index (DTW)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'davies_bouldin_dtw.png'), dpi=150)
    plt.close()
    
    # Cluster visualization
    plt.figure(figsize=(12, 8))
    for i in range(optimal_k):
        cluster_data = X[labels == i]
        plt.subplot(optimal_k, 1, i+1)
        for series in cluster_data[:min(50, len(cluster_data))]:
            plt.plot(series, alpha=0.3)
        plt.title(f'Cluster {i} (n={len(cluster_data)})')
        plt.ylabel('Value')
    plt.xlabel('Time')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'cluster_plots_dtw.png'), dpi=150)
    plt.close()
    
    return {
        "optimal_clusters": int(optimal_k),
        "plots": ["silhouette_dtw.png", "inertia_dtw.png", "davies_bouldin_dtw.png", "cluster_plots_dtw.png"]
    }

def train_lca(X, folder_path):
    """Train Latent Class Analysis"""
    # Simplified LCA using KMeans as proxy
    k_range = range(2, min(11, len(X)))
    bics = []
    aics = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate simplified BIC and AIC
        n_samples = len(X)
        n_features = X.shape[1]
        n_params = k * n_features + k - 1
        
        inertia = kmeans.inertia_
        bic = inertia + n_params * np.log(n_samples)
        aic = inertia + 2 * n_params
        
        bics.append(bic)
        aics.append(aic)
    
    # Find optimal k
    optimal_k = k_range[bics.index(min(bics))]
    
    # BIC plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, bics, 'bo-')
    plt.xlabel('Number of Classes')
    plt.ylabel('BIC')
    plt.title('BIC for LCA')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'bic_lca.png'), dpi=150)
    plt.close()
    
    # AIC plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, aics, 'go-')
    plt.xlabel('Number of Classes')
    plt.ylabel('AIC')
    plt.title('AIC for LCA')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'aic_lca.png'), dpi=150)
    plt.close()
    
    # Class probability plot
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    plt.figure(figsize=(10, 6))
    class_counts = [np.sum(labels == i) for i in range(optimal_k)]
    plt.bar(range(optimal_k), class_counts)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'class_distribution_lca.png'), dpi=150)
    plt.close()
    
    return {
        "optimal_clusters": int(optimal_k),
        "plots": ["bic_lca.png", "aic_lca.png", "class_distribution_lca.png"]
    }