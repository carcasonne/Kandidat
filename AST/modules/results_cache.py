"""
Cache analysis results to avoid reprocessing datasets.
"""

import pickle
import os
from pathlib import Path
import hashlib


class ResultsCache:
    def __init__(self, cache_dir="results_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, model_path, dataset_info, dataset_name):
        """Generate cache key from model and dataset info."""
        key_str = f"{model_path}_{dataset_info}_{dataset_name}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_results(self, model_path, dataset_info, dataset_name):
        """Get cached results if they exist."""
        cache_key = self._get_cache_key(model_path, dataset_info, dataset_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    results = pickle.load(f)
                print(f"Loaded cached results for {dataset_name}")
                return results
            except Exception as e:
                print(f"Failed to load cache: {e}")
        
        return None
    
    def save_results(self, model_path, dataset_info, dataset_name, 
                    predictions, ground_truth, confidences, misclassified_samples):
        """Save analysis results to cache."""
        cache_key = self._get_cache_key(model_path, dataset_info, dataset_name)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        results = {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'confidences': confidences,
            'misclassified_samples': misclassified_samples,
            'dataset_name': dataset_name
        }
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            print(f"Cached results for {dataset_name}")
        except Exception as e:
            print(f"Failed to save cache: {e}")


# Global cache instance
results_cache = ResultsCache()


def get_cached_results(model_path, dataset_info, dataset_name):
    """Get cached analysis results."""
    return results_cache.get_results(model_path, dataset_info, dataset_name)


def save_cached_results(model_path, dataset_info, dataset_name, 
                       predictions, ground_truth, confidences, misclassified_samples):
    """Save analysis results to cache."""
    results_cache.save_results(model_path, dataset_info, dataset_name,
                             predictions, ground_truth, confidences, misclassified_samples)