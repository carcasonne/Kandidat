"""
Simple dataset caching to avoid reprocessing datasets.
"""

from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path

from modules.datasets import ADDdataset, ADDdatasetPretrain, ASVspoofDataset, FoRdataset, FoRdatasetPretrain, StretchMelCropTime, load_ASV_dataset

import pickle

class DatasetManager:
    def __init__(self, cache_dir="dataset_cache"):
        self.loaded_datasets = {}  # Cache in memory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def get_dataset(self, dataset_type, dataset_path, model_type="ast", 
                   max_per_class=1000, batch_size=16, target_frames=300):
        """
        Load dataset with caching.
        
        Args:
            dataset_type: "ADD", "ASV", "FoR"
            dataset_path: Path to dataset
            model_type: "ast" or "pretrain"
            max_per_class: Max samples per class
            batch_size: Batch size
            target_frames: Target frames for AST
        """
        # Create cache key
        cache_key = f"{dataset_type}_{model_type}_{max_per_class}_{batch_size}_{target_frames}"
        
        # Return cached if exists in memory
        if cache_key in self.loaded_datasets:
            print(f"Using cached dataset (memory): {cache_key}")
            return self.loaded_datasets[cache_key]
        
        # Check file cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            print(f"Loading dataset from file cache: {cache_key}")
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Reconstruct dataset
                dataset = self._create_dataset(dataset_type, dataset_path, model_type, 
                                             max_per_class, target_frames)
                dataset.files = cache_data['files']
                
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                self.loaded_datasets[cache_key] = loader
                return loader
            except Exception as e:
                print(f"Failed to load from cache: {e}")
        
        # Load fresh dataset
        print(f"Loading fresh dataset: {cache_key}")
        loader = self._create_fresh_dataset(dataset_type, dataset_path, model_type,
                                          max_per_class, batch_size, target_frames)
        
        # Save to file cache
        try:
            cache_data = {'files': loader.dataset.files}
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved to cache: {cache_file}")
        except Exception as e:
            print(f"Failed to save cache: {e}")
        
        self.loaded_datasets[cache_key] = loader
        return loader
    
    def _create_dataset(self, dataset_type, dataset_path, model_type, max_per_class, target_frames):
        """Create dataset instance without loading files."""
        if model_type == "ast":
            if dataset_type == "ADD":
                return ADDdataset(dataset_path, max_per_class, target_frames=target_frames)
            elif dataset_type == "ASV":
                return ASVspoofDataset(dataset_path, max_per_class, target_frames=target_frames)
            elif dataset_type == "FoR":
                return FoRdataset(dataset_path, max_per_class, target_frames=target_frames)
        else:  # pretrain
            transform = transforms.Compose([
                StretchMelCropTime(224, 224),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            
            if dataset_type == "ADD":
                return ADDdatasetPretrain(dataset_path, max_per_class, transform=transform)
            elif dataset_type == "ASV":
                return ASVspoofDatasetPretrain(dataset_path, max_per_class, transform=transform)
            elif dataset_type == "FoR":
                return FoRdatasetPretrain(dataset_path, max_per_class, transform=transform)
    
    def _create_fresh_dataset(self, dataset_type, dataset_path, model_type, 
                            max_per_class, batch_size, target_frames):
        """Create fresh dataset and loader."""
        if model_type == "ast":
            if dataset_type == "ADD":
                dataset = ADDdataset(dataset_path, max_per_class, target_frames=target_frames)
            elif dataset_type == "ASV":
                dataset = ASVspoofDataset(dataset_path, max_per_class, target_frames=target_frames)
            elif dataset_type == "FoR":
                dataset = FoRdataset(dataset_path, max_per_class, target_frames=target_frames)
        else:  # pretrain
            transform = transforms.Compose([
                StretchMelCropTime(224, 224),
                transforms.Normalize(mean=[0.485], std=[0.229])
            ])
            
            if dataset_type == "ADD":
                dataset = ADDdatasetPretrain(dataset_path, max_per_class, transform=transform)
            elif dataset_type == "ASV":
                loader, _, _ = load_ASV_dataset(dataset_path, max_per_class, False, None, transform)
                return loader
            elif dataset_type == "FoR":
                dataset = FoRdatasetPretrain(dataset_path, max_per_class, transform=transform)
        
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Global instance
dataset_manager = DatasetManager()

def get_dataset(dataset_type, dataset_path, model_type="ast", max_per_class=1000, 
               batch_size=16, target_frames=300):
    """Get dataset with caching."""
    return dataset_manager.get_dataset(dataset_type, dataset_path, model_type, 
                                     max_per_class, batch_size, target_frames)