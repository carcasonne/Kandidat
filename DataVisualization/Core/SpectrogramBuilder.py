import os
import numpy as np
from pathlib import Path

class SpectrogramBuilder:
    """
    A class for loading and processing spectrogram data from .npy files.
    """
    
    def __init__(self, data_dir="Data"):
        """
        Initialize the SpectrogramBuilder with the directory containing spectrogram data.
        
        Args:
            data_dir (str): Path to the directory containing spectrogram data
        """
        self.data_dir = data_dir
        self.real_dir = os.path.join(data_dir, "specgrams_real")
        self.fake_dir = os.path.join(data_dir, "specgrams_fake")
        
    def load_spectrogram(self, file_path):
        """
        Load a spectrogram from a .npy file.
        
        Args:
            file_path (str): Path to the .npy file
            
        Returns:
            numpy.ndarray: The loaded spectrogram data
        """
        try:
            return np.load(file_path)
        except Exception as e:
            print(f"Error loading spectrogram from {file_path}: {e}")
            return None
    
    def load_all_spectrograms(self, category="real", max_files=None):
        """
        Load all spectrograms from a specific category.
        
        Args:
            category (str): "real" or "fake"
            max_files (int, optional): Maximum number of files to load
            
        Returns:
            dict: Dictionary with filenames as keys and spectrogram data as values
        """
        target_dir = self.real_dir if category == "real" else self.fake_dir
        
        if not os.path.exists(target_dir):
            print(f"Directory not found: {target_dir}")
            return {}
        
        spectrograms = {}
        files = [f for f in os.listdir(target_dir) if f.endswith('.npy')]
        
        if max_files is not None:
            files = files[:max_files]
        
        for filename in files:
            file_path = os.path.join(target_dir, filename)
            spec_data = self.load_spectrogram(file_path)
            if spec_data is not None:
                spectrograms[filename] = spec_data
        
        return spectrograms
    
    def get_sample_spectrograms(self, num_real=3, num_fake=3):
        """
        Get a sample of both real and fake spectrograms.
        
        Args:
            num_real (int): Number of real spectrograms to load
            num_fake (int): Number of fake spectrograms to load
            
        Returns:
            tuple: (real_spectrograms, fake_spectrograms) as dictionaries
        """
        real_spectrograms = self.load_all_spectrograms("real", num_real)
        fake_spectrograms = self.load_all_spectrograms("fake", num_fake)
        
        return real_spectrograms, fake_spectrograms
    
    def get_spectrogram_stats(self, spec_data):
        """
        Get basic statistics about a spectrogram.
        
        Args:
            spec_data (numpy.ndarray): Spectrogram data
            
        Returns:
            dict: Dictionary containing statistics
        """
        return {
            "shape": spec_data.shape,
            "min": np.min(spec_data),
            "max": np.max(spec_data),
            "mean": np.mean(spec_data),
            "std": np.std(spec_data)
        }