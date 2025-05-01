#!/usr/bin/env python3
"""
Utility functions for the spectrogram visualization project.
"""

import os
import numpy as np
from typing import List, Tuple, Dict

def load_spectrogram_files(directory: str) -> List[str]:
    """
    Load all .npy spectrogram files from a directory.
    
    Args:
        directory: Directory containing .npy files
        
    Returns:
        List of file paths
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
        
    file_paths = []
    for file in os.listdir(directory):
        if file.endswith('.npy'):
            file_paths.append(os.path.join(directory, file))
            
    return file_paths

def get_spectrogram_duration(spec_file: str) -> float:
    """
    Calculate the duration of a spectrogram in seconds.
    
    Args:
        spec_file: Path to .npy spectrogram file
        
    Returns:
        Duration in seconds
    """
    # Load the spectrogram
    spec = np.load(spec_file)
    
    # The spectrogram shape is typically (frequency_bins, time_frames)
    # Assuming standard hop size of 10ms per frame
    # This is an approximation and should be adjusted based on your exact preprocessing parameters
    time_frames = spec.shape[1]
    duration_seconds = time_frames * 0.01  # 10ms per frame
    
    return duration_seconds

def get_class_data(bonafide_dir: str, fake_dir: str) -> Dict[str, List[str]]:
    """
    Get file paths for each class.
    
    Args:
        bonafide_dir: Directory containing bonafide samples
        fake_dir: Directory containing fake samples
        
    Returns:
        Dictionary mapping class names to lists of file paths
    """
    class_data = {
        'bonafide': load_spectrogram_files(bonafide_dir),
        'fake': load_spectrogram_files(fake_dir)
    }
    
    return class_data

def get_class_durations(class_data: Dict[str, List[str]]) -> Dict[str, List[float]]:
    """
    Calculate durations for all files in each class.
    
    Args:
        class_data: Dictionary mapping class names to lists of file paths
        
    Returns:
        Dictionary mapping class names to lists of durations
    """
    class_durations = {}
    
    for class_name, file_paths in class_data.items():
        durations = []
        for file_path in file_paths:
            try:
                duration = get_spectrogram_duration(file_path)
                durations.append(duration)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                
        class_durations[class_name] = durations
        
    return class_durations