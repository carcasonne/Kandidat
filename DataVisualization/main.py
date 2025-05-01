import os
import sys
import numpy as np

# Add parent directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our custom modules
from Core.SpectrogramBuilder import SpectrogramBuilder
from Visualizers.SpectrogramVisualizer import SpectrogramVisualizer

def main():
    """
    Main function to demonstrate loading and visualizing spectrograms.
    """
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Initializing spectrogram builder and visualizer...")
    
    # Initialize builder and visualizer
    builder = SpectrogramBuilder(data_dir="Data")
    visualizer = SpectrogramVisualizer(output_dir=output_dir)
    
    # Load sample spectrograms
    print("Loading sample spectrograms...")
    real_specs, fake_specs = builder.get_sample_spectrograms(num_real=3, num_fake=3)
    
    # Check if we have spectrograms
    if not real_specs or not fake_specs:
        print("No spectrograms found. Please check your data directories.")
        return
    
    print(f"Loaded {len(real_specs)} real and {len(fake_specs)} fake spectrograms.")
    
    # Print some information about the spectrograms
    sample_real = next(iter(real_specs.values()))
    sample_fake = next(iter(fake_specs.values()))
    
    print("\nReal spectrogram stats:")
    real_stats = builder.get_spectrogram_stats(sample_real)
    for key, value in real_stats.items():
        print(f"  {key}: {value}")
    
    print("\nFake spectrogram stats:")
    fake_stats = builder.get_spectrogram_stats(sample_fake)
    for key, value in fake_stats.items():
        print(f"  {key}: {value}")
    
    # Visualize all real spectrograms
    print("\nVisualizing real spectrograms...")
    visualizer.compare_spectrograms(
        real_specs, 
        ncols=3, 
        suptitle="Real Spectrograms", 
        cmap="inferno",
        output_filename="real_spectrograms.png"
    )
    
    # Visualize all fake spectrograms
    print("Visualizing fake spectrograms...")
    visualizer.compare_spectrograms(
        fake_specs, 
        ncols=3, 
        suptitle="Fake Spectrograms", 
        cmap="inferno",
        output_filename="fake_spectrograms.png"
    )
    
    # Compare real vs fake spectrograms
    print("Creating real vs fake comparison...")
    visualizer.visualize_real_vs_fake(
        real_specs, 
        fake_specs, 
        cmap="inferno",
        output_filename="real_vs_fake_comparison.png"
    )
    
    # Visualize differences for the first pair
    print("Visualizing differences...")
    real_key = list(real_specs.keys())[0]
    fake_key = list(fake_specs.keys())[0]
    
    visualizer.visualize_difference(
        real_specs[real_key], 
        fake_specs[fake_key],
        title=f"Difference between {real_key} and {fake_key}",
        output_filename="spectrogram_difference.png"
    )
    
    # Try some different colormaps
    print("Testing different colormaps...")
    for cmap in ['magma']:
        visualizer.compare_spectrograms(
            {k: real_specs[k] for k in list(real_specs.keys())[:2]}, 
            ncols=2, 
            suptitle=f"Spectrogram Visualization with {cmap} Colormap", 
            cmap=cmap,
            output_filename=f"colormap_{cmap}.png"
        )
    
    print("\nAll visualizations complete! Check the output directory.")

if __name__ == "__main__":
    main()