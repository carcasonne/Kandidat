import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from pathlib import Path

class SpectrogramVisualizer:
    """
    A class for visualizing spectrograms with publication-quality settings.
    """
    
    def __init__(self, output_dir="output"):
        """
        Initialize the SpectrogramVisualizer.
        
        Args:
            output_dir (str): Directory where to save the visualization outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Publication-quality figure settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 10,
            'axes.linewidth': 0.8,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.6,
            'ytick.minor.width': 0.6,
            'legend.fontsize': 10,
            'legend.frameon': False,
            'figure.dpi': 300
        })
        
        # Define custom colormaps
        self.colormaps = {
            'viridis': plt.cm.viridis,
            'inferno': plt.cm.inferno,
            'magma': plt.cm.magma,
            'plasma': plt.cm.plasma,
            'jet': plt.cm.jet,
            # Custom perceptually uniform colormap suitable for spectrograms
            'spec': self._create_spectrogram_colormap()
        }
    
    def _create_spectrogram_colormap(self):
        """
        Create a custom colormap optimized for spectrograms.
        
        Returns:
            matplotlib.colors.LinearSegmentedColormap: Custom colormap
        """
        # Colors from white (low energy) to dark blue (high energy)
        colors = [(1, 1, 1), (0.8, 0.8, 1), (0.5, 0.5, 0.9), 
                 (0.2, 0.2, 0.8), (0, 0, 0.6)]
        return LinearSegmentedColormap.from_list('spec', colors, N=256)
    
    def visualize_spectrogram(self, spec_data, title=None, ax=None, cmap='inferno', 
                              normalize=True, log_scale=True, colorbar=True):
        """
        Visualize a single spectrogram.
        
        Args:
            spec_data (numpy.ndarray): The spectrogram data
            title (str, optional): Title for the plot
            ax (matplotlib.axes.Axes, optional): Axes to plot on
            cmap (str): Colormap to use
            normalize (bool): Whether to normalize the data
            log_scale (bool): Whether to apply log scaling
            colorbar (bool): Whether to show colorbar
            
        Returns:
            matplotlib.axes.Axes: The axes with the plot
        """
        # Create axis if not provided
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        
        # Process data
        data = spec_data.copy()
        
        # Apply log scaling if requested
        if log_scale and np.all(data >= 0):
            # Add small constant to avoid log(0)
            data = np.log10(data + 1e-10)
        
        # Normalize if requested
        if normalize:
            data_min = np.min(data)
            data_max = np.max(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        # Get colormap
        colormap = self.colormaps.get(cmap, plt.cm.inferno)
        
        # Plot the spectrogram
        im = ax.imshow(data, aspect='auto', origin='lower', cmap=colormap, interpolation='nearest')
        
        # Add colorbar if requested
        if colorbar:
            plt.colorbar(im, ax=ax, pad=0.01, format='%.2f')
        
        # Set title if provided
        if title:
            ax.set_title(title, fontsize=12)
        
        # Set labels
        ax.set_xlabel('Time Frame', fontsize=10)
        ax.set_ylabel('Frequency Bin', fontsize=10)
        
        # Make ticks look nice
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        
        return ax
    
    def compare_spectrograms(self, specs_dict, ncols=3, figsize=(15, 10), 
                             cmap='inferno', normalize=True, log_scale=True, 
                             suptitle=None, output_filename=None):
        """
        Compare multiple spectrograms in a grid layout.
        
        Args:
            specs_dict (dict): Dictionary with titles as keys and spectrogram data as values
            ncols (int): Number of columns in the grid
            figsize (tuple): Figure size (width, height)
            cmap (str): Colormap to use
            normalize (bool): Whether to normalize each spectrogram
            log_scale (bool): Whether to apply log scaling
            suptitle (str, optional): Super title for the figure
            output_filename (str, optional): If provided, save figure to this path
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        n_specs = len(specs_dict)
        nrows = (n_specs + ncols - 1) // ncols  # Ceiling division
        
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.3, hspace=0.4)
        
        for i, (title, spec_data) in enumerate(specs_dict.items()):
            row, col = i // ncols, i % ncols
            ax = fig.add_subplot(gs[row, col])
            self.visualize_spectrogram(
                spec_data, 
                title=title,
                ax=ax,
                cmap=cmap,
                normalize=normalize,
                log_scale=log_scale,
                colorbar=True
            )
        
        # Add super title if provided
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97] if suptitle else None)
        
        # Save figure if requested
        if output_filename:
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_path}")
        
        return fig
    
    def visualize_real_vs_fake(self, real_specs, fake_specs, figsize=(15, 8), 
                               cmap='inferno', normalize=True, log_scale=True,
                               output_filename=None):
        """
        Create a figure comparing real and fake spectrograms side by side.
        
        Args:
            real_specs (dict): Dictionary with titles and real spectrogram data
            fake_specs (dict): Dictionary with titles and fake spectrogram data
            figsize (tuple): Figure size (width, height)
            cmap (str): Colormap to use
            normalize (bool): Whether to normalize each spectrogram
            log_scale (bool): Whether to apply log scaling
            output_filename (str, optional): If provided, save figure to this path
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        n_real = len(real_specs)
        n_fake = len(fake_specs)
        n_pairs = min(n_real, n_fake)
        
        fig, axes = plt.subplots(n_pairs, 2, figsize=figsize)
        
        # Handle case with only one pair
        if n_pairs == 1:
            axes = [axes]
        
        for i, ((real_title, real_data), (fake_title, fake_data)) in enumerate(
            zip(list(real_specs.items())[:n_pairs], list(fake_specs.items())[:n_pairs])
        ):
            # Visualize real spectrogram
            self.visualize_spectrogram(
                real_data,
                title=f"Real: {real_title}",
                ax=axes[i][0],
                cmap=cmap,
                normalize=normalize,
                log_scale=log_scale
            )
            
            # Visualize fake spectrogram
            self.visualize_spectrogram(
                fake_data,
                title=f"Fake: {fake_title}",
                ax=axes[i][1],
                cmap=cmap,
                normalize=normalize,
                log_scale=log_scale
            )
        
        plt.suptitle("Comparison of Real vs. Fake Spectrograms", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Save figure if requested
        if output_filename:
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_path}")
        
        return fig
    
    def visualize_difference(self, real_spec, fake_spec, title=None, 
                             figsize=(15, 5), cmap='RdBu_r', output_filename=None):
        """
        Visualize the difference between real and fake spectrograms.
        
        Args:
            real_spec (numpy.ndarray): Real spectrogram data
            fake_spec (numpy.ndarray): Fake spectrogram data (must be same shape as real)
            title (str, optional): Title for the figure
            figsize (tuple): Figure size (width, height)
            cmap (str): Colormap to use for difference (defaults to RdBu_r)
            output_filename (str, optional): If provided, save figure to this path
            
        Returns:
            matplotlib.figure.Figure: The figure with the plots
        """
        if real_spec.shape != fake_spec.shape:
            print(f"Error: Shapes don't match ({real_spec.shape} vs {fake_spec.shape})")
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Visualize real spectrogram
        self.visualize_spectrogram(
            real_spec,
            title="Real Spectrogram",
            ax=axes[0],
            cmap='inferno'
        )
        
        # Visualize fake spectrogram
        self.visualize_spectrogram(
            fake_spec,
            title="Fake Spectrogram",
            ax=axes[1],
            cmap='inferno'
        )
        
        # Compute and visualize difference
        diff = real_spec - fake_spec
        
        # For difference, use RdBu_r colormap centered at zero
        vmax = max(abs(np.min(diff)), abs(np.max(diff)))
        vmin = -vmax
        
        im = axes[2].imshow(diff, aspect='auto', origin='lower', 
                          cmap=plt.cm.get_cmap(cmap), vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=axes[2], pad=0.01)
        
        axes[2].set_title("Difference (Real - Fake)")
        axes[2].set_xlabel('Time Frame')
        axes[2].set_ylabel('Frequency Bin')
        
        # Add super title if provided
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95] if title else None)
        
        # Save figure if requested
        if output_filename:
            output_path = os.path.join(self.output_dir, output_filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {output_path}")
        
        return fig