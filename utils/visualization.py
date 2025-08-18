"""
Interactive Visualization Interface
==================================

Interactive widgets and visualization functions for exploring
NEOS foundation model embeddings and similarity search results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from .similarity import (
    find_similar_objects, load_images, plot_umap_space, 
    get_random_object, print_similarity_statistics
)


class NEOSExplorer:
    """Interactive explorer for NEOS foundation model embeddings"""
    
    def __init__(self, embeddings, source_ids, labels, embeddings_2d, 
                 tabular_dict, visit_paths):
        self.embeddings = embeddings
        self.source_ids = source_ids
        self.labels = labels
        self.embeddings_2d = embeddings_2d
        self.tabular_dict = tabular_dict
        self.visit_paths = visit_paths
        
        # Set up matplotlib for large figures
        plt.rcParams['figure.figsize'] = [20, 15]
        plt.rcParams['figure.dpi'] = 100
        
        self._create_widgets()
        
    def _create_widgets(self):
        """Create interactive widgets"""
        
        # Control widgets
        self.num_similar_slider = widgets.IntSlider(
            value=8, min=4, max=12, step=1,
            description='Similar Objects:',
            style={'description_width': '120px'}
        )
        

        
        # Action buttons
        self.random_button = widgets.Button(
            description='Random Search',
            button_style='primary',
            layout=widgets.Layout(width='180px', height='40px')
        )
        
        self.real_button = widgets.Button(
            description='Random Real',
            button_style='success',
            layout=widgets.Layout(width='180px', height='40px')
        )
        
        self.fake_button = widgets.Button(
            description='Random Fake',
            button_style='danger',
            layout=widgets.Layout(width='180px', height='40px')
        )
        
        # Custom search
        self.custom_input = widgets.Text(
            placeholder='Enter source_id...',
            description='Custom Search:',
            style={'description_width': '120px'},
            layout=widgets.Layout(width='400px')
        )
        
        self.custom_button = widgets.Button(
            description='Search',
            button_style='info',
            layout=widgets.Layout(width='100px', height='32px')
        )
        
        # Output areas
        self.output_area = widgets.Output()
        
        # Connect events
        self.random_button.on_click(self._on_random_search)
        self.real_button.on_click(self._on_real_search)
        self.fake_button.on_click(self._on_fake_search)
        self.custom_button.on_click(self._on_custom_search)
        
    def _create_quality_filters(self):
        """Create simple quality filter widgets without complex synchronization"""
        
        # Get ranges from tabular data
        nc2snr_values = []
        nc2snrpm_values = []
        nc2mag_values = []
        
        for sid in self.source_ids:
            if sid in self.tabular_dict:
                row_data = self.tabular_dict[sid]
                
                nc2snr = row_data.get('NC2snr', np.nan)
                if pd.notna(nc2snr):
                    nc2snr_values.append(nc2snr)
                
                nc2snrpm = row_data.get('NC2snrpm', np.nan)
                if pd.notna(nc2snrpm):
                    nc2snrpm_values.append(nc2snrpm)
                
                nc2mag = row_data.get('NC2mag_1', np.nan)
                if pd.notna(nc2mag):
                    nc2mag_values.append(nc2mag)
        
        if not nc2snr_values:
            return []
        
        # Create simple filter widgets with readout enabled
        nc2snr_min, nc2snr_max = min(nc2snr_values), max(nc2snr_values)
        nc2snrpm_min, nc2snrpm_max = min(nc2snrpm_values), max(nc2snrpm_values)
        nc2mag_min, nc2mag_max = min(nc2mag_values), max(nc2mag_values)
        
        # Simple range sliders with built-in readout
        self.nc2snr_range = widgets.FloatRangeSlider(
            value=[nc2snr_min, nc2snr_max],
            min=nc2snr_min, max=nc2snr_max,
            step=(nc2snr_max - nc2snr_min) / 100,
            description='NC2snr:',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='400px')
        )
        
        self.nc2snrpm_range = widgets.FloatRangeSlider(
            value=[nc2snrpm_min, nc2snrpm_max],
            min=nc2snrpm_min, max=nc2snrpm_max,
            step=(nc2snrpm_max - nc2snrpm_min) / 100,
            description='NC2snrpm:',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='400px')
        )
        
        self.nc2mag_range = widgets.FloatRangeSlider(
            value=[nc2mag_min, nc2mag_max],
            min=nc2mag_min, max=nc2mag_max,
            step=(nc2mag_max - nc2mag_min) / 100,
            description='NC2mag:',
            readout=True,
            readout_format='.2f',
            layout=widgets.Layout(width='400px')
        )
        
        return [
            widgets.HTML("<p><i>Adjust ranges to filter objects by quality metrics</i></p>"),
            self.nc2snr_range,
            self.nc2snrpm_range,
            self.nc2mag_range,
        ]
    
    def _get_filter_params(self):
        """Get current filter parameters"""
        filters = {}
        
        if hasattr(self, 'nc2snr_range'):
            filters.update({
                'nc2snr_min': self.nc2snr_range.value[0],
                'nc2snr_max': self.nc2snr_range.value[1],
                'nc2snrpm_min': self.nc2snrpm_range.value[0],
                'nc2snrpm_max': self.nc2snrpm_range.value[1],
                'nc2mag_min': self.nc2mag_range.value[0],
                'nc2mag_max': self.nc2mag_range.value[1]
            })
        
        return filters
    
    def _get_quality_filter_mask(self):
        """Get mask for objects that pass current quality filter settings"""
        if not hasattr(self, 'nc2snr_range'):
            # No quality filters available, return all objects
            return np.ones(len(self.source_ids), dtype=bool)
        
        # Get current filter ranges
        nc2snr_range = self.nc2snr_range.value
        nc2snrpm_range = self.nc2snrpm_range.value
        nc2mag_range = self.nc2mag_range.value
        
        # Create mask for objects that pass all filters
        quality_mask = np.ones(len(self.source_ids), dtype=bool)
        
        for i, sid in enumerate(self.source_ids):
            if sid in self.tabular_dict:
                row_data = self.tabular_dict[sid]
                
                # Check NC2snr filter
                nc2snr = row_data.get('NC2snr', np.nan)
                if pd.notna(nc2snr):
                    if not (nc2snr_range[0] <= nc2snr <= nc2snr_range[1]):
                        quality_mask[i] = False
                        continue
                
                # Check NC2snrpm filter
                nc2snrpm = row_data.get('NC2snrpm', np.nan)
                if pd.notna(nc2snrpm):
                    if not (nc2snrpm_range[0] <= nc2snrpm <= nc2snrpm_range[1]):
                        quality_mask[i] = False
                        continue
                
                # Check NC2mag filter
                nc2mag = row_data.get('NC2mag_1', np.nan)
                if pd.notna(nc2mag):
                    if not (nc2mag_range[0] <= nc2mag <= nc2mag_range[1]):
                        quality_mask[i] = False
                        continue
        
        return quality_mask
    
    def _create_umap_region_selector(self):
        """Create UMAP region selector for the main interface"""
        
        x_min, x_max = self.embeddings_2d[:, 0].min(), self.embeddings_2d[:, 0].max()
        y_min, y_max = self.embeddings_2d[:, 1].min(), self.embeddings_2d[:, 1].max()
        
        # Create range sliders (default to full range)
        self.umap_x_range = widgets.FloatRangeSlider(
            value=[x_min, x_max],  # Default to full range
            min=x_min, max=x_max,
            step=(x_max - x_min) / 100,
            description='X Range:',
            readout_format='.2f',
            layout=widgets.Layout(width='400px')
        )
        
        self.umap_y_range = widgets.FloatRangeSlider(
            value=[y_min, y_max],  # Default to full range
            min=y_min, max=y_max,
            step=(y_max - y_min) / 100,
            description='Y Range:',
            readout_format='.2f',
            layout=widgets.Layout(width='400px')
        )
        
        # Compact action buttons
        self.reset_region_button = widgets.Button(
            description='Reset to Full',
            button_style='warning',
            layout=widgets.Layout(width='100px', height='30px')
        )
        
        # Tiny UMAP display - no scroller
        self.umap_output = widgets.Output(layout=widgets.Layout(height='260px'))
        
        # Connect events
        self.reset_region_button.on_click(self._on_reset_region)
        self.umap_x_range.observe(self._update_umap_display, names='value')
        self.umap_y_range.observe(self._update_umap_display, names='value')
        
        # Connect quality filter changes to UMAP update
        if hasattr(self, 'nc2snr_range'):
            self.nc2snr_range.observe(self._update_umap_display, names='value')
            self.nc2snrpm_range.observe(self._update_umap_display, names='value')
            self.nc2mag_range.observe(self._update_umap_display, names='value')
        
        # Initial UMAP display
        self._update_umap_display()
        
        return [
            widgets.HTML("<p><i>Define region to focus Quick Search (default: full space):</i></p>"),
            self.umap_x_range,
            self.umap_y_range,
            self.reset_region_button,
            self.umap_output
        ]
    
    def _update_umap_display(self, change=None):
        """Update the UMAP display with current region selection and quality filters"""
        with self.umap_output:
            clear_output(wait=True)
            
            # Create tiny UMAP plot
            fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))
            
            # Apply quality filters to get valid objects
            quality_mask = self._get_quality_filter_mask()
            
            # Create masks
            real_mask = (self.labels == 'Real') & quality_mask
            fake_mask = (self.labels == 'Fake') & quality_mask
            
            # Plot all points (smaller for compact view)
            ax.scatter(self.embeddings_2d[fake_mask, 0], self.embeddings_2d[fake_mask, 1],
                      c='red', alpha=0.5, s=8, label=f'Fake: {fake_mask.sum():,}')
            
            ax.scatter(self.embeddings_2d[real_mask, 0], self.embeddings_2d[real_mask, 1],
                      c='green', alpha=0.6, s=10, label=f'Real: {real_mask.sum():,}')
            
            # Highlight selected region
            x_range = self.umap_x_range.value
            y_range = self.umap_y_range.value
            
            # Only draw rectangle if not full range
            x_min, x_max = self.embeddings_2d[:, 0].min(), self.embeddings_2d[:, 0].max()
            y_min, y_max = self.embeddings_2d[:, 1].min(), self.embeddings_2d[:, 1].max()
            
            if not (x_range[0] == x_min and x_range[1] == x_max and 
                   y_range[0] == y_min and y_range[1] == y_max):
                # Draw selection rectangle
                from matplotlib.patches import Rectangle
                rect = Rectangle((x_range[0], y_range[0]), 
                               x_range[1] - x_range[0], y_range[1] - y_range[0],
                               linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.15)
                ax.add_patch(rect)
            
            # Count objects in region (already filtered by quality)
            in_region_mask = ((self.embeddings_2d[:, 0] >= x_range[0]) & 
                            (self.embeddings_2d[:, 0] <= x_range[1]) &
                            (self.embeddings_2d[:, 1] >= y_range[0]) & 
                            (self.embeddings_2d[:, 1] <= y_range[1]))
            
            # Combine region and quality filters
            region_and_quality_mask = in_region_mask & quality_mask
            real_in_region = np.sum(region_and_quality_mask & (self.labels == 'Real'))
            fake_in_region = np.sum(region_and_quality_mask & (self.labels == 'Fake'))
            total_in_region = np.sum(region_and_quality_mask)
            
            # Very compact plot styling
            ax.set_xlabel('UMAP X', fontsize=8)
            ax.set_ylabel('UMAP Y', fontsize=8)
            
            # Show only region count (which is already filtered)
            ax.set_title(f'Region: {total_in_region} objects (R:{real_in_region}, F:{fake_in_region})', 
                        fontsize=8, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)
            
            plt.tight_layout()
            plt.show()
    

    
    def _on_reset_region(self, button):
        """Reset region to full UMAP space"""
        x_min, x_max = self.embeddings_2d[:, 0].min(), self.embeddings_2d[:, 0].max()
        y_min, y_max = self.embeddings_2d[:, 1].min(), self.embeddings_2d[:, 1].max()
        
        self.umap_x_range.value = [x_min, x_max]
        self.umap_y_range.value = [y_min, y_max]
    
    def _get_umap_region_objects(self, object_type='both'):
        """Get objects from the current UMAP region that also pass quality filters"""
        # Get objects in the selected UMAP region
        x_range = self.umap_x_range.value
        y_range = self.umap_y_range.value
        
        in_region_mask = ((self.embeddings_2d[:, 0] >= x_range[0]) & 
                        (self.embeddings_2d[:, 0] <= x_range[1]) &
                        (self.embeddings_2d[:, 1] >= y_range[0]) & 
                        (self.embeddings_2d[:, 1] <= y_range[1]))
        
        # Apply quality filters
        quality_mask = self._get_quality_filter_mask()
        
        # Filter by object type
        if object_type == 'real':
            type_mask = self.labels == 'Real'
        elif object_type == 'fake':
            type_mask = self.labels == 'Fake'
        else:  # both
            type_mask = np.ones(len(self.labels), dtype=bool)
        
        # Combine ALL masks: region + quality + type
        valid_mask = in_region_mask & quality_mask & type_mask
        valid_indices = np.where(valid_mask)[0]
        
        return valid_indices
    
    def _on_random_search(self, button):
        """Handle random search button click - uses UMAP region"""
        with self.output_area:
            clear_output(wait=True)
            
            valid_indices = self._get_umap_region_objects('both')
            
            if len(valid_indices) == 0:
                print("No objects found matching quality filters in the selected UMAP region!")
                return
            
            # Select random object from filtered region
            selected_idx = np.random.choice(valid_indices)
            selected_source_id = self.source_ids[selected_idx]
            
            print(f"Selected from filtered region: {selected_source_id} ({self.labels[selected_idx]})")
            print(f"Available objects (quality filtered + region): {len(valid_indices)}")
            self._display_similarity_results(selected_source_id, self.num_similar_slider.value)
    
    def _on_real_search(self, button):
        """Handle real search button click - uses UMAP region"""
        with self.output_area:
            clear_output(wait=True)
            
            valid_indices = self._get_umap_region_objects('real')
            
            if len(valid_indices) == 0:
                print("No REAL objects found matching quality filters in the selected UMAP region!")
                return
            
            # Select random real object from filtered region
            selected_idx = np.random.choice(valid_indices)
            selected_source_id = self.source_ids[selected_idx]
            
            print(f"Selected REAL from filtered region: {selected_source_id}")
            print(f"Available real objects (quality filtered + region): {len(valid_indices)}")
            self._display_similarity_results(selected_source_id, self.num_similar_slider.value)
    
    def _on_fake_search(self, button):
        """Handle fake search button click - uses UMAP region"""
        with self.output_area:
            clear_output(wait=True)
            
            valid_indices = self._get_umap_region_objects('fake')
            
            if len(valid_indices) == 0:
                print("No FAKE objects found matching quality filters in the selected UMAP region!")
                return
            
            # Select random fake object from filtered region
            selected_idx = np.random.choice(valid_indices)
            selected_source_id = self.source_ids[selected_idx]
            
            print(f"Selected FAKE from filtered region: {selected_source_id}")
            print(f"Available fake objects (quality filtered + region): {len(valid_indices)}")
            self._display_similarity_results(selected_source_id, self.num_similar_slider.value)
    
    def _on_custom_search(self, button):
        """Handle custom search button click"""
        with self.output_area:
            clear_output(wait=True)
            if self.custom_input.value.strip():
                print(f"Custom search for: {self.custom_input.value.strip()}")
                self._display_similarity_results(self.custom_input.value.strip(), self.num_similar_slider.value)
            else:
                print("Please enter a source_id to search")
    
    def _display_similarity_results(self, query_source_id, num_similar=10):
        """Display comprehensive similarity search results"""
        
        print(f"SIMILARITY SEARCH RESULTS (Using 512D Learned Embeddings)")
        print("=" * 70)
        print(f"Query Object: {query_source_id}")
        
        # Find similar objects
        query_info, results = find_similar_objects(
            query_source_id, self.embeddings, self.source_ids, self.labels, num_similar
        )
        
        if query_info is None:
            print(results)  # Error message
            return
        
        print(f"Query Type: {query_info['label']}")
        print(f"Finding {num_similar} most similar objects using learned embeddings...")
        print()
        
        # Create visualization
        n_cols = 4  # 4 objects per row for larger images  
        n_rows = 2 + (num_similar - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols*2, figsize=(70, 10*n_rows), dpi=100)
        
        # Adjust spacing
        plt.subplots_adjust(hspace=0.1, wspace=0.05)
        
        # Display query object (first row, centered)
        query_nc1, query_nc2, success = load_images(query_source_id, self.visit_paths)
        
        if success:
            # Query NC1
            ax_nc1 = axes[0, 2] if n_rows > 1 else axes[2]
            ax_nc1.imshow(query_nc1, cmap='gray', origin='lower')
            ax_nc1.set_title('NC1', fontsize=32, fontweight='bold')
            ax_nc1.axis('off')
            
            # Add similarity score
            ax_nc1.text(0.5, 0.95, f'Similarity: 1.000', 
                       transform=ax_nc1.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                       fontsize=40, fontweight='bold')
            
            # Query NC2
            ax_nc2 = axes[0, 3] if n_rows > 1 else axes[3]
            ax_nc2.imshow(query_nc2, cmap='gray', origin='lower')
            ax_nc2.set_title('NC2', fontsize=32, fontweight='bold')
            ax_nc2.axis('off')
            
            # Add label (Real = Green, Fake = Red)
            color = 'green' if query_info['label'] == 'Real' else 'red'
            ax_nc2.text(0.5, 0.95, f'{query_info["label"]}', 
                       transform=ax_nc2.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle="round,pad=0.6", facecolor=color, alpha=0.9),
                       fontsize=40, fontweight='bold', color='white')
            
            # Add source ID below NC1
            ax_nc1.text(0.5, -0.02, query_source_id, 
                       transform=ax_nc1.transAxes, ha='center', va='top',
                       fontsize=30, fontweight='bold', clip_on=False)
        
        # Hide unused axes in query row
        for i in range(n_cols*2):
            if i not in [2, 3]:
                if n_rows > 1:
                    axes[0, i].axis('off')
                else:
                    axes[i].axis('off')
        
        # Display similar objects
        for i, result in enumerate(results[:min(num_similar, (n_rows-1)*n_cols)]):
            row = 1 + i // n_cols
            col_start = (i % n_cols) * 2
            
            # Load images
            sim_nc1, sim_nc2, success = load_images(result['source_id'], self.visit_paths)
            
            if success and row < n_rows:
                # NC1 image
                ax_nc1 = axes[row, col_start]
                ax_nc1.imshow(sim_nc1, cmap='gray', origin='lower')
                ax_nc1.set_title('NC1', fontsize=32, fontweight='bold')
                ax_nc1.axis('off')
                
                # Add similarity score
                ax_nc1.text(0.5, 0.95, f'Sim: {result["similarity"]:.3f}', 
                           transform=ax_nc1.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.8),
                           fontsize=40, fontweight='bold')
                
                # NC2 image
                ax_nc2 = axes[row, col_start + 1]
                ax_nc2.imshow(sim_nc2, cmap='gray', origin='lower')
                ax_nc2.set_title('NC2', fontsize=32, fontweight='bold')
                ax_nc2.axis('off')
                
                # Add real/fake label
                color = 'green' if result['label'] == 'Real' else 'red'
                ax_nc2.text(0.5, 0.95, result['label'], 
                           transform=ax_nc2.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle="round,pad=0.6", facecolor=color, alpha=0.9),
                           fontsize=40, fontweight='bold', color='white')
                
                # Add source ID below NC1
                ax_nc1.text(0.5, -0.02, result['source_id'], 
                           transform=ax_nc1.transAxes, ha='center', va='top',
                           fontsize=30, fontweight='bold', clip_on=False)
        
        # Hide remaining unused axes
        total_images = len(results)
        for i in range(total_images, n_cols * (n_rows - 1)):
            row = 1 + i // n_cols
            col_start = (i % n_cols) * 2
            if row < n_rows:
                axes[row, col_start].axis('off')
                axes[row, col_start + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print_similarity_statistics(query_info, results)
        
        # Show 2D embedding plot
        print(f"2D EMBEDDING SPACE WITH HIGHLIGHTED QUERY:")
        plot_umap_space(self.embeddings_2d, self.labels, self.source_ids, 
                       highlight_source_id=query_source_id)
    

    
    def display_interface(self):
        """Display the unified interactive interface with integrated UMAP explorer"""
        
        # Create quality filters
        filter_widgets = self._create_quality_filters()
        
        # Create UMAP region selector
        umap_widgets = self._create_umap_region_selector()
        
        # Create main control panel
        control_panel = widgets.VBox([
            widgets.HTML("<h2>NEOS Foundation Model - Interactive Similarity Explorer</h2>"),
            
            widgets.HTML("<h3>Search Settings</h3>"),
            self.num_similar_slider,
            
            widgets.HTML("<h3>Custom Search</h3>"),
            widgets.HBox([self.custom_input, self.custom_button]),
            
            widgets.HTML("<h3>Quality Filters (for Random Search)</h3>"),
            *filter_widgets,
            
            widgets.HTML("<h3>UMAP Region (for Random Search)</h3>"),
            *umap_widgets,
            
            widgets.HTML("<h3>Quick Search</h3>"),
            widgets.HBox([self.random_button, self.real_button, self.fake_button]),
            
            widgets.HTML("<hr><h3>Results</h3>"),
        ])
        
        # Display everything
        display(control_panel)
        display(self.output_area)
