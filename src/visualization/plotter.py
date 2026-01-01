"""
Visualization tools for environmental data science
"""

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm, colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

from ..utils.config_loader import ConfigLoader

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class EnvironmentalPlotter:
    """Visualization tools for environmental data"""
    
    def __init__(self, style: str = "publication"):
        """
        Initialize plotter
        
        Parameters
        ----------
        style : str
            Plot style: "publication", "presentation", or "interactive"
        """
        self.style = style
        self.config = ConfigLoader().get('visualization', {})
        self._setup_plot_style()
        self._setup_colormaps()
        
        # Gulf of Mexico extent
        self.gulf_extent = [-98, -88, 18, 30]
    
    def _setup_plot_style(self) -> None:
        """Setup matplotlib style"""
        if self.style == "publication":
            plt.style.use('seaborn-v0_8-paper')
            mpl.rcParams.update({
                'figure.figsize': self.config.get('figure_size', [12, 8]),
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 11,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.dpi': self.config.get('dpi', 300),
                'savefig.dpi': self.config.get('dpi', 300),
                'lines.linewidth': 1.5,
                'axes.linewidth': 0.8,
                'grid.linewidth': 0.5,
                'grid.alpha': 0.3
            })
        elif self.style == "presentation":
            plt.style.use('seaborn-v0_8-darkgrid')
            mpl.rcParams.update({
                'figure.figsize': [14, 10],
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.dpi': 150
            })
        else:
            plt.style.use('default')
    
    def _setup_colormaps(self) -> None:
        """Setup colormaps for different variables"""
        self.colormaps = {}
        
        # Primary productivity (sequential blues)
        pp_colors = ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
                    '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
        self.colormaps['primary_productivity'] = colors.LinearSegmentedColormap.from_list(
            'primary_productivity', pp_colors
        )
        
        # Chlorophyll (sequential greens)
        chl_colors = ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b',
                     '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
        self.colormaps['chlorophyll'] = colors.LinearSegmentedColormap.from_list(
            'chlorophyll', chl_colors
        )
        
        # Currents (diverging)
        self.colormaps['current'] = cm.RdBu_r
        
        # Uncertainty (sequential)
        self.colormaps['uncertainty'] = cm.plasma
        
        # Oil concentration (sequential with transparency)
        oil_colors = cm.hot(np.linspace(0, 1, 256))
        oil_colors[:, 3] = np.linspace(0.2, 0.9, 256)
        self.colormaps['oil_concentration'] = colors.ListedColormap(oil_colors)
    
    def plot_spatial_field(self, data: xr.DataArray,
                          title: Optional[str] = None,
                          cmap: Optional[str] = None,
                          projection: ccrs.Projection = ccrs.PlateCarree(),
                          add_features: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot spatial field
        
        Parameters
        ----------
        data : xr.DataArray
            Data to plot
        title : str, optional
            Plot title
        cmap : str, optional
            Colormap name
        projection : ccrs.Projection
            Cartopy projection
        add_features : bool
            Add map features
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        # Create figure
        fig = plt.figure(figsize=mpl.rcParams['figure.figsize'])
        ax = plt.axes(projection=projection)
        ax.set_extent(self.gulf_extent, crs=ccrs.PlateCarree())
        
        # Add map features
        if add_features:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5, zorder=1)
            ax.add_feature(cfeature.STATES, linewidth=0.5, zorder=2)
            ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        
        # Determine colormap
        if cmap is None:
            var_name = data.name.lower() if hasattr(data, 'name') else ''
            if 'pp' in var_name or 'productivity' in var_name:
                cmap = self.colormaps['primary_productivity']
            elif 'chl' in var_name:
                cmap = self.colormaps['chlorophyll']
            elif 'current' in var_name or 'uo' in var_name or 'vo' in var_name:
                cmap = self.colormaps['current']
            elif 'uncertainty' in var_name:
                cmap = self.colormaps['uncertainty']
            else:
                cmap = 'viridis'
        
        # Plot data
        if hasattr(data, 'time'):
            # Plot mean over time
            plot_data = data.mean(dim='time', skipna=True)
        else:
            plot_data = data
        
        # Determine vmin and vmax
        if 'current' in str(cmap):
            # Diverging colormap centered at zero
            vmax = max(abs(plot_data.quantile(0.05).values),
                      abs(plot_data.quantile(0.95).values))
            vmin = -vmax
        else:
            # Sequential colormap
            vmin = plot_data.quantile(0.05).values
            vmax = plot_data.quantile(0.95).values
        
        # Create plot
        im = plot_data.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
            zorder=0
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, orientation='vertical',
                          pad=0.02, aspect=30, shrink=0.8)
        
        # Add units to colorbar if available
        if hasattr(data, 'attrs') and 'units' in data.attrs:
            cbar.set_label(data.attrs['units'])
        
        # Add title
        if title is None and hasattr(data, 'long_name'):
            title = data.long_name
        
        if title:
            ax.set_title(title, fontsize=mpl.rcParams['axes.titlesize'], pad=15)
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.5,
                         color='gray', alpha=0.5, linestyle='--',
                         zorder=3)
        gl.top_labels = False
        gl.right_labels = False
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=mpl.rcParams['savefig.dpi'])
            logger.info(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_uncertainty_decomposition(self, uncertainty_ds: xr.Dataset,
                                      time_idx: int = 0,
                                      save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot uncertainty decomposition
        
        Parameters
        ----------
        uncertainty_ds : xr.Dataset
            Uncertainty dataset
        time_idx : int
            Time index to plot
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        axes = axes.flatten()
        
        # Plot components
        components = [
            ('mean', 'Mean Prediction'),
            ('aleatoric_uncertainty', 'Aleatoric Uncertainty'),
            ('epistemic_uncertainty', 'Epistemic Uncertainty'),
            ('total_uncertainty', 'Total Uncertainty')
        ]
        
        for idx, (var_name, title) in enumerate(components):
            if var_name in uncertainty_ds:
                ax = axes[idx]
                ax.set_extent(self.gulf_extent, crs=ccrs.PlateCarree())
                
                # Add features
                ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
                ax.add_feature(cfeature.STATES, linewidth=0.5)
                
                # Get data
                if 'time' in uncertainty_ds[var_name].dims:
                    data = uncertainty_ds[var_name].isel(time=time_idx)
                else:
                    data = uncertainty_ds[var_name]
                
                # Determine colormap
                if 'uncertainty' in var_name:
                    cmap = self.colormaps['uncertainty']
                    vmax = data.quantile(0.95).values
                    vmin = 0
                else:
                    cmap = self.colormaps['primary_productivity']
                    vmin = data.quantile(0.05).values
                    vmax = data.quantile(0.95).values
                
                # Plot
                im = data.plot(
                    ax=ax,
                    transform=ccrs.PlateCarree(),
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    add_colorbar=False
                )
                
                # Add colorbar
                plt.colorbar(im, ax=ax, orientation='vertical',
                           pad=0.02, aspect=30, shrink=0.8)
                
                ax.set_title(title, fontsize=11, pad=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=mpl.rcParams['savefig.dpi'])
        
        return fig
    
    def plot_oil_spill_trajectory(self, trajectory_data: pd.DataFrame,
                                 concentration_field: Optional[xr.Dataset] = None,
                                 background_field: Optional[xr.DataArray] = None,
                                 title: str = "Oil Spill Trajectory",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot oil spill trajectory
        
        Parameters
        ----------
        trajectory_data : pd.DataFrame
            Trajectory data
        concentration_field : xr.Dataset, optional
            Oil concentration field
        background_field : xr.DataArray, optional
            Background field (e.g., currents)
        title : str
            Plot title
        save_path : str, optional
            Path to save figure
            
        Returns
        -------
        plt.Figure
            Figure object
        """
        # Create figure
        fig = plt.figure(figsize=(14, 10))
        
        # Create grid for subplots
        if concentration_field is not None or background_field is not None:
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1])
            ax_map = fig.add_subplot(gs[:, 0], projection=ccrs.PlateCarree())
            ax_time = fig.add_subplot(gs[0, 1])
            ax_stats = fig.add_subplot(gs[1, 1])
        else:
            ax_map = plt.axes(projection=ccrs.PlateCarree())
        
        # Setup map
        ax_map.set_extent(self.gulf_extent, crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=3)
        ax_map.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7, zorder=2)
        ax_map.add_feature(cfeature.STATES, linewidth=0.8, zorder=3)
        
        # Plot background field
        if background_field is not None:
            bg_mean = background_field.mean(dim='time', skipna=True)
            bg_plot = bg_mean.plot(
                ax=ax_map,
                transform=ccrs.PlateCarree(),
                cmap=self.colormaps['current'],
                add_colorbar=False,
                alpha=0.6,
                zorder=0
            )
            plt.colorbar(bg_plot, ax=ax_map, orientation='vertical',
                        pad=0.02, aspect=30, shrink=0.7)
        
        # Plot concentration field
        if concentration_field is not None:
            conc_mean = concentration_field['oil_concentration'].mean(dim='time', skipna=True)
            conc_plot = conc_mean.plot(
                ax=ax_map,
                transform=ccrs.PlateCarree(),
                cmap=self.colormaps['oil_concentration'],
                add_colorbar=False,
                alpha=0.7,
                zorder=1
            )
            plt.colorbar(conc_plot, ax=ax_map, orientation='vertical',
                        pad=0.15, aspect=30, shrink=0.7,
                        label='Oil Concentration (kg)')
        
        # Plot trajectories
        if not trajectory_data.empty:
            # Plot subset of particles for clarity
            unique_particles = trajectory_data['particle_id'].unique()
            plot_particles = unique_particles[:min(50, len(unique_particles))]
            
            for pid in plot_particles:
                particle_data = trajectory_data[trajectory_data['particle_id'] == pid]
                particle_data = particle_data.sort_values('time')
                
                # Plot trajectory line
                ax_map.plot(particle_data['longitude'], particle_data['latitude'],
                          transform=ccrs.PlateCarree(),
                          linewidth=0.5, alpha=0.3, color='black', zorder=2)
                
                # Plot start and end points
                if len(particle_data) > 0:
                    ax_map.scatter(particle_data['longitude'].iloc[0],
                                 particle_data['latitude'].iloc[0],
                                 transform=ccrs.PlateCarree(),
                                 s=20, color='green', edgecolor='black',
                                 linewidth=0.5, zorder=3, label='Start' if pid == plot_particles[0] else "")
                    
                    ax_map.scatter(particle_data['longitude'].iloc[-1],
                                 particle_data['latitude'].iloc[-1],
                                 transform=ccrs.PlateCarree(),
                                 s=20, color='red', edgecolor='black',
                                 linewidth=0.5, zorder=3, label='End' if pid == plot_particles[0] else "")
        
        # Add gridlines
        gl = ax_map.gridlines(draw_labels=True, linewidth=0.5,
                             color='gray', alpha=0.5, linestyle='--',
                             zorder=1)
        gl.top_labels = gl.right_labels = False
        
        # Add legend
        ax_map.legend(loc='upper left', fontsize=9)
        
        # Set title
        ax_map.set_title(title, fontsize=14, pad=20)
        
        # Add time series and statistics if side plots exist
        if concentration_field is not None or background_field is not None:
            # Time series of total mass
            if not trajectory_data.empty:
                mass_over_time = trajectory_data.groupby('time')['mass_kg'].sum()
                ax_time.plot(mass_over_time.index, mass_over_time.values,
                           linewidth=2, color='darkred')
                ax_time.set_xlabel('Time')
                ax_time.set_ylabel('Total Mass (kg)')
                ax_time.set_title('Mass Over Time')
                ax_time.grid(True, alpha=0.3)
                ax_time.tick_params(axis='x', rotation=45)
            
            # Statistics
            if not trajectory_data.empty:
                stats_text = [
                    f"Total Particles: {trajectory_data['particle_id'].nunique()}",
                    f"Total Mass: {trajectory_data['mass_kg'].sum():.0f} kg",
                    f"Mean Age: {trajectory_data['age_hours'].mean():.1f} hours"
                ]
                
                ax_stats.text(0.1, 0.9, '\n'.join(stats_text),
                            transform=ax_stats.transAxes,
                            fontsize=10, verticalalignment='top')
                ax_stats.set_title('Statistics')
                ax_stats.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=mpl.rcParams['savefig.dpi'])
        
        return fig
    
    def create_interactive_dashboard(self, dataset: xr.Dataset,
                                   uncertainty_results: Optional[Dict] = None,
                                   oil_spill_data: Optional[pd.DataFrame] = None,
                                   save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive dashboard
        
        Parameters
        ----------
        dataset : xr.Dataset
            Environmental dataset
        uncertainty_results : dict, optional
            Uncertainty analysis results
        oil_spill_data : pd.DataFrame, optional
            Oil spill trajectory data
        save_path : str, optional
            Path to save HTML dashboard
            
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Determine number of subplots
        n_plots = 0
        if 'pp' in dataset:
            n_plots += 1
        if 'chl' in dataset:
            n_plots += 1
        if all(v in dataset for v in ['uo', 'vo']):
            n_plots += 1
        if uncertainty_results is not None:
            n_plots += 1
        if oil_spill_data is not None:
            n_plots += 1
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=max(2, (n_plots + 1) // 2),
            subplot_titles=self._get_subplot_titles(dataset, uncertainty_results, oil_spill_data),
            specs=[[{'type': 'choroplethmapbox'} for _ in range(max(2, (n_plots + 1) // 2))]
                   for _ in range(2)]
        )
        
        plot_idx = 1
        row, col = 1, 1
        
        # Plot primary productivity
        if 'pp' in dataset:
            self._add_mapbox_trace(fig, dataset['pp'], 'Primary Productivity', row, col)
            plot_idx += 1
            col = (plot_idx - 1) % max(2, (n_plots + 1) // 2) + 1
            row = (plot_idx - 1) // max(2, (n_plots + 1) // 2) + 1
        
        # Plot chlorophyll
        if 'chl' in dataset:
            self._add_mapbox_trace(fig, dataset['chl'], 'Chlorophyll', row, col)
            plot_idx += 1
            col = (plot_idx - 1) % max(2, (n_plots + 1) // 2) + 1
            row = (plot_idx - 1) // max(2, (n_plots + 1) // 2) + 1
        
        # Plot current speed
        if all(v in dataset for v in ['uo', 'vo']):
            current_speed = np.sqrt(dataset['uo']**2 + dataset['vo']**2)
            self._add_mapbox_trace(fig, current_speed, 'Current Speed', row, col)
            plot_idx += 1
            col = (plot_idx - 1) % max(2, (n_plots + 1) // 2) + 1
            row = (plot_idx - 1) // max(2, (n_plots + 1) // 2) + 1
        
        # Plot uncertainty
        if uncertainty_results is not None and 'spatial_uncertainty' in uncertainty_results:
            unc_data = uncertainty_results['spatial_uncertainty']
            if 'total_uncertainty' in unc_data:
                self._add_mapbox_trace(fig, unc_data['total_uncertainty'], 'Uncertainty', row, col)
                plot_idx += 1
                col = (plot_idx - 1) % max(2, (n_plots + 1) // 2) + 1
                row = (plot_idx - 1) // max(2, (n_plots + 1) // 2) + 1
        
        # Plot oil spill trajectory
        if oil_spill_data is not None:
            self._add_trajectory_trace(fig, oil_spill_data, row, col)
        
        # Update layout
        fig.update_layout(
            title_text="Environmental Data Science Dashboard",
            showlegend=True,
            mapbox_style="carto-positron",
            mapbox=dict(
                center=dict(lat=24, lon=-93),
                zoom=5
            ),
            height=800
        )
        
        # Save if requested
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        return fig
    
    def _add_mapbox_trace(self, fig: go.Figure, data: xr.DataArray,
                         name: str, row: int, col: int) -> None:
        """Add Mapbox trace to figure"""
        # Prepare data
        if hasattr(data, 'time'):
            plot_data = data.isel(time=0)
        else:
            plot_data = data
        
        # Get coordinates
        lats = plot_data.lat.values
        lons = plot_data.lon.values
        
        # Create meshgrid
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        values = plot_data.values.flatten()
        
        # Create trace
        trace = go.Choroplethmapbox(
            geojson=self._create_grid_geojson(lats, lons),
            locations=np.arange(len(values)),
            z=values,
            colorscale="Viridis",
            zmin=np.nanpercentile(values, 5),
            zmax=np.nanpercentile(values, 95),
            marker_opacity=0.7,
            marker_line_width=0,
            name=name,
            colorbar=dict(title=plot_data.attrs.get('units', ''))
        )
        
        fig.add_trace(trace, row=row, col=col)
    
    def _add_trajectory_trace(self, fig: go.Figure, trajectory_data: pd.DataFrame,
                             row: int, col: int) -> None:
        """Add trajectory trace to figure"""
        # Plot subset of particles
        unique_particles = trajectory_data['particle_id'].unique()
        plot_particles = unique_particles[:min(20, len(unique_particles))]
        
        for pid in plot_particles:
            particle_data = trajectory_data[trajectory_data['particle_id'] == pid]
            particle_data = particle_data.sort_values('time')
            
            trace = go.Scattermapbox(
                lat=particle_data['latitude'],
                lon=particle_data['longitude'],
                mode='lines+markers',
                line=dict(width=1, color='red'),
                marker=dict(size=3, color='blue'),
                name=f'Particle {pid}',
                showlegend=pid == plot_particles[0]
            )
            
            fig.add_trace(trace, row=row, col=col)
    
    def _create_grid_geojson(self, lats: np.ndarray, lons: np.ndarray) -> Dict:
        """Create GeoJSON for grid cells"""
        features = []
        
        dlat = np.abs(lats[1] - lats[0])
        dlon = np.abs(lons[1] - lons[0])
        
        for i in range(len(lats) - 1):
            for j in range(len(lons) - 1):
                polygon = [
                    [lons[j], lats[i]],
                    [lons[j+1], lats[i]],
                    [lons[j+1], lats[i+1]],
                    [lons[j], lats[i+1]],
                    [lons[j], lats[i]]
                ]
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [polygon]
                    },
                    "properties": {
                        "id": i * len(lons) + j
                    }
                }
                features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def _get_subplot_titles(self, dataset: xr.Dataset,
                          uncertainty_results: Optional[Dict],
                          oil_spill_data: Optional[pd.DataFrame]) -> List[str]:
        """Get subplot titles based on available data"""
        titles = []
        
        if 'pp' in dataset:
            titles.append('Primary Productivity')
        if 'chl' in dataset:
            titles.append('Chlorophyll')
        if all(v in dataset for v in ['uo', 'vo']):
            titles.append('Current Speed')
        if uncertainty_results is not None:
            titles.append('Uncertainty')
        if oil_spill_data is not None:
            titles.append('Oil Spill Trajectory')
        
        return titles
