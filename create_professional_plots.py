# create_professional_plots.py
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Professional settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [14, 10]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

print("=" * 70)
print("PROFESSIONAL OCEAN DATA VISUALIZATION")
print("=" * 70)

class OceanDataVisualizer:
    def __init__(self):
        self.data_path = "data/raw/"
        self.output_path = "results/visualizations/"
        self.models_path = "models/"
        
        # Create output directories
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "static"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "interactive"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "maps"), exist_ok=True)
        
    def load_netcdf_data(self, filename):
        """Load NetCDF file"""
        filepath = os.path.join(self.data_path, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            return None
        
        print(f"Loading: {filename}")
        try:
            ds = xr.open_dataset(filepath)
            print(f"  Variables: {list(ds.data_vars.keys())}")
            print(f"  Dimensions: {dict(ds.dims)}")
            return ds
        except Exception as e:
            print(f"  Error: {e}")
            return None
    
    def create_chlorophyll_heatmap(self):
        """Create heatmap of chlorophyll concentration"""
        print("\n1. Creating Chlorophyll Heatmap...")
        
        # Try different possible files
        chl_files = [
            "chlorophyll_concentration.nc",
            "diffuse_attenuation.nc"
        ]
        
        ds = None
        for f in chl_files:
            ds = self.load_netcdf_data(f)
            if ds is not None:
                break
        
        if ds is None:
            print("No chlorophyll data found")
            return
        
        # Find chlorophyll variable
        chl_vars = [var for var in ds.data_vars if 'CHL' in var or 'chlorophyll' in var.lower()]
        if not chl_vars:
            print("No chlorophyll variable found")
            return
        
        chl_var = chl_vars[0]
        chl_data = ds[chl_var]
        
        # Select a time slice if available
        if 'time' in chl_data.dims and len(chl_data.time) > 0:
            chl_slice = chl_data.isel(time=0)
        else:
            chl_slice = chl_data
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Basic heatmap
        if 'latitude' in chl_slice.dims and 'longitude' in chl_slice.dims:
            im1 = axes[0, 0].imshow(chl_slice.values, cmap='YlGnBu', aspect='auto')
            axes[0, 0].set_title(f'{chl_var} - Spatial Distribution')
            axes[0, 0].set_xlabel('Longitude Index')
            axes[0, 0].set_ylabel('Latitude Index')
            plt.colorbar(im1, ax=axes[0, 0], label='Chlorophyll (mg/m³)')
        
        # 2. Histogram
        axes[0, 1].hist(chl_data.values.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('Chlorophyll Distribution')
        axes[0, 1].set_xlabel('Chlorophyll (mg/m³)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Statistical summary boxplot
        if 'time' in chl_data.dims and len(chl_data.time) > 3:
            time_slices = min(10, len(chl_data.time))
            sample_data = []
            sample_labels = []
            
            for i in range(time_slices):
                sample_data.append(chl_data.isel(time=i).values.flatten()[:1000])
                sample_labels.append(f'T{i+1}')
            
            axes[1, 0].boxplot(sample_data, labels=sample_labels)
            axes[1, 0].set_title('Chlorophyll Variation Over Time')
            axes[1, 0].set_ylabel('Chlorophyll (mg/m³)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Kernel Density Estimate
        import scipy.stats as stats
        data_flat = chl_data.values.flatten()
        data_flat = data_flat[~np.isnan(data_flat)]
        data_flat = data_flat[data_flat < np.percentile(data_flat, 99)]  # Remove outliers
        
        kde = stats.gaussian_kde(data_flat)
        x_range = np.linspace(data_flat.min(), data_flat.max(), 100)
        axes[1, 1].plot(x_range, kde(x_range), color='darkred', linewidth=2)
        axes[1, 1].fill_between(x_range, kde(x_range), alpha=0.3, color='darkred')
        axes[1, 1].set_title('Probability Density Function')
        axes[1, 1].set_xlabel('Chlorophyll (mg/m³)')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "static", "chlorophyll_analysis.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_path, "static", "chlorophyll_analysis.pdf"))
        plt.show()
        
        print("✓ Chlorophyll heatmap saved")
    
    def create_geographic_maps(self):
        """Create geographic maps of ocean parameters"""
        print("\n2. Creating Geographic Maps...")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            
            # Load any dataset with lat/lon
            ds = self.load_netcdf_data("secchi_depth.nc")
            if ds is None:
                ds = self.load_netcdf_data("chlorophyll_concentration.nc")
            
            if ds is None:
                print("No data for mapping")
                return
            
            # Find a data variable
            data_vars = list(ds.data_vars.keys())
            if not data_vars:
                print("No data variables found")
                return
            
            var_name = data_vars[0]
            data = ds[var_name]
            
            # Take first time slice if available
            if 'time' in data.dims and len(data.time) > 0:
                data_slice = data.isel(time=0)
            else:
                data_slice = data
            
            # Create map
            fig = plt.figure(figsize=(15, 10))
            
            # Different map projections
            projections = [
                (ccrs.PlateCarree(), 'Plate Carree'),
                (ccrs.Robinson(), 'Robinson'),
                (ccrs.Mercator(), 'Mercator'),
                (ccrs.Orthographic(central_longitude=0, central_latitude=20), 'Orthographic')
            ]
            
            for idx, (proj, title) in enumerate(projections, 1):
                ax = plt.subplot(2, 2, idx, projection=proj)
                
                # Add map features
                ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.7)
                ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
                ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
                ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.3)
                
                # Plot data if it has lat/lon dimensions
                if 'latitude' in data_slice.dims and 'longitude' in data_slice.dims:
                    # Simple scatter for demonstration
                    lats = ds.latitude.values.flatten()[::100]  # Sample every 100th point
                    lons = ds.longitude.values.flatten()[::100]
                    vals = data_slice.values.flatten()[::100]
                    
                    scatter = ax.scatter(lons, lats, c=vals, cmap='viridis', 
                                        s=10, alpha=0.6, transform=ccrs.PlateCarree())
                
                ax.set_title(f'{var_name}\n{title} Projection', fontsize=10)
                ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5)
                
                if idx == 1:
                    plt.colorbar(scatter, ax=ax, orientation='horizontal', 
                                pad=0.05, label=var_name)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "maps", "geographic_distribution.png"), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✓ Geographic maps saved")
            
        except ImportError:
            print("Cartopy not installed. Install with: pip install cartopy")
        except Exception as e:
            print(f"Map creation error: {e}")
    
    def create_time_series_analysis(self):
        """Create time series analysis plots"""
        print("\n3. Creating Time Series Analysis...")
        
        # Try to find data with time dimension
        for filename in os.listdir(self.data_path):
            if filename.endswith('.nc'):
                ds = self.load_netcdf_data(filename)
                if ds is None:
                    continue
                
                if 'time' in ds.dims and len(ds.time) > 1:
                    print(f"Found time series data in: {filename}")
                    
                    # Create time series for each variable
                    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
                    
                    for idx, var_name in enumerate(list(ds.data_vars.keys())[:3]):
                        if idx >= 3:
                            break
                        
                        data = ds[var_name]
                        
                        # Calculate mean over space for each time point
                        if 'latitude' in data.dims and 'longitude' in data.dims:
                            time_series = data.mean(dim=['latitude', 'longitude'])
                        else:
                            time_series = data.mean(dim=data.dims[1:]) if len(data.dims) > 1 else data
                        
                        # Plot time series
                        axes[idx].plot(ds.time.values, time_series.values, 
                                     marker='o', markersize=3, linewidth=2, alpha=0.7)
                        axes[idx].set_title(f'{var_name} - Time Series', fontsize=12)
                        axes[idx].set_xlabel('Time')
                        axes[idx].set_ylabel(var_name)
                        axes[idx].grid(True, alpha=0.3)
                        
                        # Add trend line
                        if len(time_series) > 1:
                            x_num = np.arange(len(time_series))
                            z = np.polyfit(x_num, time_series.values, 1)
                            p = np.poly1d(z)
                            axes[idx].plot(ds.time.values, p(x_num), 
                                         'r--', linewidth=1, label='Trend')
                            axes[idx].legend()
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_path, "static", f"time_series_{filename.split('.')[0]}.png"), 
                               dpi=300, bbox_inches='tight')
                    plt.show()
                    break
        
        print("✓ Time series analysis saved")
    
    def create_interactive_plots(self):
        """Create interactive Plotly plots"""
        print("\n4. Creating Interactive Visualizations...")
        
        try:
            # Load chlorophyll data
            ds = None
            for f in ["diffuse_attenuation.nc", "chlorophyll_concentration.nc"]:
                ds = self.load_netcdf_data(f)
                if ds is not None:
                    break
            
            if ds is None:
                print("No data for interactive plots")
                return
            
            # Find chlorophyll data
            chl_vars = [var for var in ds.data_vars if 'CHL' in var]
            if not chl_vars:
                print("No chlorophyll data found")
                return
            
            chl_var = chl_vars[0]
            chl_data = ds[chl_var]
            
            # Prepare data for plotting
            if 'latitude' in chl_data.dims and 'longitude' in chl_data.dims:
                # Take first time slice
                if 'time' in chl_data.dims and len(chl_data.time) > 0:
                    chl_slice = chl_data.isel(time=0)
                else:
                    chl_slice = chl_data
                
                # Sample data for performance
                lats = ds.latitude.values.flatten()[::10]
                lons = ds.longitude.values.flatten()[::10]
                values = chl_slice.values.flatten()[::10]
                
                # Create interactive scatter plot
                fig = px.scatter_geo(
                    lat=lats,
                    lon=lons,
                    color=values,
                    color_continuous_scale='Viridis',
                    title=f'Interactive {chl_var} Distribution',
                    labels={'color': 'Chlorophyll (mg/m³)'},
                    projection='natural earth'
                )
                
                fig.update_geos(
                    showland=True,
                    landcolor="lightgray",
                    showocean=True,
                    oceancolor="lightblue",
                    showlakes=True,
                    lakecolor="blue",
                    showrivers=True,
                    rivercolor="blue"
                )
                
                # Save interactive plot
                fig.write_html(os.path.join(self.output_path, "interactive", "chlorophyll_map.html"))
                
                # Create 3D surface plot
                if len(values) > 1000:
                    # Create heatmap
                    fig2 = go.Figure(data=go.Heatmap(
                        z=chl_slice.values[:100, :100],  # First 100x100 pixels
                        colorscale='YlGnBu'
                    ))
                    
                    fig2.update_layout(
                        title=f'{chl_var} Heatmap',
                        xaxis_title='Longitude Index',
                        yaxis_title='Latitude Index'
                    )
                    
                    fig2.write_html(os.path.join(self.output_path, "interactive", "chlorophyll_heatmap.html"))
                
                print("✓ Interactive plots saved as HTML")
                
        except Exception as e:
            print(f"Interactive plot error: {e}")
    
    def create_model_performance_plots(self):
        """Create plots showing model performance"""
        print("\n5. Creating Model Performance Visualizations...")
        
        try:
            # Load feature importance if available
            importance_file = os.path.join(self.models_path, "smart_feature_importance.csv")
            if os.path.exists(importance_file):
                importance_df = pd.read_csv(importance_file)
                
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))
                
                # Feature importance bar plot
                importance_df = importance_df.sort_values('importance', ascending=True)
                axes[0].barh(importance_df['feature'], importance_df['importance'], 
                           color='steelblue', alpha=0.7)
                axes[0].set_xlabel('Importance')
                axes[0].set_title('Feature Importance Ranking')
                axes[0].grid(True, alpha=0.3, axis='x')
                
                # Feature importance pie chart
                axes[1].pie(importance_df['importance'], 
                          labels=importance_df['feature'], 
                          autopct='%1.1f%%',
                          startangle=90,
                          colors=plt.cm.Set3(np.linspace(0, 1, len(importance_df))))
                axes[1].set_title('Feature Importance Distribution')
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, "static", "feature_importance.png"), 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
                print("✓ Model performance plots saved")
                
        except Exception as e:
            print(f"Model plot error: {e}")
    
    def create_correlation_matrix(self):
        """Create correlation matrix of ocean parameters"""
        print("\n6. Creating Correlation Analysis...")
        
        # Try to load multiple datasets and combine
        all_data = []
        
        for filename in os.listdir(self.data_path):
            if filename.endswith('.nc'):
                ds = self.load_netcdf_data(filename)
                if ds is None:
                    continue
                
                # Extract data variables
                for var_name in list(ds.data_vars.keys())[:2]:  # Take first 2 variables from each
                    data = ds[var_name]
                    flat_data = data.values.flatten()[:10000]  # Take up to 10000 points
                    all_data.append((f"{filename.split('.')[0]}_{var_name}", flat_data))
        
        if len(all_data) < 2:
            print("Not enough data for correlation analysis")
            return
        
        # Create correlation matrix
        min_len = min(len(d) for _, d in all_data)
        corr_data = np.array([d[:min_len] for _, d in all_data]).T
        corr_df = pd.DataFrame(corr_data, columns=[name for name, _ in all_data])
        
        # Calculate correlation matrix
        correlation_matrix = corr_df.corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f', 
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8})
        
        plt.title('Correlation Matrix of Ocean Parameters', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_path, "static", "correlation_matrix.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Correlation analysis saved")
    
    def create_quality_report(self):
        """Create data quality report with visualizations"""
        print("\n7. Creating Data Quality Report...")
        
        quality_report = []
        
        for filename in os.listdir(self.data_path):
            if filename.endswith('.nc'):
                ds = self.load_netcdf_data(filename)
                if ds is None:
                    continue
                
                report = {
                    'filename': filename,
                    'size_mb': os.path.getsize(os.path.join(self.data_path, filename)) / (1024*1024),
                    'variables': list(ds.data_vars.keys()),
                    'num_variables': len(ds.data_vars),
                    'dimensions': dict(ds.dims),
                    'has_time': 'time' in ds.dims,
                    'has_latlon': ('latitude' in ds.dims and 'longitude' in ds.dims) or 
                                 ('lat' in ds.dims and 'lon' in ds.dims)
                }
                
                # Calculate data statistics
                stats_data = []
                for var in ds.data_vars:
                    data = ds[var].values.flatten()
                    valid_data = data[~np.isnan(data)]
                    if len(valid_data) > 0:
                        stats_data.append({
                            'variable': var,
                            'mean': np.mean(valid_data),
                            'std': np.std(valid_data),
                            'min': np.min(valid_data),
                            'max': np.max(valid_data),
                            'nan_percent': (np.isnan(data).sum() / len(data)) * 100
                        })
                
                report['statistics'] = stats_data
                quality_report.append(report)
        
        # Save quality report
        report_file = os.path.join(self.output_path, "data_quality_report.txt")
        with open(report_file, 'w') as f:
            f.write("OCEAN DATA QUALITY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            for report in quality_report:
                f.write(f"File: {report['filename']}\n")
                f.write(f"Size: {report['size_mb']:.1f} MB\n")
                f.write(f"Variables: {report['num_variables']}\n")
                f.write(f"Dimensions: {report['dimensions']}\n")
                f.write(f"Has time dimension: {report['has_time']}\n")
                f.write(f"Has geographic coordinates: {report['has_latlon']}\n\n")
                
                f.write("Variable Statistics:\n")
                for stat in report.get('statistics', []):
                    f.write(f"  {stat['variable']}:\n")
                    f.write(f"    Mean: {stat['mean']:.4f}, Std: {stat['std']:.4f}\n")
                    f.write(f"    Range: [{stat['min']:.4f}, {stat['max']:.4f}]\n")
                    f.write(f"    NaN: {stat['nan_percent']:.1f}%\n\n")
                
                f.write("-" * 50 + "\n\n")
        
        print(f"✓ Quality report saved: {report_file}")
        
        # Create visualization of data quality
        if quality_report:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # File sizes
            filenames = [r['filename'] for r in quality_report]
            sizes = [r['size_mb'] for r in quality_report]
            axes[0, 0].bar(filenames, sizes, color='steelblue', alpha=0.7)
            axes[0, 0].set_title('File Sizes (MB)')
            axes[0, 0].set_ylabel('Size (MB)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Number of variables
            num_vars = [r['num_variables'] for r in quality_report]
            axes[0, 1].bar(filenames, num_vars, color='forestgreen', alpha=0.7)
            axes[0, 1].set_title('Number of Variables per File')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
            
            # NaN percentages (average)
            nan_percents = []
            for report in quality_report:
                if report.get('statistics'):
                    avg_nan = np.mean([s['nan_percent'] for s in report['statistics']])
                    nan_percents.append(avg_nan)
                else:
                    nan_percents.append(0)
            
            axes[1, 0].bar(filenames, nan_percents, color='coral', alpha=0.7)
            axes[1, 0].set_title('Average NaN Percentage')
            axes[1, 0].set_ylabel('NaN %')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3, axis='y')
            
            # Data dimensions
            total_points = []
            for report in quality_report:
                dims = report['dimensions']
                total = np.prod(list(dims.values())) if dims else 0
                total_points.append(total / 1e6)  # Convert to millions
            
            axes[1, 1].bar(filenames, total_points, color='purple', alpha=0.7)
            axes[1, 1].set_title('Total Data Points (Millions)')
            axes[1, 1].set_ylabel('Millions of Points')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "static", "data_quality_summary.png"), 
                       dpi=300, bbox_inches='tight')
            plt.show()
    
    def create_all_visualizations(self):
        """Create all visualizations"""
        print("\n" + "=" * 70)
        print("STARTING COMPREHENSIVE VISUALIZATION PIPELINE")
        print("=" * 70)
        
        # Check if data exists
        if not os.path.exists(self.data_path):
            print(f"Data path not found: {self.data_path}")
            return
        
        # Create all visualizations
        self.create_chlorophyll_heatmap()
        self.create_geographic_maps()
        self.create_time_series_analysis()
        self.create_interactive_plots()
        self.create_model_performance_plots()
        self.create_correlation_matrix()
        self.create_quality_report()
        
        print("\n" + "=" * 70)
        print("VISUALIZATION PIPELINE COMPLETED!")
        print("=" * 70)
        print(f"\nOutput saved in: {self.output_path}")
        print("\nGenerated files:")
        print("  results/visualizations/static/ - Static PNG/PDF plots")
        print("  results/visualizations/interactive/ - Interactive HTML plots")
        print("  results/visualizations/maps/ - Geographic maps")
        print("  results/visualizations/data_quality_report.txt - Quality report")

# Installation instructions
def show_installation_guide():
    print("\n" + "=" * 70)
    print("INSTALLATION GUIDE FOR VISUALIZATION")
    print("=" * 70)
    print("\nRequired packages (install if missing):")
    print("  pip install xarray netCDF4 matplotlib seaborn plotly pandas numpy")
    print("  pip install cartopy scipy")  # Optional but recommended
    print("\nFor geographic maps (cartopy), you might need additional system dependencies.")
    print("On Ubuntu/Debian: sudo apt-get install libproj-dev proj-data proj-bin")
    print("On Windows: Use conda: conda install -c conda-forge cartopy")

def main():
    """Main function"""
    # Show installation guide first
    show_installation_guide()
    
    # Ask user if they want to continue
    response = input("\nDo you want to create visualizations? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y', '']:
        visualizer = OceanDataVisualizer()
        visualizer.create_all_visualizations()
    else:
        print("Visualization cancelled.")

if __name__ == "__main__":
    main()
