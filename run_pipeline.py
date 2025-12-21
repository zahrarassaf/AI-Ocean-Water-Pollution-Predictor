#!/usr/bin/env python3
"""
Main pipeline script for Environmental Data Science Portfolio
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Import project modules
from src.data.cmems_processor import CMEMSDataProcessor
from src.analysis.uncertainty_analyzer import PrimaryProductivityUncertaintyAnalyzer
from src.models.oil_spill_tracker import OilSpillModel, SpillEvent, OilType
from src.visualization.plotter import EnvironmentalPlotter
from src.utils.config_loader import ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def run_data_processing(config: dict, download_dir: str = None) -> bool:
    """Run data processing pipeline"""
    try:
        logger.info("=" * 70)
        logger.info("DATA PROCESSING PIPELINE")
        logger.info("=" * 70)
        
        processor = CMEMSDataProcessor(config)
        
        if download_dir:
            dataset = processor.process_pipeline(download_dir)
        else:
            # Load existing processed data
            processed_file = Path("data/processed/gulf_mexico_processed.nc")
            if processed_file.exists():
                import xarray as xr
                dataset = xr.open_dataset(processed_file)
                logger.info(f"Loaded existing dataset from {processed_file}")
            else:
                raise FileNotFoundError("No processed data found. Please provide download directory.")
        
        logger.info("Data processing completed successfully")
        return True, dataset
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return False, None

def run_uncertainty_analysis(dataset, config: dict) -> bool:
    """Run uncertainty analysis"""
    try:
        logger.info("=" * 70)
        logger.info("UNCERTAINTY ANALYSIS")
        logger.info("=" * 70)
        
        analyzer = PrimaryProductivityUncertaintyAnalyzer(dataset, config)
        results = analyzer.run_ensemble_analysis()
        
        # Save results
        output_dir = Path("results/uncertainty_analysis")
        analyzer.save_results(output_dir)
        
        # Create visualizations
        plotter = EnvironmentalPlotter(style="publication")
        
        if 'spatial_uncertainty' in results:
            fig = plotter.plot_uncertainty_decomposition(
                results['spatial_uncertainty'],
                save_path=output_dir / "uncertainty_decomposition.png"
            )
            plt.close(fig)
        
        logger.info("Uncertainty analysis completed successfully")
        return True, results
        
    except Exception as e:
        logger.error(f"Uncertainty analysis failed: {e}")
        return False, None

def run_oil_spill_simulation(dataset, config: dict, scenario: str = "test") -> bool:
    """Run oil spill simulation"""
    try:
        logger.info("=" * 70)
        logger.info("OIL SPILL SIMULATION")
        logger.info("=" * 70)
        
        # Prepare current data
        current_vars = ['uo', 'vo']
        if all(var in dataset for var in current_vars):
            current_data = dataset[current_vars]
        else:
            raise ValueError("Current data not found in dataset")
        
        # Create model
        model = OilSpillModel(current_data, config.get('oil_spill', {}))
        
        # Define spill scenario
        if scenario == "test":
            spill = SpillEvent(
                latitude=28.0,
                longitude=-90.0,
                start_time=datetime(2023, 6, 1),
                duration_hours=24.0,
                total_volume_m3=1000.0,
                oil_type=OilType.MEDIUM_CRUDE
            )
        elif scenario == "deepwater_horizon":
            spill = SpillEvent(
                latitude=28.7366,
                longitude=-88.3659,
                start_time=datetime(2010, 4, 20, 22, 0),
                duration_hours=87 * 24,
                total_volume_m3=779000,
                oil_type=OilType.MEDIUM_CRUDE,
                release_depth_m=1500
            )
        else:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        # Add spill and run simulation
        model.add_spill_event(spill)
        trajectory_data = model.run_simulation()
        
        # Save results
        output_dir = Path(f"results/oil_spill/{scenario}")
        model.save_results(output_dir)
        
        # Create visualizations
        plotter = EnvironmentalPlotter(style="publication")
        
        # Calculate concentration field
        concentration_field = model.calculate_concentration_field()
        
        # Plot trajectory
        fig = plotter.plot_oil_spill_trajectory(
            trajectory_data,
            concentration_field=concentration_field,
            save_path=output_dir / "trajectory_plot.png"
        )
        plt.close(fig)
        
        logger.info(f"Oil spill simulation completed for scenario: {scenario}")
        return True, trajectory_data
        
    except Exception as e:
        logger.error(f"Oil spill simulation failed: {e}")
        return False, None

def create_dashboard(dataset, uncertainty_results=None, oil_spill_data=None):
    """Create interactive dashboard"""
    try:
        logger.info("=" * 70)
        logger.info("CREATING INTERACTIVE DASHBOARD")
        logger.info("=" * 70)
        
        plotter = EnvironmentalPlotter(style="interactive")
        
        dashboard = plotter.create_interactive_dashboard(
            dataset=dataset,
            uncertainty_results=uncertainty_results,
            oil_spill_data=oil_spill_data,
            save_path="results/dashboard.html"
        )
        
        logger.info("Dashboard created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Dashboard creation failed: {e}")
        return False

def main():
    """Main pipeline function"""
    parser = argparse.ArgumentParser(
        description="Environmental Data Science Pipeline"
    )
    
    parser.add_argument(
        "--mode",
        choices=["all", "process", "analyze", "simulate", "dashboard"],
        default="all",
        help="Pipeline mode"
    )
    
    parser.add_argument(
        "--download-dir",
        help="Directory containing downloaded CMEMS files"
    )
    
    parser.add_argument(
        "--scenario",
        choices=["test", "deepwater_horizon"],
        default="test",
        help="Oil spill scenario"
    )
    
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.config
    
    logger.info("=" * 70)
    logger.info("ENVIRONMENTAL DATA SCIENCE PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Scenario: {args.scenario}")
    
    # Run pipeline components
    success = True
    dataset = None
    uncertainty_results = None
    oil_spill_data = None
    
    # Data processing
    if args.mode in ["all", "process"]:
        process_success, dataset = run_data_processing(config, args.download_dir)
        success = success and process_success
    
    # Uncertainty analysis
    if args.mode in ["all", "analyze"] and success and dataset is not None:
        analyze_success, uncertainty_results = run_uncertainty_analysis(dataset, config)
        success = success and analyze_success
    
    # Oil spill simulation
    if args.mode in ["all", "simulate"] and success and dataset is not None:
        simulate_success, oil_spill_data = run_oil_spill_simulation(
            dataset, config, args.scenario
        )
        success = success and simulate_success
    
    # Dashboard creation
    if args.mode in ["all", "dashboard"] and success and dataset is not None:
        dashboard_success = create_dashboard(
            dataset, uncertainty_results, oil_spill_data
        )
        success = success and dashboard_success
    
    # Final status
    logger.info("=" * 70)
    if success:
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    else:
        logger.error("PIPELINE FAILED")
    logger.info("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
