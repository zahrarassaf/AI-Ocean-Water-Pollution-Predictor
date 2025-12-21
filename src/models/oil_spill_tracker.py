"""
Oil spill trajectory modeling using Lagrangian particle tracking
"""

import numpy as np
import xarray as xr
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from scipy import interpolate, stats, spatial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ..utils.config_loader import ConfigLoader

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class OilType(Enum):
    """Oil type classification based on API gravity"""
    LIGHT_CRUDE = {"api": 35.0, "viscosity": 5.0, "density": 850.0, "evaporation_rate": 0.3}
    MEDIUM_CRUDE = {"api": 30.0, "viscosity": 15.0, "density": 876.0, "evaporation_rate": 0.25}
    HEAVY_CRUDE = {"api": 22.0, "viscosity": 50.0, "density": 920.0, "evaporation_rate": 0.15}
    DIESEL = {"api": 40.0, "viscosity": 3.0, "density": 830.0, "evaporation_rate": 0.4}
    BUNKER_FUEL = {"api": 15.0, "viscosity": 200.0, "density": 960.0, "evaporation_rate": 0.1}

@dataclass
class SpillEvent:
    """Oil spill event specification"""
    latitude: float
    longitude: float
    start_time: pd.Timestamp
    duration_hours: float = 24.0
    total_volume_m3: float = 1000.0
    oil_type: OilType = OilType.MEDIUM_CRUDE
    release_depth_m: float = 0.0
    release_rate_m3_per_hour: Optional[float] = None
    
    def __post_init__(self):
        if self.release_rate_m3_per_hour is None:
            self.release_rate_m3_per_hour = self.total_volume_m3 / self.duration_hours

class LagrangianParticle:
    """Individual particle for oil spill tracking"""
    
    def __init__(self, particle_id: int, latitude: float, longitude: float,
                 release_time: pd.Timestamp, oil_type: OilType, mass_kg: float):
        self.particle_id = particle_id
        self.latitude = latitude
        self.longitude = longitude
        self.release_time = release_time
        self.oil_type = oil_type
        self.mass_kg = mass_kg
        self.age_hours = 0.0
        self.trajectory = []
        self.active = True
        self.evaporated_mass = 0.0
        self.dispersed_mass = 0.0
        self.beached = False
        
        # Record initial position
        self.record_position(release_time)
    
    def record_position(self, time: pd.Timestamp) -> None:
        """Record current position in trajectory"""
        self.trajectory.append({
            'time': time,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'mass': self.mass_kg,
            'age': self.age_hours
        })
    
    def update_position(self, dlat: float, dlon: float, time: pd.Timestamp) -> None:
        """Update particle position"""
        self.latitude += dlat
        self.longitude += dlon
        self.age_hours = (time - self.release_time).total_seconds() / 3600.0
        self.record_position(time)
    
    def apply_weathering(self, wind_speed: float, wave_height: float,
                        water_temp: float, time_step_hours: float) -> None:
        """Apply weathering processes to particle"""
        # Evaporation (simplified Fay model)
        evaporation_factor = self.oil_type.value["evaporation_rate"]
        evaporation_rate = evaporation_factor * wind_speed * np.sqrt(self.age_hours + 1)
        evaporated = self.mass_kg * evaporation_rate * time_step_hours / 24.0
        evaporated = min(evaporated, self.mass_kg * 0.95)  # Cap at 95%
        
        self.mass_kg -= evaporated
        self.evaporated_mass += evaporated
        
        # Natural dispersion (simplified Delvigne & Sweeney model)
        if wave_height > 0.5:
            dispersion_rate = 0.001 * wave_height**2 * self.oil_type.value["viscosity"]**-0.5
            dispersed = self.mass_kg * dispersion_rate * time_step_hours
            dispersed = min(dispersed, self.mass_kg * 0.8)  # Cap at 80%
            
            self.mass_kg -= dispersed
            self.dispersed_mass += dispersed
        
        # Deactivate if mass is too low
        if self.mass_kg < 0.1:
            self.active = False
    
    def beach_particle(self) -> None:
        """Mark particle as beached"""
        self.beached = True
        self.active = False
    
    def get_state(self) -> Dict:
        """Get current particle state"""
        return {
            'particle_id': self.particle_id,
            'latitude': self.latitude,
            'longitude': self.longitude,
            'age_hours': self.age_hours,
            'mass_kg': self.mass_kg,
            'active': self.active,
            'beached': self.beached,
            'evaporated_mass': self.evaporated_mass,
            'dispersed_mass': self.dispersed_mass
        }

class OilSpillModel:
    """Oil spill trajectory model with Lagrangian particle tracking"""
    
    def __init__(self, current_data: xr.Dataset, config: Optional[Dict] = None):
        """
        Initialize oil spill model
        
        Parameters
        ----------
        current_data : xr.Dataset
            Ocean current data with u and v components
        config : dict, optional
            Model configuration
        """
        self.current_data = current_data
        self.config = config or ConfigLoader().get('models.oil_spill', {})
        
        # Model components
        self.particles = []
        self.spill_events = []
        self.trajectory_data = None
        
        # Interpolators
        self.u_interpolators = []
        self.v_interpolators = []
        self._initialize_interpolators()
        
        # Random seed for reproducibility
        np.random.seed(self.config.get('random_seed', 42))
    
    def _initialize_interpolators(self) -> None:
        """Initialize interpolators for current fields"""
        # Extract time, lat, lon
        times = self.current_data.time.values
        lats = self.current_data.lat.values
        lons = self.current_data.lon.values
        
        # Create interpolators for each time step
        for t_idx in range(len(times)):
            u_data = self.current_data['uo'].isel(time=t_idx).values
            v_data = self.current_data['vo'].isel(time=t_idx).values
            
            u_interp = interpolate.RegularGridInterpolator(
                (lats, lons), u_data,
                bounds_error=False, fill_value=0.0
            )
            v_interp = interpolate.RegularGridInterpolator(
                (lats, lons), v_data,
                bounds_error=False, fill_value=0.0
            )
            
            self.u_interpolators.append(u_interp)
            self.v_interpolators.append(v_interp)
        
        self.times = times
        self.lats = lats
        self.lons = lons
        
        logger.info(f"Initialized interpolators for {len(times)} time steps")
    
    def add_spill_event(self, spill: SpillEvent) -> None:
        """
        Add spill event to model
        
        Parameters
        ----------
        spill : SpillEvent
            Spill event specification
        """
        self.spill_events.append(spill)
        
        # Create particles for this spill
        particles_per_hour = self.config.get('particle_count', 1000) / spill.duration_hours
        total_particles = 0
        
        current_time = spill.start_time
        end_time = spill.start_time + pd.Timedelta(hours=spill.duration_hours)
        
        while current_time < end_time and total_particles < self.config.get('particle_count', 1000):
            # Number of particles for this time step
            particles_this_step = int(particles_per_hour * self.config.get('time_step_hours', 1.0))
            particles_this_step = min(
                particles_this_step,
                self.config.get('particle_count', 1000) - total_particles
            )
            
            if particles_this_step > 0:
                # Mass per particle
                mass_per_particle = (
                    spill.release_rate_m3_per_hour *
                    self.config.get('time_step_hours', 1.0) *
                    spill.oil_type.value["density"] /
                    particles_this_step
                )
                
                # Create particles with spatial dispersion
                for i in range(particles_this_step):
                    # Add random offset (approximately 100m)
                    lat_offset = np.random.normal(0, 0.0009)  # ~100m
                    lon_offset = np.random.normal(0, 0.0009 / np.cos(np.deg2rad(spill.latitude)))
                    
                    particle = LagrangianParticle(
                        particle_id=len(self.particles) + 1,
                        latitude=spill.latitude + lat_offset,
                        longitude=spill.longitude + lon_offset,
                        release_time=current_time,
                        oil_type=spill.oil_type,
                        mass_kg=mass_per_particle
                    )
                    
                    self.particles.append(particle)
                    total_particles += 1
            
            current_time += pd.Timedelta(hours=self.config.get('time_step_hours', 1.0))
        
        logger.info(f"Created {total_particles} particles for spill at "
                   f"({spill.latitude:.4f}, {spill.longitude:.4f})")
    
    def get_current_velocity(self, latitude: float, longitude: float,
                           time: pd.Timestamp) -> Tuple[float, float]:
        """
        Get current velocity at specific location and time
        
        Parameters
        ----------
        latitude : float
            Latitude in degrees
        longitude : float
            Longitude in degrees
        time : pd.Timestamp
            Time
            
        Returns
        -------
        tuple
            (u, v) current velocities in m/s
        """
        # Find nearest time index
        time_deltas = np.abs(self.times - time)
        time_idx = np.argmin(time_deltas)
        
        # Clamp index
        time_idx = max(0, min(time_idx, len(self.times) - 1))
        
        # Interpolate velocities
        try:
            u = self.u_interpolators[time_idx]([latitude, longitude])[0]
            v = self.v_interpolators[time_idx]([latitude, longitude])[0]
            return float(u), float(v)
        except:
            return 0.0, 0.0
    
    def calculate_wind_drift(self, latitude: float, longitude: float,
                           time: pd.Timestamp) -> Tuple[float, float]:
        """
        Calculate wind-induced drift
        
        Parameters
        ----------
        latitude : float
            Latitude
        longitude : float
            Longitude
        time : pd.Timestamp
            Time
            
        Returns
        -------
        tuple
            Wind drift velocities (u_wind, v_wind) in m/s
        """
        # Simplified wind model (would be replaced with actual wind data)
        # Typical Gulf of Mexico wind: 5-10 m/s from southeast
        wind_speed = 7.5  # m/s
        wind_direction = 135.0  # degrees from north (southeast)
        
        # Convert to radians (mathematical convention)
        wind_rad = np.deg2rad(90 - wind_direction)
        
        # Calculate components
        u_wind = wind_speed * np.cos(wind_rad)
        v_wind = wind_speed * np.sin(wind_rad)
        
        # Apply wind drift factor (typically 0.03 for oil)
        drift_factor = self.config.get('wind_drift_factor', 0.03)
        u_wind *= drift_factor
        v_wind *= drift_factor
        
        return u_wind, v_wind
    
    def calculate_stokes_drift(self, u_wind: float, v_wind: float) -> Tuple[float, float]:
        """
        Calculate Stokes drift from wind
        
        Parameters
        ----------
        u_wind : float
            Wind u-component
        v_wind : float
            Wind v-component
            
        Returns
        -------
        tuple
            Stokes drift velocities (u_stokes, v_stokes)
        """
        stokes_factor = self.config.get('stokes_drift_factor', 0.01)
        
        u_stokes = u_wind * stokes_factor
        v_stokes = v_wind * stokes_factor
        
        return u_stokes, v_stokes
    
    def calculate_turbulent_diffusion(self) -> Tuple[float, float]:
        """
        Calculate turbulent diffusion displacement
        
        Returns
        -------
        tuple
            Diffusion displacements (dlat, dlon) in degrees
        """
        diffusion_coeff = self.config.get('diffusion_coefficient', 0.01)  # km²/hour
        
        # Convert to degrees²/hour (1 degree ≈ 111 km)
        diffusion_deg2 = diffusion_coeff / (111.0**2)
        
        # Calculate displacement (Brownian motion)
        dt = self.config.get('time_step_hours', 1.0)
        std_dev = np.sqrt(2 * diffusion_deg2 * dt)
        
        dlat = np.random.normal(0, std_dev)
        dlon = np.random.normal(0, std_dev)
        
        return dlat, dlon
    
    def move_particle(self, particle: LagrangianParticle,
                     current_time: pd.Timestamp) -> bool:
        """
        Move a single particle based on all forcing mechanisms
        
        Parameters
        ----------
        particle : LagrangianParticle
            Particle to move
        current_time : pd.Timestamp
            Current simulation time
            
        Returns
        -------
        bool
            True if particle is still active
        """
        if not particle.active:
            return False
        
        # Get ocean currents
        u_current, v_current = self.get_current_velocity(
            particle.latitude, particle.longitude, current_time
        )
        
        # Get wind drift
        u_wind, v_wind = self.calculate_wind_drift(
            particle.latitude, particle.longitude, current_time
        )
        
        # Get Stokes drift
        u_stokes, v_stokes = self.calculate_stokes_drift(u_wind, v_wind)
        
        # Get turbulent diffusion
        dlat_diff, dlon_diff = self.calculate_turbulent_diffusion()
        
        # Total velocity components
        u_total = u_current + u_wind + u_stokes
        v_total = v_current + v_wind + v_stokes
        
        # Convert velocities to displacement (degrees)
        dt = self.config.get('time_step_hours', 1.0) * 3600.0  # Convert to seconds
        
        # Convert m/s to degrees (1 degree ≈ 111 km)
        dlat_advection = (v_total * dt) / 111000.0
        dlon_advection = (u_total * dt) / (111000.0 * np.cos(np.deg2rad(particle.latitude)))
        
        # Total displacement
        dlat = dlat_advection + dlat_diff
        dlon = dlon_advection + dlon_diff
        
        # Apply weathering
        if self.config.get('weathering_enabled', True):
            # Simplified environmental parameters
            wind_speed = np.sqrt(u_wind**2 + v_wind**2) / self.config.get('wind_drift_factor', 0.03)
            wave_height = 1.0  # Would come from wave model
            water_temp = 25.0  # Would come from SST data
            
            particle.apply_weathering(
                wind_speed, wave_height, water_temp,
                self.config.get('time_step_hours', 1.0)
            )
        
        # Update position
        new_lat = particle.latitude + dlat
        new_lon = particle.longitude + dlon
        
        # Check boundaries (Gulf of Mexico)
        if self._is_outside_boundaries(new_lat, new_lon):
            particle.active = False
            return False
        
        # Check for beaching (simplified)
        if self._is_near_coast(new_lat, new_lon):
            particle.beach_particle()
            return False
        
        particle.update_position(dlat, dlon, current_time)
        return particle.active
    
    def _is_outside_boundaries(self, lat: float, lon: float) -> bool:
        """Check if location is outside study area"""
        lat_min = self.config.get('region_lat_min', 18.0)
        lat_max = self.config.get('region_lat_max', 30.0)
        lon_min = self.config.get('region_lon_min', -98.0)
        lon_max = self.config.get('region_lon_max', -88.0)
        
        return not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max)
    
    def _is_near_coast(self, lat: float, lon: float, threshold_km: float = 1.0) -> bool:
        """Check if location is near coast (simplified)"""
        # Simplified coast check - would use actual coastline data
        # For Gulf of Mexico, coasts are at boundaries
        lat_min = self.config.get('region_lat_min', 18.0)
        lat_max = self.config.get('region_lat_max', 30.0)
        lon_min = self.config.get('region_lon_min', -98.0)
        lon_max = self.config.get('region_lon_max', -88.0)
        
        # Convert threshold to degrees
        threshold_deg = threshold_km / 111.0
        
        near_north_coast = abs(lat - lat_max) < threshold_deg
        near_south_coast = abs(lat - lat_min) < threshold_deg
        near_east_coast = abs(lon - lon_max) < threshold_deg
        near_west_coast = abs(lon - lon_min) < threshold_deg
        
        return near_north_coast or near_south_coast or near_east_coast or near_west_coast
    
    def run_simulation(self) -> pd.DataFrame:
        """
        Run complete oil spill simulation
        
        Returns
        -------
        pd.DataFrame
            Simulation results
        """
        if not self.spill_events:
            raise ValueError("No spill events defined. Use add_spill_event() first.")
        
        logger.info("Starting oil spill simulation")
        
        # Determine simulation time range
        start_time = min(spill.start_time for spill in self.spill_events)
        max_hours = self.config.get('max_simulation_hours', 168)
        end_time = start_time + pd.Timedelta(hours=max_hours)
        
        time_step = pd.Timedelta(hours=self.config.get('time_step_hours', 1.0))
        
        # Initialize tracking
        trajectory_records = []
        current_time = start_time
        
        logger.info(f"Simulation period: {start_time} to {end_time}")
        logger.info(f"Particles: {len(self.particles)}")
        
        # Main simulation loop
        simulation_step = 0
        while current_time <= end_time:
            active_particles = 0
            
            # Move all active particles
            for particle in self.particles:
                if particle.active:
                    is_active = self.move_particle(particle, current_time)
                    if is_active:
                        active_particles += 1
                    
                    # Record particle state
                    state = particle.get_state()
                    state['time'] = current_time
                    trajectory_records.append(state)
            
            # Progress reporting
            if simulation_step % 24 == 0:  # Report every 24 time steps
                elapsed_hours = (current_time - start_time).total_seconds() / 3600.0
                logger.info(f"  Time: {current_time}, "
                           f"Elapsed: {elapsed_hours:.1f}h, "
                           f"Active: {active_particles}")
            
            # Break if no active particles
            if active_particles == 0:
                logger.info("No active particles remaining")
                break
            
            current_time += time_step
            simulation_step += 1
        
        # Create results DataFrame
        self.trajectory_data = pd.DataFrame(trajectory_records)
        
        logger.info(f"Simulation complete. Recorded {len(self.trajectory_data)} trajectory points")
        
        return self.trajectory_data
    
    def calculate_concentration_field(self, grid_resolution: float = 0.1) -> xr.Dataset:
        """
        Calculate oil concentration field
        
        Parameters
        ----------
        grid_resolution : float
            Grid resolution in degrees
            
        Returns
        -------
        xr.Dataset
            Oil concentration field
        """
        if self.trajectory_data is None or self.trajectory_data.empty:
            raise ValueError("No simulation data available. Run simulation first.")
        
        # Define grid
        lat_min = self.config.get('region_lat_min', 18.0)
        lat_max = self.config.get('region_lat_max', 30.0)
        lon_min = self.config.get('region_lon_min', -98.0)
        lon_max = self.config.get('region_lon_max', -88.0)
        
        lat_bins = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lon_bins = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        
        # Get unique times
        times = sorted(self.trajectory_data['time'].unique())
        
        # Initialize concentration array
        concentration = np.zeros((len(times), len(lat_bins)-1, len(lon_bins)-1))
        
        # Calculate concentration for each time step
        for t_idx, time in enumerate(times):
            time_data = self.trajectory_data[self.trajectory_data['time'] == time]
            
            if len(time_data) > 0:
                # Create 2D histogram of particle mass
                hist, _, _ = np.histogram2d(
                    time_data['latitude'], time_data['longitude'],
                    bins=[lat_bins, lon_bins],
                    weights=time_data['mass_kg'],
                    density=False
                )
                
                concentration[t_idx] = hist
        
        # Create dataset
        concentration_ds = xr.Dataset(
            {
                'oil_concentration': (
                    ['time', 'lat', 'lon'],
                    concentration,
                    {
                        'long_name': 'Oil concentration',
                        'units': 'kg',
                        'description': 'Oil mass per grid cell'
                    }
                )
            },
            coords={
                'time': times,
                'lat': (lat_bins[:-1] + lat_bins[1:]) / 2,
                'lon': (lon_bins[:-1] + lon_bins[1:]) / 2
            }
        )
        
        return concentration_ds
    
    def calculate_impact_statistics(self) -> Dict:
        """
        Calculate impact statistics
        
        Returns
        -------
        dict
            Impact statistics
        """
        if self.trajectory_data is None:
            return {}
        
        stats = {}
        
        # Mass balance
        total_mass = sum(p.mass_kg + p.evaporated_mass + p.dispersed_mass 
                        for p in self.particles)
        remaining_mass = sum(p.mass_kg for p in self.particles if p.active)
        evaporated_mass = sum(p.evaporated_mass for p in self.particles)
        dispersed_mass = sum(p.dispersed_mass for p in self.particles)
        beached_mass = sum(p.mass_kg for p in self.particles if p.beached)
        
        stats['mass_balance'] = {
            'total_released_kg': total_mass,
            'remaining_kg': remaining_mass,
            'evaporated_kg': evaporated_mass,
            'dispersed_kg': dispersed_mass,
            'beached_kg': beached_mass,
            'fraction_remaining': remaining_mass / total_mass if total_mass > 0 else 0,
            'fraction_evaporated': evaporated_mass / total_mass if total_mass > 0 else 0,
            'fraction_dispersed': dispersed_mass / total_mass if total_mass > 0 else 0,
            'fraction_beached': beached_mass / total_mass if total_mass > 0 else 0
        }
        
        # Particle statistics
        stats['particles'] = {
            'total': len(self.particles),
            'active': sum(1 for p in self.particles if p.active),
            'beached': sum(1 for p in self.particles if p.beached),
            'evaporated': sum(1 for p in self.particles if p.mass_kg < 0.1 and not p.beached)
        }
        
        # Spatial statistics
        if not self.trajectory_data.empty:
            final_positions = self.trajectory_data[
                self.trajectory_data['time'] == self.trajectory_data['time'].max()
            ]
            
            if len(final_positions) > 0:
                stats['spatial'] = {
                    'mean_latitude': float(final_positions['latitude'].mean()),
                    'mean_longitude': float(final_positions['longitude'].mean()),
                    'spread_km': self._calculate_spread_km(final_positions)
                }
        
        return stats
    
    def _calculate_spread_km(self, positions: pd.DataFrame) -> float:
        """Calculate spatial spread in kilometers"""
        if len(positions) < 2:
            return 0.0
        
        # Calculate convex hull area
        points = positions[['latitude', 'longitude']].values
        
        try:
            hull = spatial.ConvexHull(points)
            area_deg2 = hull.volume
            
            # Convert to km² (approximate)
            mean_lat = points[:, 0].mean()
            area_km2 = area_deg2 * 111.0**2 * np.cos(np.deg2rad(mean_lat))
            
            return float(np.sqrt(area_km2))  # Return approximate radius in km
        except:
            return 0.0
    
    def save_results(self, output_dir: Union[str, Path]) -> None:
        """Save simulation results"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory data
        if self.trajectory_data is not None:
            csv_file = output_dir / "trajectory_data.csv"
            self.trajectory_data.to_csv(csv_file, index=False)
            logger.info(f"Trajectory data saved to {csv_file}")
        
        # Save particle states
        particle_states = [p.get_state() for p in self.particles]
        import json
        states_file = output_dir / "particle_states.json"
        with open(states_file, 'w') as f:
            json.dump(particle_states, f, indent=2, default=str)
        
        # Save configuration
        config_file = output_dir / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save impact statistics
        stats = self.calculate_impact_statistics()
        stats_file = output_dir / "impact_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"All results saved to {output_dir}")
