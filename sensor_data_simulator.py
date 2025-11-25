"""
Predictive Maintenance - Sensor Data Simulator
================================================
This module generates synthetic time-series sensor data for predictive maintenance simulation.
It creates realistic sensor readings with normal operation patterns, gradual degradation,
and various failure modes.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import json


@dataclass
class SensorConfig:
    """Configuration for a single sensor."""
    name: str
    baseline: float
    amplitude: float
    frequency: float
    noise_level: float
    unit: str


@dataclass
class MachineConfig:
    """Configuration for a machine with multiple sensors."""
    machine_id: str
    machine_type: str
    location: str
    sensors: List[SensorConfig]
    degradation_start: float  # as fraction of total time (0.0 to 1.0)
    failure_point: float  # as fraction of total time (0.0 to 1.0)


class SensorDataGenerator:
    """
    Generates synthetic sensor data simulating machine operation,
    degradation, and eventual failure.
    """
    
    def __init__(self, machine_config: MachineConfig, sampling_rate_hz: float = 1.0):
        """
        Initialize the sensor data generator.
        
        Args:
            machine_config: Configuration for the machine
            sampling_rate_hz: Sampling rate in Hz (samples per second)
        """
        self.config = machine_config
        self.sampling_rate = sampling_rate_hz
        self.dt = 1.0 / sampling_rate_hz
        
    def generate_normal_operation(
        self,
        sensor: SensorConfig,
        duration_hours: float
    ) -> np.ndarray:
        """
        Generate sensor data for normal operation.
        
        Args:
            sensor: Sensor configuration
            duration_hours: Duration in hours
            
        Returns:
            Array of sensor readings
        """
        num_samples = int(duration_hours * 3600 * self.sampling_rate)
        t = np.linspace(0, duration_hours * 3600, num_samples)
        
        # Base cyclic signal (normal operation pattern)
        base_signal = sensor.baseline + sensor.amplitude * np.sin(
            2 * np.pi * sensor.frequency * t
        )
        
        # Add realistic noise
        noise = np.random.normal(0, sensor.noise_level, num_samples)
        
        return base_signal + noise
    
    def add_degradation(
        self,
        signal: np.ndarray,
        start_fraction: float,
        end_fraction: float,
        sensor: SensorConfig
    ) -> np.ndarray:
        """
        Add degradation patterns to the signal.
        
        Degradation characteristics:
        - Gradual drift (wear)
        - Increasing variance (loosening components)
        - Harmonic distortion (imbalance)
        
        Args:
            signal: Original signal
            start_fraction: When degradation starts (0.0 to 1.0)
            end_fraction: When failure occurs (0.0 to 1.0)
            sensor: Sensor configuration
            
        Returns:
            Signal with degradation
        """
        n = len(signal)
        degraded = signal.copy()
        
        # Calculate degradation region
        start_idx = int(n * start_fraction)
        end_idx = int(n * end_fraction)
        
        if start_idx >= end_idx:
            return degraded
        
        degradation_length = end_idx - start_idx
        
        # 1. Gradual drift (e.g., bearing wear causing baseline shift)
        drift_magnitude = sensor.baseline * 0.3  # 30% drift
        drift = np.linspace(0, drift_magnitude, degradation_length)
        degraded[start_idx:end_idx] += drift
        
        # 2. Increasing variance (loosening components)
        for i in range(start_idx, end_idx):
            progress = (i - start_idx) / degradation_length
            variance_multiplier = 1 + 3 * progress  # Up to 4x noise
            additional_noise = np.random.normal(
                0, sensor.noise_level * variance_multiplier
            )
            degraded[i] += additional_noise
        
        # 3. Harmonic distortion (imbalance, misalignment)
        t_degradation = np.arange(degradation_length) * self.dt
        # Add second and third harmonics
        harmonic_2 = 0.1 * sensor.amplitude * np.sin(
            2 * np.pi * 2 * sensor.frequency * t_degradation
        )
        harmonic_3 = 0.05 * sensor.amplitude * np.sin(
            2 * np.pi * 3 * sensor.frequency * t_degradation
        )
        degraded[start_idx:end_idx] += harmonic_2 + harmonic_3
        
        return degraded
    
    def add_failure_mode(
        self,
        signal: np.ndarray,
        failure_fraction: float,
        failure_type: str
    ) -> np.ndarray:
        """
        Add failure modes to the signal.
        
        Args:
            signal: Signal with degradation
            failure_fraction: When failure occurs (0.0 to 1.0)
            failure_type: Type of failure ('spike', 'dropout', 'oscillation')
            
        Returns:
            Signal with failure mode
        """
        n = len(signal)
        failure_idx = int(n * failure_fraction)
        failed = signal.copy()
        
        if failure_type == 'spike':
            # Sudden spike (e.g., bearing seizure)
            spike_magnitude = np.abs(signal[failure_idx]) * 5
            spike_duration = int(0.01 * n)  # 1% of signal
            failed[failure_idx:failure_idx + spike_duration] += spike_magnitude
            
        elif failure_type == 'dropout':
            # Signal dropout (sensor or component failure)
            dropout_duration = int(0.05 * n)  # 5% of signal
            failed[failure_idx:failure_idx + dropout_duration] = 0
            
        elif failure_type == 'oscillation':
            # Violent oscillation (catastrophic failure)
            oscillation_duration = n - failure_idx
            t_osc = np.arange(oscillation_duration) * self.dt
            oscillation = 3 * np.abs(signal[failure_idx]) * np.sin(
                2 * np.pi * 50 * t_osc  # High frequency oscillation
            )
            failed[failure_idx:] += oscillation
            
        return failed
    
    def generate_dataset(
        self,
        duration_hours: float,
        failure_type: str = 'spike'
    ) -> pd.DataFrame:
        """
        Generate complete dataset for a machine with all sensors.
        
        Args:
            duration_hours: Total duration in hours
            failure_type: Type of failure to simulate
            
        Returns:
            DataFrame with timestamp and all sensor readings
        """
        num_samples = int(duration_hours * 3600 * self.sampling_rate)
        
        # Generate timestamps
        start_time = datetime.now()
        timestamps = [
            start_time + timedelta(seconds=i * self.dt)
            for i in range(num_samples)
        ]
        
        # Initialize DataFrame
        data = {'timestamp': timestamps}
        
        # Generate data for each sensor
        for sensor in self.config.sensors:
            # Normal operation
            signal = self.generate_normal_operation(sensor, duration_hours)
            
            # Add degradation
            signal = self.add_degradation(
                signal,
                self.config.degradation_start,
                self.config.failure_point,
                sensor
            )
            
            # Add failure mode
            signal = self.add_failure_mode(
                signal,
                self.config.failure_point,
                failure_type
            )
            
            data[sensor.name] = signal
        
        df = pd.DataFrame(data)
        
        # Add metadata columns
        df['machine_id'] = self.config.machine_id
        df['machine_type'] = self.config.machine_type
        df['location'] = self.config.location
        
        # Add labels
        df['is_degraded'] = (
            df.index >= int(num_samples * self.config.degradation_start)
        ).astype(int)
        df['is_failed'] = (
            df.index >= int(num_samples * self.config.failure_point)
        ).astype(int)
        
        # Calculate remaining useful life (RUL) in hours
        failure_time = duration_hours * self.config.failure_point
        df['rul_hours'] = failure_time - (df.index * self.dt / 3600)
        df['rul_hours'] = df['rul_hours'].clip(lower=0)
        
        return df
    
    def add_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add engineered features to the dataset.
        
        Args:
            df: DataFrame with sensor readings
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Rolling statistics (1-hour window)
        window_size = int(3600 * self.sampling_rate)
        
        for sensor in self.config.sensors:
            sensor_name = sensor.name
            
            # Rolling mean and std
            df[f'{sensor_name}_rolling_mean_1h'] = (
                df[sensor_name].rolling(window=window_size, min_periods=1).mean()
            )
            df[f'{sensor_name}_rolling_std_1h'] = (
                df[sensor_name].rolling(window=window_size, min_periods=1).std()
            )
            
            # Rate of change
            df[f'{sensor_name}_rate_of_change'] = df[sensor_name].diff()
            
            # Deviation from baseline
            df[f'{sensor_name}_deviation'] = np.abs(
                df[sensor_name] - sensor.baseline
            )
        
        return df


class MultiMachineSimulator:
    """Simulate multiple machines with different degradation patterns."""
    
    def __init__(self, num_machines: int = 10):
        """
        Initialize simulator for multiple machines.
        
        Args:
            num_machines: Number of machines to simulate
        """
        self.num_machines = num_machines
        self.machines = self._create_machine_configs()
    
    def _create_machine_configs(self) -> List[MachineConfig]:
        """Create configurations for multiple machines with varying parameters."""
        configs = []
        
        machine_types = ['cnc_mill', 'lathe', 'press', 'grinder']
        locations = ['floor_1', 'floor_2', 'floor_3']
        failure_types = ['spike', 'dropout', 'oscillation']
        
        for i in range(self.num_machines):
            # Randomize parameters for variety
            degradation_start = np.random.uniform(0.5, 0.7)
            failure_point = np.random.uniform(0.85, 0.95)
            
            sensors = [
                SensorConfig(
                    name='vibration_x',
                    baseline=0.2,
                    amplitude=0.05,
                    frequency=np.random.uniform(0.01, 0.03),
                    noise_level=0.01,
                    unit='g'
                ),
                SensorConfig(
                    name='vibration_y',
                    baseline=0.18,
                    amplitude=0.04,
                    frequency=np.random.uniform(0.01, 0.03),
                    noise_level=0.01,
                    unit='g'
                ),
                SensorConfig(
                    name='vibration_z',
                    baseline=0.22,
                    amplitude=0.06,
                    frequency=np.random.uniform(0.01, 0.03),
                    noise_level=0.012,
                    unit='g'
                ),
                SensorConfig(
                    name='temperature',
                    baseline=np.random.uniform(60, 75),
                    amplitude=5,
                    frequency=np.random.uniform(0.001, 0.005),
                    noise_level=1.5,
                    unit='celsius'
                ),
                SensorConfig(
                    name='pressure',
                    baseline=100,
                    amplitude=10,
                    frequency=np.random.uniform(0.002, 0.008),
                    noise_level=2,
                    unit='bar'
                ),
                SensorConfig(
                    name='rpm',
                    baseline=1500,
                    amplitude=50,
                    frequency=np.random.uniform(0.001, 0.003),
                    noise_level=10,
                    unit='rpm'
                )
            ]
            
            config = MachineConfig(
                machine_id=f'M-{i+1:03d}',
                machine_type=np.random.choice(machine_types),
                location=np.random.choice(locations),
                sensors=sensors,
                degradation_start=degradation_start,
                failure_point=failure_point
            )
            
            configs.append(config)
        
        return configs
    
    def generate_all_machines(
        self,
        duration_hours: float = 720,  # 30 days
        sampling_rate_hz: float = 0.1  # 10-second intervals
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate data for all machines.
        
        Args:
            duration_hours: Duration in hours
            sampling_rate_hz: Sampling rate
            
        Returns:
            Dictionary mapping machine_id to DataFrame
        """
        datasets = {}
        
        failure_types = ['spike', 'dropout', 'oscillation']
        
        for i, config in enumerate(self.machines):
            generator = SensorDataGenerator(config, sampling_rate_hz)
            
            # Rotate through failure types
            failure_type = failure_types[i % len(failure_types)]
            
            df = generator.generate_dataset(duration_hours, failure_type)
            df = generator.add_feature_engineering(df)
            
            datasets[config.machine_id] = df
            
            print(f"Generated data for {config.machine_id}: "
                  f"{len(df)} samples, "
                  f"failure at {config.failure_point*100:.1f}%")
        
        return datasets
    
    def export_to_kafka_format(self, datasets: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Convert datasets to Kafka message format.
        
        Args:
            datasets: Dictionary of DataFrames
            
        Returns:
            List of messages in Kafka format
        """
        messages = []
        
        for machine_id, df in datasets.items():
            for _, row in df.iterrows():
                message = {
                    'machine_id': machine_id,
                    'timestamp': row['timestamp'].isoformat(),
                    'sensors': {
                        'vibration_x': float(row['vibration_x']),
                        'vibration_y': float(row['vibration_y']),
                        'vibration_z': float(row['vibration_z']),
                        'temperature': float(row['temperature']),
                        'pressure': float(row['pressure']),
                        'rpm': float(row['rpm'])
                    },
                    'metadata': {
                        'machine_type': row['machine_type'],
                        'location': row['location']
                    },
                    'labels': {
                        'is_degraded': int(row['is_degraded']),
                        'is_failed': int(row['is_failed']),
                        'rul_hours': float(row['rul_hours'])
                    }
                }
                messages.append(message)
        
        return messages


def example_usage():
    """Example usage of the sensor data generator."""
    
    print("="*70)
    print("Predictive Maintenance - Sensor Data Simulation")
    print("="*70)
    
    # Example 1: Single machine
    print("\n1. Generating data for a single machine...")
    
    sensors = [
        SensorConfig('vibration_x', 0.2, 0.05, 0.02, 0.01, 'g'),
        SensorConfig('temperature', 70, 5, 0.003, 1.5, 'celsius'),
        SensorConfig('pressure', 100, 10, 0.005, 2, 'bar')
    ]
    
    machine = MachineConfig(
        machine_id='M-001',
        machine_type='cnc_mill',
        location='factory_floor_1',
        sensors=sensors,
        degradation_start=0.6,  # Degradation starts at 60% of timeline
        failure_point=0.9  # Failure at 90% of timeline
    )
    
    generator = SensorDataGenerator(machine, sampling_rate_hz=0.1)
    df = generator.generate_dataset(duration_hours=168, failure_type='spike')  # 1 week
    df = generator.add_feature_engineering(df)
    
    print(f"\nGenerated {len(df)} samples")
    print(f"\nFirst few rows:")
    print(df[['timestamp', 'vibration_x', 'temperature', 'pressure', 
              'is_degraded', 'is_failed', 'rul_hours']].head(10))
    
    print(f"\nData statistics:")
    print(df[['vibration_x', 'temperature', 'pressure']].describe())
    
    # Example 2: Multiple machines
    print("\n\n2. Generating data for multiple machines...")
    
    simulator = MultiMachineSimulator(num_machines=5)
    datasets = simulator.generate_all_machines(
        duration_hours=168,  # 1 week
        sampling_rate_hz=0.1  # Every 10 seconds
    )
    
    print(f"\nGenerated data for {len(datasets)} machines")
    print(f"Total samples: {sum(len(df) for df in datasets.values())}")
    
    # Example 3: Export to Kafka format
    print("\n\n3. Converting to Kafka message format...")
    
    messages = simulator.export_to_kafka_format(datasets)
    print(f"Generated {len(messages)} Kafka messages")
    print(f"\nSample message:")
    print(json.dumps(messages[0], indent=2))
    
    # Save sample data
    print("\n\n4. Saving sample data...")
    sample_machine_id = list(datasets.keys())[0]
    sample_df = datasets[sample_machine_id]
    sample_df.to_csv('sample_sensor_data.csv', index=False)
    print(f"Saved sample data to 'sample_sensor_data.csv'")
    
    print("\n" + "="*70)
    print("Simulation complete!")
    print("="*70)


if __name__ == '__main__':
    example_usage()
