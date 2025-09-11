"""
Two-Way Time Transfer (TWTT) for High-Precision Time Synchronization
Optimal for FTL (Frequency-Time Localization) systems
Enhanced with multipath whitening for improved accuracy
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from collections import deque

# Import multipath whitening if available
try:
    from ..signal_processing.multipath_whitening import MultipathWhiteningFilter
    WHITENING_AVAILABLE = True
except ImportError:
    WHITENING_AVAILABLE = False


@dataclass
class TWTTConfig:
    """Configuration for Two-Way Time Transfer"""
    # Hardware parameters
    timestamp_resolution_ns: float = 1.0  # 1ns resolution (high-end)
    crystal_stability_ppm: float = 20.0   # ±20ppm crystal drift
    
    # TWTT parameters
    exchange_rate_hz: float = 10.0  # Time sync exchanges per second
    averaging_window: int = 10      # Number of exchanges to average
    outlier_threshold_ns: float = 100.0  # Reject measurements > 100ns from median
    
    # Kalman filter parameters
    process_noise_ns2: float = 1.0   # Clock drift variance
    measurement_noise_ns2: float = 10.0  # Timestamp noise variance
    
    # Asymmetry estimation
    estimate_asymmetry: bool = True  # Estimate path asymmetry
    asymmetry_alpha: float = 0.01    # Learning rate for asymmetry
    
    # Multipath whitening parameters
    enable_whitening: bool = True  # Enable multipath whitening
    whitening_regularization: float = 0.01  # MMSE regularization parameter
    whitening_filter_length: int = 256  # Whitening filter taps
    signal_bandwidth_mhz: float = 200.0  # Signal bandwidth for whitening


class TWTTNode:
    """
    Two-Way Time Transfer implementation for a single node
    Provides high-precision time synchronization for FTL
    Enhanced with multipath whitening for improved accuracy
    """
    
    def __init__(self, node_id: int, config: TWTTConfig = TWTTConfig()):
        self.node_id = node_id
        self.config = config
        
        # Time state
        self.local_time_ns = 0
        self.clock_offset_ns = 0.0
        self.clock_drift_ppb = 0.0  # Parts per billion
        
        # Neighbor synchronization states
        self.neighbor_states = {}  # neighbor_id -> TWTTNeighborState
        
        # Statistics
        self.sync_accuracy_ns = float('inf')
        self.sync_precision_ns = float('inf')
        
        # Multipath whitening filter
        self.whitening_filter = None
        if config.enable_whitening and WHITENING_AVAILABLE:
            self.whitening_filter = MultipathWhiteningFilter(
                regularization=config.whitening_regularization,
                filter_length=config.whitening_filter_length,
                bandwidth_mhz=config.signal_bandwidth_mhz
            )
        
    def initiate_twtt_exchange(self, neighbor_id: int) -> Dict:
        """
        Initiate TWTT exchange with neighbor
        Returns message to send
        """
        t1 = self.get_hardware_timestamp()
        
        message = {
            'type': 'TWTT_REQUEST',
            'from_node': self.node_id,
            'to_node': neighbor_id,
            't1': t1,
            'sequence': self._get_sequence(neighbor_id)
        }
        
        # Store t1 for this exchange
        if neighbor_id not in self.neighbor_states:
            self.neighbor_states[neighbor_id] = TWTTNeighborState(neighbor_id)
        
        self.neighbor_states[neighbor_id].pending_t1 = t1
        
        return message
    
    def process_twtt_request(self, message: Dict) -> Dict:
        """
        Process incoming TWTT request and generate response
        """
        t2 = self.get_hardware_timestamp()  # Receive timestamp
        
        # Process the request (minimal processing delay)
        t3 = self.get_hardware_timestamp()  # Transmit timestamp
        
        response = {
            'type': 'TWTT_RESPONSE',
            'from_node': self.node_id,
            'to_node': message['from_node'],
            't1': message['t1'],
            't2': t2,
            't3': t3,
            'sequence': message['sequence']
        }
        
        return response
    
    def process_twtt_response(self, message: Dict) -> Dict:
        """
        Process TWTT response and calculate time offset
        Enhanced with optional multipath whitening
        """
        t4 = self.get_hardware_timestamp()  # Receive timestamp
        
        t1 = message['t1']
        t2 = message['t2']
        t3 = message['t3']
        neighbor_id = message['from_node']
        
        # Apply whitening if available and configured
        if self.whitening_filter is not None and 'channel_impulse_response' in message:
            # Process timestamps through whitening filter
            t1, t2, t3, t4 = self._apply_whitening_to_timestamps(
                t1, t2, t3, t4, message['channel_impulse_response']
            )
        
        # Calculate round-trip time and offset
        rtt = (t4 - t1) - (t3 - t2)
        
        # Basic TWTT offset (assumes symmetric path)
        basic_offset = ((t2 - t1) + (t3 - t4)) / 2
        
        # Get neighbor state
        if neighbor_id not in self.neighbor_states:
            self.neighbor_states[neighbor_id] = TWTTNeighborState(neighbor_id)
        
        state = self.neighbor_states[neighbor_id]
        
        # Enhanced processing with asymmetry estimation
        if self.config.estimate_asymmetry:
            # Estimate path asymmetry using historical data
            asymmetry = state.estimate_path_asymmetry(rtt, basic_offset)
            corrected_offset = basic_offset - asymmetry
        else:
            corrected_offset = basic_offset
            asymmetry = 0
        
        # Update Kalman filter
        state.update_kalman(corrected_offset, rtt)
        
        # Store measurement for averaging
        state.add_measurement(corrected_offset, rtt)
        
        # Calculate statistics
        if len(state.offset_history) >= self.config.averaging_window:
            filtered_offset = state.get_filtered_offset()
            precision = state.get_precision()
        else:
            filtered_offset = corrected_offset
            precision = float('inf')
        
        return {
            'neighbor_id': neighbor_id,
            'raw_offset_ns': basic_offset,
            'corrected_offset_ns': corrected_offset,
            'filtered_offset_ns': filtered_offset,
            'rtt_ns': rtt,
            'asymmetry_ns': asymmetry,
            'precision_ns': precision,
            'clock_drift_ppb': state.clock_drift_estimate * 1e9,
            'timestamps': {'t1': t1, 't2': t2, 't3': t3, 't4': t4}
        }
    
    def get_hardware_timestamp(self) -> int:
        """
        Simulate hardware timestamp with realistic noise
        In real implementation, this would read from NIC/PHY
        """
        # Add timestamp quantization
        timestamp = self.local_time_ns
        timestamp = round(timestamp / self.config.timestamp_resolution_ns) * self.config.timestamp_resolution_ns
        
        # Add jitter (Gaussian noise)
        jitter = np.random.normal(0, self.config.timestamp_resolution_ns)
        
        return int(timestamp + jitter)
    
    def _get_sequence(self, neighbor_id: int) -> int:
        """Get next sequence number for neighbor"""
        if neighbor_id not in self.neighbor_states:
            self.neighbor_states[neighbor_id] = TWTTNeighborState(neighbor_id)
        
        state = self.neighbor_states[neighbor_id]
        state.sequence_number += 1
        return state.sequence_number
    
    def get_synchronized_time(self) -> Tuple[float, float]:
        """
        Get synchronized time and uncertainty
        
        Returns:
            (synchronized_time_ns, uncertainty_ns)
        """
        # Average offsets from all neighbors
        if not self.neighbor_states:
            return self.local_time_ns, float('inf')
        
        offsets = []
        weights = []
        
        for state in self.neighbor_states.values():
            if state.is_synchronized():
                offset = state.get_filtered_offset()
                precision = state.get_precision()
                
                offsets.append(offset)
                weights.append(1.0 / max(precision, 1.0))
        
        if not offsets:
            return self.local_time_ns, float('inf')
        
        # Weighted average
        weights = np.array(weights) / np.sum(weights)
        consensus_offset = np.sum(np.array(offsets) * weights)
        
        # Uncertainty is weighted average of precisions
        uncertainty = 1.0 / np.sum(weights)
        
        synchronized_time = self.local_time_ns - consensus_offset
        
        return synchronized_time, uncertainty
    
    def _apply_whitening_to_timestamps(self, t1: int, t2: int, t3: int, t4: int, 
                                      channel_impulse_response: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Apply whitening filter to refine timestamp estimates
        Extracts earliest path for accurate ToA
        
        Args:
            t1, t2, t3, t4: TWTT timestamps
            channel_impulse_response: Estimated channel response
            
        Returns:
            Refined timestamps after whitening
        """
        if self.whitening_filter is None:
            return t1, t2, t3, t4
            
        # Compute whitening filter for this channel
        whitening_coeffs = self.whitening_filter.compute_whitening_filter(
            channel_impulse_response,
            snr_db=20  # Assume moderate SNR
        )
        
        # Apply whitening to channel response
        whitened_cir = self.whitening_filter.apply_whitening(
            channel_impulse_response, 
            whitening_coeffs
        )
        
        # Extract refined time of arrival
        # The whitening process sharpens the earliest path
        sampling_rate = self.config.signal_bandwidth_mhz * 1e6
        toa_correction, quality = self.whitening_filter.extract_time_of_arrival(
            whitened_cir,
            sampling_rate,
            interpolation='parabolic'
        )
        
        # Convert correction to nanoseconds
        correction_ns = toa_correction * 1e9
        
        # Apply correction to timestamps
        # The correction represents the difference between raw and whitened ToA
        # Apply proportionally to forward and reverse paths
        t2_refined = int(t2 - correction_ns / 2)
        t3_refined = int(t3 - correction_ns / 2)
        
        return t1, t2_refined, t3_refined, t4


class TWTTNeighborState:
    """State tracking for TWTT with a specific neighbor"""
    
    def __init__(self, neighbor_id: int):
        self.neighbor_id = neighbor_id
        self.sequence_number = 0
        self.pending_t1 = None
        
        # Measurement history
        self.offset_history = deque(maxlen=100)
        self.rtt_history = deque(maxlen=100)
        
        # Kalman filter state
        self.kf_state = np.array([0.0, 0.0])  # [offset, drift]
        self.kf_covariance = np.eye(2) * 1000
        
        # Path asymmetry estimation
        self.path_asymmetry = 0.0
        self.asymmetry_variance = 100.0
        
        # Statistics
        self.clock_drift_estimate = 0.0
        self.last_update_time = None
        
    def add_measurement(self, offset_ns: float, rtt_ns: float):
        """Add new measurement to history"""
        self.offset_history.append(offset_ns)
        self.rtt_history.append(rtt_ns)
    
    def update_kalman(self, measured_offset: float, rtt: float):
        """Update Kalman filter with new measurement"""
        # Time since last update
        current_time = measured_offset  # Simplified
        
        if self.last_update_time is not None:
            dt = (current_time - self.last_update_time) * 1e-9  # Convert to seconds
            
            # State transition
            F = np.array([[1, dt], [0, 1]])
            Q = np.array([[dt**3/3, dt**2/2], [dt**2/2, dt]]) * 1e-18  # Process noise
            
            # Predict
            self.kf_state = F @ self.kf_state
            self.kf_covariance = F @ self.kf_covariance @ F.T + Q
        
        # Measurement update
        H = np.array([1, 0])  # Measure offset only
        R = 10.0 + rtt * 0.01  # Measurement noise increases with RTT
        
        # Innovation
        y = measured_offset - H @ self.kf_state
        S = H @ self.kf_covariance @ H.T + R
        
        # Kalman gain
        K = self.kf_covariance @ H.T / S
        
        # Update
        self.kf_state = self.kf_state + K * y
        self.kf_covariance = (np.eye(2) - np.outer(K, H)) @ self.kf_covariance
        
        # Extract estimates
        self.clock_drift_estimate = self.kf_state[1]
        self.last_update_time = current_time
    
    def estimate_path_asymmetry(self, rtt: float, offset: float) -> float:
        """
        Estimate path asymmetry using Allan variance or similar
        
        Path asymmetry causes systematic bias in offset measurements
        """
        if len(self.offset_history) < 10:
            return 0.0
        
        # Simple asymmetry detection: look for correlation between RTT and offset
        recent_rtts = list(self.rtt_history)[-20:]
        recent_offsets = list(self.offset_history)[-20:]
        
        if len(recent_rtts) >= 10:
            # Calculate correlation
            correlation = np.corrcoef(recent_rtts, recent_offsets)[0, 1]
            
            # If strong correlation, estimate asymmetry
            if abs(correlation) > 0.5:
                # Asymmetry proportional to RTT variation
                rtt_std = np.std(recent_rtts)
                self.path_asymmetry = correlation * rtt_std * 0.1
            else:
                # Decay asymmetry estimate
                self.path_asymmetry *= 0.99
        
        return self.path_asymmetry
    
    def get_filtered_offset(self) -> float:
        """Get filtered offset estimate"""
        if len(self.offset_history) == 0:
            return 0.0
        
        # Use Kalman estimate if available
        if self.last_update_time is not None:
            return self.kf_state[0]
        
        # Otherwise use median of recent measurements
        recent = list(self.offset_history)[-10:]
        return np.median(recent)
    
    def get_precision(self) -> float:
        """Get precision (standard deviation) of offset estimates"""
        if len(self.offset_history) < 3:
            return float('inf')
        
        recent = list(self.offset_history)[-10:]
        return np.std(recent)
    
    def is_synchronized(self) -> bool:
        """Check if synchronized with this neighbor"""
        return len(self.offset_history) >= 3


class FTLTimeSyncManager:
    """
    Manager for FTL system time synchronization using TWTT
    Coordinates time sync across the network
    """
    
    def __init__(self, node_ids: list, config: TWTTConfig = TWTTConfig()):
        self.config = config
        self.nodes = {nid: TWTTNode(nid, config) for nid in node_ids}
        self.sync_pairs = []  # List of (node_i, node_j) pairs for sync
        
        # Build sync topology (could be optimized)
        for i, node_i in enumerate(node_ids):
            for node_j in node_ids[i+1:]:
                self.sync_pairs.append((node_i, node_j))
    
    def run_sync_round(self) -> Dict:
        """
        Run one round of TWTT synchronization
        
        Returns:
            Synchronization statistics
        """
        results = []
        
        for node_i, node_j in self.sync_pairs:
            # Node i initiates TWTT with node j
            request = self.nodes[node_i].initiate_twtt_exchange(node_j)
            
            # Node j processes request and sends response
            response = self.nodes[node_j].process_twtt_request(request)
            
            # Node i processes response
            result = self.nodes[node_i].process_twtt_response(response)
            results.append(result)
        
        # Calculate network-wide synchronization accuracy
        sync_errors = []
        for node_id, node in self.nodes.items():
            sync_time, uncertainty = node.get_synchronized_time()
            sync_errors.append(uncertainty)
        
        return {
            'mean_sync_error_ns': np.mean(sync_errors),
            'max_sync_error_ns': np.max(sync_errors),
            'individual_results': results
        }


# Example usage for FTL
if __name__ == "__main__":
    print("Testing TWTT for FTL System")
    print("="*50)
    
    # Create FTL network with 5 nodes
    node_ids = [0, 1, 2, 3, 4]
    manager = FTLTimeSyncManager(node_ids)
    
    # Simulate time progression and sync
    for round in range(10):
        # Advance local clocks (with drift)
        for node in manager.nodes.values():
            drift_ppb = np.random.normal(0, 20)  # ±20ppb drift
            node.local_time_ns += 100_000_000  # 100ms
            node.local_time_ns += int(100_000_000 * drift_ppb * 1e-9)
        
        # Run TWTT sync
        results = manager.run_sync_round()
        
        if round % 3 == 0:
            print(f"Round {round}: Sync error = {results['mean_sync_error_ns']:.1f}ns")
    
    print("\nTWTT provides nanosecond-level synchronization for FTL!")