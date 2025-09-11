"""
Multipath Whitening Filter for RF Ranging
Implements MMSE whitening to suppress multipath and sharpen correlation peaks
Based on S-band approach for 2-5cm ranging accuracy
"""

import numpy as np
from typing import Tuple, Optional
from scipy import signal
from scipy.linalg import toeplitz


class MultipathWhiteningFilter:
    """
    MMSE whitening filter to suppress multipath in ranging measurements
    Transforms multipath-corrupted channel to near-ideal impulse response
    """
    
    def __init__(self, 
                 regularization: float = 0.01,
                 filter_length: int = 1024,
                 bandwidth_mhz: float = 200.0):
        """
        Initialize whitening filter
        
        Args:
            regularization: Lambda parameter for regularized inverse (0.001-0.1 typical)
            filter_length: Number of taps in whitening filter
            bandwidth_mhz: Signal bandwidth in MHz (affects resolution)
        """
        self.regularization = regularization
        self.filter_length = filter_length
        self.bandwidth = bandwidth_mhz * 1e6  # Convert to Hz
        
        # Speed of light for ranging
        self.c = 299792458.0  # m/s
        
        # Resolution in meters
        self.range_resolution = self.c / (2 * self.bandwidth)
        
        # Cache for filters per channel
        self.filter_cache = {}
        
    def estimate_channel_impulse_response(self, 
                                         received_signal: np.ndarray,
                                         transmitted_signal: np.ndarray,
                                         oversample_factor: int = 4) -> np.ndarray:
        """
        Estimate channel impulse response via matched filtering
        
        Args:
            received_signal: Received waveform
            transmitted_signal: Known transmitted waveform (e.g., CAZAC sequence)
            oversample_factor: Oversampling for better resolution
            
        Returns:
            Channel impulse response h[n]
        """
        # Matched filter (correlation)
        correlation = signal.correlate(received_signal, transmitted_signal, mode='same')
        
        # Normalize by autocorrelation peak
        auto_corr = signal.correlate(transmitted_signal, transmitted_signal, mode='same')
        peak_idx = np.argmax(np.abs(auto_corr))
        normalization = auto_corr[peak_idx]
        
        if np.abs(normalization) > 1e-10:
            cir = correlation / normalization
        else:
            cir = correlation
            
        # Optional: Oversample for finer resolution
        if oversample_factor > 1:
            cir = signal.resample(cir, len(cir) * oversample_factor)
            
        return cir
    
    def compute_whitening_filter(self, 
                                channel_impulse_response: np.ndarray,
                                snr_db: Optional[float] = None) -> np.ndarray:
        """
        Compute MMSE whitening filter using regularized inverse
        W(z) = argmin ||W * h - δ||² + λ||W||²
        
        Args:
            channel_impulse_response: CIR h[n] from correlation
            snr_db: SNR in dB for adaptive regularization
            
        Returns:
            Whitening filter coefficients w[n]
        """
        h = channel_impulse_response
        
        # Adaptive regularization based on SNR
        if snr_db is not None:
            # Lower regularization for high SNR (more aggressive whitening)
            # Higher regularization for low SNR (prevent noise amplification)
            snr_linear = 10 ** (snr_db / 10)
            adaptive_reg = self.regularization * (1 + 1/snr_linear)
        else:
            adaptive_reg = self.regularization
            
        # Build Toeplitz matrix for convolution
        # We want to find W such that W * h ≈ δ (impulse)
        n = min(self.filter_length, len(h))
        
        # Create convolution matrix
        h_padded = np.concatenate([h[:n], np.zeros(n-1)])
        H = toeplitz(h_padded[:n], np.concatenate([[h[0]], np.zeros(n-1)]))
        
        # MMSE solution: W = (H^H * H + λI)^(-1) * H^H * δ
        # Where δ is unit impulse at desired delay
        HTH = H.T.conj() @ H
        I = np.eye(n)
        
        # Target: unit impulse (earliest path)
        delta = np.zeros(n)
        delta[0] = 1.0  # Target earliest arrival
        
        # Regularized inverse
        try:
            W_matrix = np.linalg.solve(HTH + adaptive_reg * I, H.T.conj() @ delta)
        except np.linalg.LinAlgError:
            # Fallback to simple matched filter if inversion fails
            W_matrix = np.conj(h[:n][::-1])
            W_matrix = W_matrix / np.sum(np.abs(W_matrix)**2)
            
        return W_matrix
    
    def apply_whitening(self, 
                       correlation_output: np.ndarray,
                       whitening_filter: np.ndarray) -> np.ndarray:
        """
        Apply whitening filter to correlation output
        Sharpens peaks and suppresses multipath sidelobes
        
        Args:
            correlation_output: Raw correlation result
            whitening_filter: Whitening filter coefficients
            
        Returns:
            Whitened correlation with sharpened peaks
        """
        # Apply filter via convolution
        whitened = signal.convolve(correlation_output, whitening_filter, mode='same')
        
        return whitened
    
    def extract_time_of_arrival(self,
                               whitened_correlation: np.ndarray,
                               sampling_rate_hz: float,
                               interpolation: str = 'parabolic') -> Tuple[float, float]:
        """
        Extract precise time of arrival from whitened correlation
        
        Args:
            whitened_correlation: Whitened correlation output
            sampling_rate_hz: Sampling rate in Hz
            interpolation: 'parabolic', 'sinc', or 'none'
            
        Returns:
            (toa_seconds, quality_metric)
        """
        # Find peak (earliest significant peak after whitening)
        correlation_magnitude = np.abs(whitened_correlation)
        
        # Threshold to avoid noise peaks (10% of max)
        threshold = 0.1 * np.max(correlation_magnitude)
        
        # Find first peak above threshold
        above_threshold = correlation_magnitude > threshold
        if not np.any(above_threshold):
            # No valid peak found
            return 0.0, 0.0
            
        first_peak_idx = np.argmax(above_threshold)
        
        # Search for actual peak near this point
        search_window = 10
        start_idx = max(0, first_peak_idx - search_window)
        end_idx = min(len(correlation_magnitude), first_peak_idx + search_window + 1)
        
        local_peak_idx = start_idx + np.argmax(correlation_magnitude[start_idx:end_idx])
        
        # Sub-sample interpolation
        if interpolation == 'parabolic' and local_peak_idx > 0 and local_peak_idx < len(correlation_magnitude) - 1:
            # Parabolic interpolation using 3 points
            y1 = correlation_magnitude[local_peak_idx - 1]
            y2 = correlation_magnitude[local_peak_idx]
            y3 = correlation_magnitude[local_peak_idx + 1]
            
            # Parabolic peak offset
            delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3) if (y1 - 2*y2 + y3) != 0 else 0
            
            # Refined index
            refined_idx = local_peak_idx + delta
        else:
            refined_idx = local_peak_idx
            
        # Convert to time
        toa_seconds = refined_idx / sampling_rate_hz
        
        # Quality metric: ratio of peak to noise floor
        noise_floor = np.median(correlation_magnitude)
        peak_value = correlation_magnitude[local_peak_idx]
        quality = peak_value / (noise_floor + 1e-10)
        
        return toa_seconds, quality
    
    def process_ranging_measurement(self,
                                  received_signal: np.ndarray,
                                  transmitted_signal: np.ndarray,
                                  sampling_rate_hz: float,
                                  snr_db: Optional[float] = None) -> dict:
        """
        Complete processing pipeline for multipath-resilient ranging
        
        Args:
            received_signal: Received waveform
            transmitted_signal: Reference transmitted waveform
            sampling_rate_hz: Sampling rate
            snr_db: Estimated SNR for adaptive processing
            
        Returns:
            Dictionary with ranging results and diagnostics
        """
        # Step 1: Estimate channel impulse response
        cir = self.estimate_channel_impulse_response(
            received_signal, transmitted_signal, oversample_factor=1
        )
        
        # Step 2: Compute whitening filter
        whitening_filter = self.compute_whitening_filter(cir, snr_db)
        
        # Step 3: Apply whitening
        whitened = self.apply_whitening(cir, whitening_filter)
        
        # Step 4: Extract ToA
        toa, quality = self.extract_time_of_arrival(whitened, sampling_rate_hz)
        
        # Convert to distance
        distance_m = toa * self.c
        
        # Estimate multipath metrics
        multipath_metric = self.estimate_multipath_severity(cir)
        
        return {
            'distance_m': distance_m,
            'toa_seconds': toa,
            'quality': quality,
            'multipath_severity': multipath_metric,
            'cir': cir,
            'whitened_correlation': whitened,
            'whitening_filter': whitening_filter
        }
    
    def estimate_multipath_severity(self, cir: np.ndarray) -> float:
        """
        Estimate multipath severity from channel impulse response
        
        Args:
            cir: Channel impulse response
            
        Returns:
            Multipath severity metric (0=no multipath, 1=severe)
        """
        magnitude = np.abs(cir)
        
        # Find main peak
        peak_idx = np.argmax(magnitude)
        peak_value = magnitude[peak_idx]
        
        if peak_value < 1e-10:
            return 1.0  # No valid signal
            
        # Measure energy in sidelobes vs main peak
        # Define main peak region (±2 samples)
        main_peak_start = max(0, peak_idx - 2)
        main_peak_end = min(len(magnitude), peak_idx + 3)
        
        # Energy in main peak
        main_energy = np.sum(magnitude[main_peak_start:main_peak_end]**2)
        
        # Total energy
        total_energy = np.sum(magnitude**2)
        
        # Multipath severity: ratio of sidelobe energy to total
        if total_energy > 0:
            multipath_severity = 1.0 - (main_energy / total_energy)
        else:
            multipath_severity = 1.0
            
        return np.clip(multipath_severity, 0.0, 1.0)
    
    def simulate_multipath_channel(self,
                                  num_paths: int = 5,
                                  delay_spread_ns: float = 100,
                                  sampling_rate_hz: float = 250e6) -> np.ndarray:
        """
        Simulate a multipath channel for testing
        
        Args:
            num_paths: Number of multipath components
            delay_spread_ns: RMS delay spread in nanoseconds
            sampling_rate_hz: Sampling rate
            
        Returns:
            Channel impulse response
        """
        # Convert delay spread to samples
        delay_spread_samples = delay_spread_ns * 1e-9 * sampling_rate_hz
        
        # Generate random delays (exponential distribution)
        delays_samples = np.random.exponential(delay_spread_samples, num_paths)
        delays_samples[0] = 0  # Direct path at t=0
        
        # Generate complex amplitudes (Rayleigh fading)
        amplitudes = np.random.rayleigh(1.0, num_paths)
        phases = np.random.uniform(-np.pi, np.pi, num_paths)
        complex_amplitudes = amplitudes * np.exp(1j * phases)
        
        # Ensure direct path is strongest (Rician channel)
        K_factor = 10  # Rician K-factor in linear scale
        complex_amplitudes[0] = np.sqrt(K_factor) + complex_amplitudes[0]
        
        # Normalize total power
        complex_amplitudes = complex_amplitudes / np.sqrt(np.sum(np.abs(complex_amplitudes)**2))
        
        # Build CIR
        max_delay_samples = int(np.max(delays_samples) + 100)
        cir = np.zeros(max_delay_samples, dtype=complex)
        
        for delay, amplitude in zip(delays_samples, complex_amplitudes):
            delay_idx = int(delay)
            if delay_idx < len(cir):
                cir[delay_idx] += amplitude
                
        return cir


class AdaptiveMultipathProcessor:
    """
    Adaptive processor that adjusts whitening based on channel conditions
    """
    
    def __init__(self, base_whitener: MultipathWhiteningFilter):
        """
        Initialize adaptive processor
        
        Args:
            base_whitener: Base whitening filter instance
        """
        self.whitener = base_whitener
        self.channel_history = []
        self.max_history = 10
        
    def process_with_adaptation(self,
                               received_signal: np.ndarray,
                               transmitted_signal: np.ndarray,
                               sampling_rate_hz: float) -> dict:
        """
        Process ranging with adaptive whitening
        
        Args:
            received_signal: Received waveform
            transmitted_signal: Reference waveform
            sampling_rate_hz: Sampling rate
            
        Returns:
            Ranging results with adaptive processing
        """
        # First pass: estimate channel and SNR
        initial_result = self.whitener.process_ranging_measurement(
            received_signal, transmitted_signal, sampling_rate_hz
        )
        
        # Estimate SNR from initial result
        signal_power = np.max(np.abs(initial_result['cir'])**2)
        noise_power = np.median(np.abs(initial_result['cir'])**2)
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # Adapt regularization based on multipath severity
        multipath_severity = initial_result['multipath_severity']
        
        if multipath_severity > 0.7:
            # Severe multipath: more aggressive whitening
            self.whitener.regularization = 0.001
        elif multipath_severity > 0.3:
            # Moderate multipath: balanced whitening
            self.whitener.regularization = 0.01
        else:
            # Low multipath: conservative whitening
            self.whitener.regularization = 0.1
            
        # Second pass with adapted parameters
        final_result = self.whitener.process_ranging_measurement(
            received_signal, transmitted_signal, sampling_rate_hz, snr_db
        )
        
        # Add adaptation info
        final_result['snr_db'] = snr_db
        final_result['adapted_regularization'] = self.whitener.regularization
        
        # Update history
        self.channel_history.append({
            'multipath_severity': multipath_severity,
            'snr_db': snr_db,
            'distance': final_result['distance_m']
        })
        
        if len(self.channel_history) > self.max_history:
            self.channel_history.pop(0)
            
        return final_result


# Example usage for S-band parameters
if __name__ == "__main__":
    # S-band configuration
    fc = 2.45e9  # 2.45 GHz center frequency
    bandwidth = 200e6  # 200 MHz bandwidth
    fs = 250e6  # 250 MS/s sampling rate
    
    # Create whitening filter
    whitener = MultipathWhiteningFilter(
        regularization=0.01,
        filter_length=1024,
        bandwidth_mhz=200
    )
    
    print(f"Range resolution: {whitener.range_resolution:.3f} m")
    print(f"Time resolution: {whitener.range_resolution/whitener.c*1e9:.1f} ns")
    
    # Simulate multipath channel
    cir = whitener.simulate_multipath_channel(
        num_paths=5,
        delay_spread_ns=100,
        sampling_rate_hz=fs
    )
    
    multipath_severity = whitener.estimate_multipath_severity(cir)
    print(f"Multipath severity: {multipath_severity:.2%}")
    
    # Compute whitening filter
    w = whitener.compute_whitening_filter(cir, snr_db=20)
    print(f"Whitening filter length: {len(w)}")
    
    # Apply whitening
    whitened = whitener.apply_whitening(cir, w)
    
    # Compare peak sharpness
    original_peak = np.max(np.abs(cir))
    whitened_peak = np.max(np.abs(whitened))
    print(f"Peak enhancement: {whitened_peak/original_peak:.1f}x")