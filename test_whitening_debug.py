#!/usr/bin/env python3
"""
Debug whitening filter issues - why is it making things worse?
"""

import numpy as np
import matplotlib.pyplot as plt
from src.signal_processing.multipath_whitening import MultipathWhiteningFilter


def debug_whitening_failure():
    """Debug why whitening makes severe multipath worse"""
    print("="*70)
    print("DEBUGGING WHITENING FILTER FAILURE")
    print("="*70)
    
    c = 299792458.0
    fs = 250e6  # 250 MS/s
    true_distance = 10.0
    true_delay_samples = int(true_distance / c * fs)
    
    # Create whitener
    whitener = MultipathWhiteningFilter(
        regularization=0.01,
        filter_length=256,
        bandwidth_mhz=200
    )
    
    print(f"\nTrue distance: {true_distance} m")
    print(f"True delay: {true_delay_samples} samples @ {fs/1e6:.0f} MS/s")
    print(f"Range resolution: {whitener.range_resolution:.3f} m")
    
    # Test Case 1: Simple two-path channel
    print("\n" + "="*50)
    print("TEST 1: Simple Two-Path Channel")
    print("="*50)
    
    # Create simple channel: direct + one reflection
    cir_simple = np.zeros(100, dtype=complex)
    cir_simple[true_delay_samples] = 1.0  # Direct path
    cir_simple[true_delay_samples + 10] = 0.5  # Reflection 10 samples later
    
    test_whitening(whitener, cir_simple, true_distance, fs, "Two-path")
    
    # Test Case 2: Severe multipath (weak direct path)
    print("\n" + "="*50)
    print("TEST 2: Severe Multipath (Weak Direct Path)")
    print("="*50)
    
    cir_severe = np.zeros(200, dtype=complex)
    cir_severe[true_delay_samples] = 0.2  # Weak direct path
    cir_severe[true_delay_samples + 20] = 1.0  # Strong reflection
    cir_severe[true_delay_samples + 40] = 0.8  # Another strong reflection
    cir_severe[true_delay_samples + 60] = 0.6
    
    test_whitening(whitener, cir_severe, true_distance, fs, "Severe multipath")
    
    # Test Case 3: NLOS (no direct path)
    print("\n" + "="*50)
    print("TEST 3: NLOS (No Direct Path)")
    print("="*50)
    
    cir_nlos = np.zeros(200, dtype=complex)
    # No signal at true_delay_samples!
    cir_nlos[true_delay_samples + 30] = 1.0  # First arrival is reflected
    cir_nlos[true_delay_samples + 50] = 0.8
    cir_nlos[true_delay_samples + 70] = 0.6
    
    test_whitening(whitener, cir_nlos, true_distance, fs, "NLOS")
    
    # Test Case 4: Check regularization impact
    print("\n" + "="*50)
    print("TEST 4: Regularization Parameter Sweep")
    print("="*50)
    
    regularizations = [0.0001, 0.001, 0.01, 0.1, 1.0]
    
    for reg in regularizations:
        whitener_test = MultipathWhiteningFilter(
            regularization=reg,
            filter_length=256,
            bandwidth_mhz=200
        )
        
        # Use severe multipath case
        whitening_coeffs = whitener_test.compute_whitening_filter(cir_severe, snr_db=20)
        whitened = whitener_test.apply_whitening(cir_severe, whitening_coeffs)
        
        # Find peak
        peak_idx = np.argmax(np.abs(whitened))
        estimated_distance = peak_idx / fs * c
        error = abs(estimated_distance - true_distance)
        
        print(f"  λ = {reg:6.4f}: Error = {error:6.2f} m")


def test_whitening(whitener, cir, true_distance, fs, scenario_name):
    """Test whitening on a given CIR"""
    c = 299792458.0
    
    # Original peak
    original_peak_idx = np.argmax(np.abs(cir))
    original_distance = original_peak_idx / fs * c
    original_error = abs(original_distance - true_distance)
    
    print(f"\n{scenario_name}:")
    print(f"  Original peak at sample {original_peak_idx}")
    print(f"  Original distance: {original_distance:.3f} m")
    print(f"  Original error: {original_error:.3f} m")
    
    # Apply whitening
    whitening_coeffs = whitener.compute_whitening_filter(cir, snr_db=20)
    whitened = whitener.apply_whitening(cir, whitening_coeffs)
    
    # Check what whitening did
    whitened_peak_idx = np.argmax(np.abs(whitened))
    whitened_distance = whitened_peak_idx / fs * c
    whitened_error = abs(whitened_distance - true_distance)
    
    print(f"  Whitened peak at sample {whitened_peak_idx}")
    print(f"  Whitened distance: {whitened_distance:.3f} m")
    print(f"  Whitened error: {whitened_error:.3f} m")
    
    if whitened_error > original_error:
        print(f"  ⚠️ WHITENING MADE IT WORSE by {whitened_error - original_error:.3f} m!")
    else:
        print(f"  ✓ Whitening improved by {original_error - whitened_error:.3f} m")
    
    # Plot to visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    samples = np.arange(len(cir))
    ax1.stem(samples, np.abs(cir), basefmt=' ', label='Original CIR')
    ax1.axvline(x=original_peak_idx, color='r', linestyle='--', alpha=0.5, label='Peak')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Magnitude')
    ax1.set_title(f'{scenario_name}: Original')
    ax1.legend()
    ax1.set_xlim([0, min(100, len(cir))])
    
    ax2.stem(samples[:len(whitened)], np.abs(whitened), basefmt=' ', label='Whitened CIR')
    ax2.axvline(x=whitened_peak_idx, color='g', linestyle='--', alpha=0.5, label='Peak')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Magnitude')
    ax2.set_title(f'{scenario_name}: Whitened')
    ax2.legend()
    ax2.set_xlim([0, min(100, len(whitened))])
    
    plt.tight_layout()
    plt.savefig(f'debug_whitening_{scenario_name.replace(" ", "_").lower()}.png')
    plt.show()
    
    return whitened_error


def analyze_whitening_filter():
    """Analyze the whitening filter itself"""
    print("\n" + "="*70)
    print("ANALYZING WHITENING FILTER BEHAVIOR")
    print("="*70)
    
    # Create simple impulse response
    h = np.zeros(50)
    h[10] = 1.0  # Main path
    h[15] = 0.5  # Multipath
    
    # Compute whitening filter
    whitener = MultipathWhiteningFilter(regularization=0.01)
    w = whitener.compute_whitening_filter(h, snr_db=20)
    
    print(f"\nChannel impulse response:")
    print(f"  Main path at index 10")
    print(f"  Multipath at index 15 (50% strength)")
    
    print(f"\nWhitening filter stats:")
    print(f"  Length: {len(w)}")
    print(f"  Max value: {np.max(np.abs(w)):.3f}")
    print(f"  Energy: {np.sum(np.abs(w)**2):.3f}")
    
    # Apply to original
    result = np.convolve(h, w, mode='same')
    
    print(f"\nResult after whitening:")
    print(f"  Peak at index: {np.argmax(np.abs(result))}")
    print(f"  Peak value: {np.max(np.abs(result)):.3f}")
    
    # The problem: convolution shifts the peak!
    print("\n⚠️ PROBLEM IDENTIFIED:")
    print("The convolution in apply_whitening() can shift the peak position!")
    print("This causes incorrect distance estimates, especially with severe multipath.")


if __name__ == "__main__":
    debug_whitening_failure()
    analyze_whitening_filter()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
The whitening filter is failing because:

1. Convolution shifts peak positions unpredictably
2. Regularization is too aggressive for severe multipath
3. The filter assumes a direct path exists (fails in NLOS)
4. Peak detection after whitening is unreliable

The filter needs to:
- Preserve time-of-arrival of earliest path
- Handle NLOS scenarios gracefully
- Use adaptive regularization based on channel conditions
- Apply phase-preserving whitening
""")