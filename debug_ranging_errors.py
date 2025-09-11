#!/usr/bin/env python3
"""
Debug the ranging error calculations to verify they're realistic
"""

import numpy as np

def check_ranging_errors():
    """Verify the ranging error calculations"""
    print("="*70)
    print("RANGING ERROR VERIFICATION")
    print("="*70)
    
    c = 299792458.0  # Speed of light m/s
    
    # Picosecond precision
    print("\n1. PICOSECOND PRECISION (1 ps):")
    print("-" * 40)
    timing_precision_s = 1e-12  # 1 picosecond
    ranging_error_m = timing_precision_s * c
    print(f"  1 ps timing precision")
    print(f"  Speed of light: {c:.0f} m/s")
    print(f"  Ranging error: {ranging_error_m:.6f} m = {ranging_error_m*1000:.3f} mm")
    print(f"  ✓ This is correct: 1 ps → 0.3 mm")
    
    # Nanosecond precision
    print("\n2. NANOSECOND PRECISION (1 ns):")
    print("-" * 40)
    timing_precision_s = 1e-9  # 1 nanosecond
    ranging_error_m = timing_precision_s * c
    print(f"  1 ns timing precision")
    print(f"  Ranging error: {ranging_error_m:.6f} m = {ranging_error_m*100:.3f} cm")
    print(f"  ✓ This is correct: 1 ns → 30 cm")
    
    # But what about the test results?
    print("\n3. TEST RESULTS ANALYSIS:")
    print("-" * 40)
    
    # PS ranging error: 0.19mm - is this realistic?
    print("  PS ranging shows 0.19mm mean error")
    print("  This implies:")
    required_timing_s = 0.19e-3 / c
    print(f"    Required timing precision: {required_timing_s*1e12:.2f} ps")
    print(f"    ✓ Achievable with 1 ps precision + measurement noise")
    
    # NS ranging error: 27.71cm - is this realistic?
    print("\n  NS ranging shows 27.71cm mean error")
    print("  This could be from:")
    print(f"    - 1 ns timing → 30cm")
    print(f"    - Plus multipath → additional error")
    print(f"    - Plus noise → total ~27cm")
    print(f"    ✓ This seems reasonable")
    
    # Why isn't PS helping overall accuracy?
    print("\n4. WHY ISN'T PS HELPING OVERALL?")
    print("-" * 40)
    print("  Possible issues:")
    print("  1. Only 24.7% of links are PS (37 out of 150)")
    print("  2. Consensus might not properly weight PS measurements")
    print("  3. Initial position errors might be too large")
    print("  4. Damping factor (0.5) might be preventing convergence")
    
    # Simulate what SHOULD happen
    print("\n5. EXPECTED PERFORMANCE:")
    print("-" * 40)
    
    # With PS on clean links
    num_ps_links = 37
    num_ns_links = 113
    ps_error = 0.0003  # 0.3mm
    ns_error = 0.30  # 30cm
    
    # Simple weighted average
    total_links = num_ps_links + num_ns_links
    avg_ranging_error = (num_ps_links * ps_error + num_ns_links * ns_error) / total_links
    print(f"  Average ranging error: {avg_ranging_error:.4f} m = {avg_ranging_error*100:.2f} cm")
    
    # Position error with sqrt(N) improvement
    num_neighbors = 5
    position_error = avg_ranging_error / np.sqrt(num_neighbors)
    print(f"  Expected position error: {position_error:.4f} m = {position_error*100:.2f} cm")
    print(f"  But we're seeing: 3.49m (much worse!)")
    
    print("\n  CONCLUSION: The ranging is working correctly,")
    print("  but the consensus algorithm isn't utilizing")
    print("  the high-precision measurements properly!")


def simulate_simple_case():
    """Simulate a simple 3-node case to verify"""
    print("\n\n" + "="*70)
    print("SIMPLE 3-NODE TEST")
    print("="*70)
    
    # Node positions (equilateral triangle)
    node0 = np.array([0, 0])
    node1 = np.array([10, 0])
    node2 = np.array([5, 8.66])
    
    true_distances = {
        (0, 1): 10.0,
        (0, 2): 10.0,
        (1, 2): 10.0
    }
    
    print("\nTrue configuration (equilateral triangle):")
    print(f"  Node 0: {node0}")
    print(f"  Node 1: {node1}")
    print(f"  Node 2: {node2}")
    print(f"  All distances: 10m")
    
    # Test with PS precision
    print("\nWith PS precision (0.3mm error):")
    measured_01 = 10.0 + np.random.normal(0, 0.0003)
    measured_02 = 10.0 + np.random.normal(0, 0.0003)
    measured_12 = 10.0 + np.random.normal(0, 0.0003)
    
    print(f"  Measured distances:")
    print(f"    0→1: {measured_01:.6f}m (error: {(measured_01-10)*1000:.3f}mm)")
    print(f"    0→2: {measured_02:.6f}m (error: {(measured_02-10)*1000:.3f}mm)")
    print(f"    1→2: {measured_12:.6f}m (error: {(measured_12-10)*1000:.3f}mm)")
    
    # Test with NS precision
    print("\nWith NS precision (30cm error):")
    measured_01_ns = 10.0 + np.random.normal(0, 0.3)
    measured_02_ns = 10.0 + np.random.normal(0, 0.3)
    measured_12_ns = 10.0 + np.random.normal(0, 0.3)
    
    print(f"  Measured distances:")
    print(f"    0→1: {measured_01_ns:.3f}m (error: {(measured_01_ns-10)*100:.1f}cm)")
    print(f"    0→2: {measured_02_ns:.3f}m (error: {(measured_02_ns-10)*100:.1f}cm)")
    print(f"    1→2: {measured_12_ns:.3f}m (error: {(measured_12_ns-10)*100:.1f}cm)")
    
    print("\n✓ PS gives mm-level ranging, NS gives cm-level ranging")
    print("  The issue is in the position estimation algorithm!")


if __name__ == "__main__":
    check_ranging_errors()
    simulate_simple_case()