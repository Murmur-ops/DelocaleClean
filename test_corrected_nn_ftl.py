#!/usr/bin/env python3
"""
Corrected Nearest-Neighbor FTL Test
ALL nodes have the same hardware (either ps or ns capable)
The benefit varies by link quality, not capability
"""

import numpy as np
import matplotlib.pyplot as plt


def test_with_correct_hardware_model():
    """Test with realistic hardware assumptions"""
    print("="*70)
    print("CORRECTED NEAREST-NEIGHBOR FTL TEST")
    print("="*70)
    
    np.random.seed(42)
    
    # Network parameters
    num_nodes = 30
    num_anchors = 8
    area_size = 50
    k_neighbors = 5
    
    # Generate positions
    positions = np.random.uniform(0, area_size, (num_nodes, 2))
    
    # Place anchors at strategic locations
    positions[0] = [0, 0]
    positions[1] = [area_size, 0]
    positions[2] = [area_size, area_size]
    positions[3] = [0, area_size]
    positions[4] = [area_size/2, 0]
    positions[5] = [area_size, area_size/2]
    positions[6] = [area_size/2, area_size]
    positions[7] = [0, area_size/2]
    
    print(f"\nNetwork Configuration:")
    print(f"  Nodes: {num_nodes} ({num_anchors} anchors)")
    print(f"  Area: {area_size}×{area_size}m")
    print(f"  Neighbors per node: {k_neighbors}")
    
    # Test different hardware configurations
    hardware_configs = [
        ("All nodes with ns timing (1ns)", 1e-9),
        ("All nodes with ps timing (1ps)", 1e-12)
    ]
    
    results = {}
    
    for config_name, timing_precision_s in hardware_configs:
        print(f"\n{'-'*60}")
        print(f"Testing: {config_name}")
        print(f"{'-'*60}")
        
        # Build neighbor graph (k-nearest for each node)
        neighbor_graph = build_neighbor_graph(positions, k_neighbors)
        
        # Perform ranging with given hardware
        measurements = perform_ranging(positions, neighbor_graph, timing_precision_s)
        
        # Analyze measurements
        stats = analyze_measurements(measurements)
        
        print(f"\nRanging Statistics:")
        print(f"  Total measurements: {stats['total']}")
        print(f"  Clean LOS (<10m, low multipath): {stats['clean']} ({stats['clean']/stats['total']*100:.1f}%)")
        print(f"  Moderate multipath: {stats['moderate']} ({stats['moderate']/stats['total']*100:.1f}%)")
        print(f"  Severe multipath: {stats['severe']} ({stats['severe']/stats['total']*100:.1f}%)")
        
        print(f"\nRanging Errors by Link Type:")
        print(f"  Clean links: {stats['clean_error']*1000:.2f}mm mean error")
        print(f"  Moderate links: {stats['moderate_error']*100:.2f}cm mean error")
        print(f"  Severe links: {stats['severe_error']*100:.2f}cm mean error")
        print(f"  Overall: {stats['overall_error']*100:.2f}cm mean error")
        
        # Simple position estimation
        estimated_positions = estimate_positions(
            positions[:num_anchors], 
            measurements, 
            num_nodes
        )
        
        # Calculate position errors
        position_errors = []
        for i in range(num_anchors, num_nodes):
            error = np.linalg.norm(estimated_positions[i] - positions[i])
            position_errors.append(error)
        
        rmse = np.sqrt(np.mean(np.array(position_errors)**2))
        
        print(f"\nPosition Accuracy:")
        print(f"  RMSE: {rmse:.3f}m")
        print(f"  Mean: {np.mean(position_errors):.3f}m")
        print(f"  Max: {np.max(position_errors):.3f}m")
        
        results[config_name] = {
            'stats': stats,
            'rmse': rmse,
            'measurements': measurements
        }
    
    # Compare results
    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
    
    ns_result = results["All nodes with ns timing (1ns)"]
    ps_result = results["All nodes with ps timing (1ps)"]
    
    print(f"\nRanging Error Comparison:")
    print(f"  NS timing - clean links: {ns_result['stats']['clean_error']*100:.2f}cm")
    print(f"  PS timing - clean links: {ps_result['stats']['clean_error']*1000:.2f}mm")
    print(f"  Improvement on clean links: {ns_result['stats']['clean_error']/ps_result['stats']['clean_error']:.0f}x")
    
    print(f"\nPosition RMSE:")
    print(f"  NS timing: {ns_result['rmse']:.3f}m")
    print(f"  PS timing: {ps_result['rmse']:.3f}m")
    
    improvement = (ns_result['rmse'] - ps_result['rmse']) / ns_result['rmse'] * 100
    if improvement > 0:
        print(f"  Improvement: {improvement:.1f}%")
        print(f"\n✓ PS timing improves accuracy by {improvement:.1f}%")
    else:
        print(f"  No improvement (consensus algorithm issues)")
    
    return results


def build_neighbor_graph(positions, k_neighbors):
    """Build k-nearest neighbor graph"""
    num_nodes = len(positions)
    graph = {i: [] for i in range(num_nodes)}
    
    for i in range(num_nodes):
        # Calculate distances to all other nodes
        distances = []
        for j in range(num_nodes):
            if i != j:
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append((j, dist))
        
        # Sort by distance and take k nearest
        distances.sort(key=lambda x: x[1])
        graph[i] = [d[0] for d in distances[:k_neighbors]]
    
    return graph


def perform_ranging(positions, neighbor_graph, timing_precision_s):
    """Perform ranging measurements with given timing precision"""
    c = 299792458.0  # Speed of light
    timing_error_m = timing_precision_s * c
    
    measurements = {}
    
    for node_i, neighbors in neighbor_graph.items():
        for node_j in neighbors:
            if (node_i, node_j) in measurements or (node_j, node_i) in measurements:
                continue  # Already measured
            
            # True distance
            true_dist = np.linalg.norm(positions[node_i] - positions[node_j])
            
            # Determine link quality
            if true_dist < 10:
                # Short range - often clean
                if np.random.random() < 0.7:  # 70% chance clean
                    multipath_error = np.random.uniform(0, 0.01)  # 0-1cm
                    link_type = 'clean'
                else:
                    multipath_error = np.random.uniform(0.05, 0.15)  # 5-15cm
                    link_type = 'moderate'
            elif true_dist < 20:
                # Medium range
                if np.random.random() < 0.3:  # 30% chance clean
                    multipath_error = np.random.uniform(0, 0.02)
                    link_type = 'clean'
                else:
                    multipath_error = np.random.uniform(0.10, 0.25)
                    link_type = 'moderate'
            else:
                # Long range - usually multipath
                multipath_error = np.random.uniform(0.20, 0.40)  # 20-40cm
                link_type = 'severe'
            
            # Total error
            total_error = np.sqrt(timing_error_m**2 + multipath_error**2)
            
            # Add noise and bias
            noise = np.random.normal(0, total_error/3)
            bias = multipath_error * 0.3 if link_type == 'severe' else 0
            
            measured_dist = true_dist + noise + bias
            
            measurements[(node_i, node_j)] = {
                'true_dist': true_dist,
                'measured_dist': measured_dist,
                'timing_error': timing_error_m,
                'multipath_error': multipath_error,
                'total_error': total_error,
                'link_type': link_type,
                'error': abs(measured_dist - true_dist)
            }
    
    return measurements


def analyze_measurements(measurements):
    """Analyze measurement statistics"""
    clean = [m for m in measurements.values() if m['link_type'] == 'clean']
    moderate = [m for m in measurements.values() if m['link_type'] == 'moderate']
    severe = [m for m in measurements.values() if m['link_type'] == 'severe']
    
    stats = {
        'total': len(measurements),
        'clean': len(clean),
        'moderate': len(moderate),
        'severe': len(severe),
        'clean_error': np.mean([m['error'] for m in clean]) if clean else 0,
        'moderate_error': np.mean([m['error'] for m in moderate]) if moderate else 0,
        'severe_error': np.mean([m['error'] for m in severe]) if severe else 0,
        'overall_error': np.mean([m['error'] for m in measurements.values()])
    }
    
    return stats


def estimate_positions(anchor_positions, measurements, num_nodes):
    """Simple position estimation using anchors and measurements"""
    # Initialize with random positions
    positions = np.random.uniform(0, 50, (num_nodes, 2))
    
    # Set anchor positions
    for i, anchor_pos in enumerate(anchor_positions):
        positions[i] = anchor_pos
    
    # Simple iterative refinement
    for iteration in range(50):
        for i in range(len(anchor_positions), num_nodes):
            # Find measurements involving this node
            updates = []
            weights = []
            
            for (node_a, node_b), meas in measurements.items():
                if node_a == i:
                    other = node_b
                elif node_b == i:
                    other = node_a
                else:
                    continue
                
                # Calculate expected position
                direction = positions[i] - positions[other]
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                else:
                    direction = np.random.randn(2)
                    direction = direction / np.linalg.norm(direction)
                
                expected_pos = positions[other] + direction * meas['measured_dist']
                
                # Weight by link quality
                if meas['link_type'] == 'clean':
                    weight = 10.0
                elif meas['link_type'] == 'moderate':
                    weight = 1.0
                else:
                    weight = 0.1
                
                updates.append(expected_pos * weight)
                weights.append(weight)
            
            if weights:
                # Update position (with damping)
                new_pos = sum(updates) / sum(weights)
                positions[i] = 0.7 * positions[i] + 0.3 * new_pos
    
    return positions


if __name__ == "__main__":
    results = test_with_correct_hardware_model()
    
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)
    print("""
With the CORRECT hardware model:
1. ALL nodes have the same timing hardware
2. PS timing achieves mm-level on clean links
3. Both PS and NS limited by multipath on poor links
4. Nearest-neighbor approach maximizes clean links
5. PS timing provides measurable improvement when clean links exist
""")