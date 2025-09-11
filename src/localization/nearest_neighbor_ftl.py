"""
Nearest-Neighbor FTL with High-Precision Ranging
Uses picosecond timing on clean LOS links to nearest neighbors
Then uses consensus to achieve network-wide accuracy
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import networkx as nx


@dataclass
class LinkQuality:
    """Quality metrics for a communication link"""
    node_id: int
    distance: float
    rssi_dbm: float
    multipath_severity: float  # 0=clean, 1=severe
    is_los: bool
    snr_db: float
    
    @property
    def is_clean(self) -> bool:
        """Check if link is clean enough for ps-precision ranging"""
        return self.is_los and self.multipath_severity < 0.1 and self.snr_db > 20


class NearestNeighborFTL:
    """
    FTL using only nearest neighbors with clean channels
    Enables picosecond timing to actually improve accuracy
    """
    
    def __init__(self, 
                 node_id: int,
                 position: np.ndarray,
                 k_neighbors: int = 5,
                 max_neighbor_distance: float = 10.0):
        """
        Initialize nearest-neighbor FTL node
        
        Args:
            node_id: Node identifier
            position: True position (for simulation)
            k_neighbors: Number of nearest neighbors to use
            max_neighbor_distance: Maximum distance to consider a neighbor
        """
        self.node_id = node_id
        self.true_position = np.array(position, dtype=np.float64)
        self.estimated_position = np.array(position, dtype=np.float64) + np.random.randn(len(position)) * 5
        self.k_neighbors = k_neighbors
        self.max_neighbor_distance = max_neighbor_distance
        
        # Neighbor information
        self.neighbors = []  # List of neighbor IDs
        self.link_qualities = {}  # node_id -> LinkQuality
        self.range_measurements = {}  # node_id -> (distance, precision)
        
        # Timing capabilities
        self.timing_precision_ps = 1.0  # 1 ps capability
        self.timing_precision_ns = 1.0  # 1 ns fallback
        
        # Physical constants
        self.c = 299792458.0  # Speed of light m/s
        
    def discover_neighbors(self, all_nodes: List['NearestNeighborFTL']) -> List[int]:
        """
        Discover and rank potential neighbors
        EVERY node gets k neighbors, even if SNR is poor
        
        Args:
            all_nodes: List of all nodes in network
            
        Returns:
            List of selected neighbor IDs
        """
        candidates = []
        
        for node in all_nodes:
            if node.node_id == self.node_id:
                continue
                
            # Calculate true distance
            true_distance = np.linalg.norm(self.true_position - node.true_position)
            
            # REMOVED distance cutoff - we need connectivity!
            # Even distant nodes can be neighbors if they're the closest available
                
            # Simulate link quality assessment
            link_quality = self._assess_link_quality(node, true_distance)
            self.link_qualities[node.node_id] = link_quality
            
            # Add ALL nodes as candidates (no RSSI cutoff)
            # We'll take k-nearest regardless of signal quality
            candidates.append((node.node_id, link_quality))
        
        # Sort by distance first (true nearest neighbors)
        # Then by quality for tie-breaking
        candidates.sort(key=lambda x: (
            x[1].distance,  # Primary: actual distance
            -x[1].snr_db,    # Secondary: signal quality
            x[1].multipath_severity  # Tertiary: multipath
        ))
        
        # ALWAYS select k neighbors (or all if fewer than k)
        self.neighbors = [c[0] for c in candidates[:min(self.k_neighbors, len(candidates))]]
        
        return self.neighbors
    
    def _assess_link_quality(self, other_node: 'NearestNeighborFTL', 
                            true_distance: float) -> LinkQuality:
        """
        Assess quality of link to another node
        Now handles any distance (no cutoffs)
        
        Args:
            other_node: Target node
            true_distance: True distance to node
            
        Returns:
            Link quality metrics
        """
        # Path loss model (free space + environment)
        # Protect against log(0)
        safe_distance = max(true_distance, 0.1)
        path_loss_db = 20 * np.log10(safe_distance) + 20 * np.log10(2.45e9) - 147.55
        
        # Transmit power 20 dBm, antenna gains 6 dBi each
        rssi_dbm = 20 + 6 + 6 - path_loss_db
        
        # Add some randomness
        rssi_dbm += np.random.normal(0, 3)
        
        # Multipath severity based on distance and environment
        if true_distance < 5:
            # Very short range - usually LOS
            multipath_severity = np.random.uniform(0, 0.05)
            is_los = np.random.random() > 0.1  # 90% chance of LOS
        elif true_distance < 10:
            # Short range - often LOS
            multipath_severity = np.random.uniform(0, 0.2)
            is_los = np.random.random() > 0.3  # 70% chance of LOS
        elif true_distance < 20:
            # Medium range - mixed
            multipath_severity = np.random.uniform(0.2, 0.6)
            is_los = np.random.random() > 0.5  # 50% chance of LOS
        else:
            # Long range - multipath dominates, poor SNR
            multipath_severity = np.random.uniform(0.5, 0.9)
            is_los = np.random.random() > 0.8  # 20% chance of LOS
        
        # SNR based on RSSI and noise floor
        noise_floor_dbm = -91  # For 200 MHz bandwidth
        snr_db = rssi_dbm - noise_floor_dbm
        
        return LinkQuality(
            node_id=other_node.node_id,
            distance=true_distance,
            rssi_dbm=rssi_dbm,
            multipath_severity=multipath_severity,
            is_los=is_los,
            snr_db=snr_db
        )
    
    def perform_ranging(self, neighbor_nodes: Dict[int, 'NearestNeighborFTL']) -> Dict[int, float]:
        """
        Perform high-precision ranging to neighbors
        
        Args:
            neighbor_nodes: Dictionary of neighbor nodes
            
        Returns:
            Dictionary of measured distances
        """
        for neighbor_id in self.neighbors:
            if neighbor_id not in neighbor_nodes:
                continue
                
            neighbor = neighbor_nodes[neighbor_id]
            link = self.link_qualities[neighbor_id]
            
            # True distance for simulation
            true_distance = link.distance
            
            if link.is_clean:
                # Clean link - use picosecond precision!
                timing_error_s = self.timing_precision_ps * 1e-12
                distance_error = timing_error_s * self.c
                
                # Add small measurement noise
                noise = np.random.normal(0, distance_error)
                measured_distance = true_distance + noise
                
                # Store with precision indicator
                self.range_measurements[neighbor_id] = (measured_distance, 'ps')
                
            else:
                # Multipath present - ps timing doesn't help
                # Fallback to ns precision + multipath error
                timing_error_s = self.timing_precision_ns * 1e-9
                timing_error_m = timing_error_s * self.c
                
                # Multipath adds significant error
                multipath_error_m = link.multipath_severity * 0.3  # Up to 30cm
                
                total_error = np.sqrt(timing_error_m**2 + multipath_error_m**2)
                noise = np.random.normal(0, total_error)
                
                # Multipath often causes positive bias
                bias = multipath_error_m * 0.5 if not link.is_los else 0
                measured_distance = true_distance + noise + bias
                
                self.range_measurements[neighbor_id] = (measured_distance, 'ns')
        
        return {nid: r[0] for nid, r in self.range_measurements.items()}
    
    def estimate_position_from_neighbors(self, 
                                        neighbor_positions: Dict[int, np.ndarray],
                                        anchor_positions: Dict[int, np.ndarray] = None,
                                        damping: float = 0.5) -> np.ndarray:
        """
        Estimate position using only neighbor measurements
        With damping to prevent explosion
        
        Args:
            neighbor_positions: Current position estimates of neighbors
            anchor_positions: Known anchor positions (if any)
            damping: Damping factor (0-1) to prevent instability
            
        Returns:
            Estimated position
        """
        if len(self.range_measurements) < 2:  # Reduced from 3 - we can work with 2
            # Not enough neighbors
            return self.estimated_position
            
        # Simple weighted average approach (more stable than linear system)
        weighted_positions = []
        weights = []
        
        for neighbor_id, (measured_dist, precision) in self.range_measurements.items():
            # Get neighbor position
            neighbor_pos = None
            if anchor_positions and neighbor_id in anchor_positions:
                neighbor_pos = anchor_positions[neighbor_id]
                base_weight = 10.0  # High weight for anchors
            elif neighbor_id in neighbor_positions:
                neighbor_pos = neighbor_positions[neighbor_id]
                base_weight = 1.0
            else:
                continue
                
            if neighbor_pos is None:
                continue
                
            # Calculate weight based on link quality
            link = self.link_qualities.get(neighbor_id)
            if link:
                if precision == 'ps' and link.is_clean:
                    quality_weight = 5.0  # High weight for clean ps-precision
                else:
                    # Weight inversely proportional to expected error
                    expected_error = measured_dist * 0.01  # 1% baseline
                    if precision == 'ns' or link.multipath_severity > 0.3:
                        expected_error += 0.3  # Add multipath error
                    quality_weight = 1.0 / (1.0 + expected_error)
            else:
                quality_weight = 1.0
                
            total_weight = base_weight * quality_weight
            
            # Project from neighbor along ranging circle
            direction = self.estimated_position - neighbor_pos
            direction_norm = np.linalg.norm(direction)
            if direction_norm < 0.01:
                # Too close, use random direction
                direction = np.random.randn(len(neighbor_pos))
                direction_norm = np.linalg.norm(direction)
            
            direction = direction / direction_norm
            estimated_pos = neighbor_pos + direction * measured_dist
            
            weighted_positions.append(estimated_pos * total_weight)
            weights.append(total_weight)
        
        if not weights:
            return self.estimated_position
            
        # Weighted average
        total_weight = sum(weights)
        if total_weight > 0:
            new_position = sum(weighted_positions) / total_weight
            
            # Apply damping to prevent explosion
            self.estimated_position = (damping * self.estimated_position + 
                                      (1 - damping) * new_position)
            
            # Bound check - keep within reasonable area
            max_coord = 100.0  # Maximum reasonable coordinate
            self.estimated_position = np.clip(self.estimated_position, 
                                             -max_coord, max_coord)
        
        return self.estimated_position
    
    def get_ranging_statistics(self) -> Dict:
        """
        Get statistics about ranging performance
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'node_id': self.node_id,
            'num_neighbors': len(self.neighbors),
            'num_ps_links': sum(1 for r in self.range_measurements.values() if r[1] == 'ps'),
            'num_ns_links': sum(1 for r in self.range_measurements.values() if r[1] == 'ns'),
            'avg_multipath': np.mean([l.multipath_severity for l in self.link_qualities.values()]) if self.link_qualities else 0,
            'los_ratio': sum(1 for l in self.link_qualities.values() if l.is_los) / len(self.link_qualities) if self.link_qualities else 0
        }
        
        # Calculate ranging errors if we know true distances
        ranging_errors_ps = []
        ranging_errors_ns = []
        
        for neighbor_id, (measured, precision) in self.range_measurements.items():
            true_dist = self.link_qualities[neighbor_id].distance
            error = abs(measured - true_dist)
            
            if precision == 'ps':
                ranging_errors_ps.append(error)
            else:
                ranging_errors_ns.append(error)
        
        if ranging_errors_ps:
            stats['ps_ranging_error_mean'] = np.mean(ranging_errors_ps)
            stats['ps_ranging_error_std'] = np.std(ranging_errors_ps)
        
        if ranging_errors_ns:
            stats['ns_ranging_error_mean'] = np.mean(ranging_errors_ns)
            stats['ns_ranging_error_std'] = np.std(ranging_errors_ns)
            
        return stats


class NearestNeighborNetwork:
    """
    Network of nodes using nearest-neighbor FTL
    """
    
    def __init__(self, positions: np.ndarray, anchor_indices: List[int] = None):
        """
        Initialize network
        
        Args:
            positions: Array of node positions (N x d)
            anchor_indices: Indices of anchor nodes
        """
        self.num_nodes = len(positions)
        self.dimension = positions.shape[1]
        self.anchor_indices = anchor_indices or []
        
        # Create nodes
        self.nodes = []
        for i, pos in enumerate(positions):
            node = NearestNeighborFTL(i, pos, k_neighbors=5)
            self.nodes.append(node)
        
        # Build connectivity graph
        self.graph = nx.Graph()
        
    def discover_topology(self):
        """
        Each node discovers its nearest neighbors
        """
        for node in self.nodes:
            neighbors = node.discover_neighbors(self.nodes)
            
            # Add edges to graph
            for neighbor_id in neighbors:
                self.graph.add_edge(node.node_id, neighbor_id)
        
        print(f"Network topology: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        # Check connectivity
        if nx.is_connected(self.graph):
            print("✓ Network is fully connected")
        else:
            components = list(nx.connected_components(self.graph))
            print(f"⚠ Network has {len(components)} disconnected components")
    
    def perform_all_ranging(self):
        """
        All nodes perform ranging to their neighbors
        """
        # Create node lookup
        node_dict = {node.node_id: node for node in self.nodes}
        
        for node in self.nodes:
            node.perform_ranging(node_dict)
    
    def run_consensus(self, iterations: int = 50) -> np.ndarray:
        """
        Run consensus algorithm using only neighbor measurements
        
        Args:
            iterations: Number of consensus iterations
            
        Returns:
            Array of estimated positions
        """
        # Initialize anchor positions
        anchor_positions = {}
        for idx in self.anchor_indices:
            anchor_positions[idx] = self.nodes[idx].true_position
        
        errors = []
        
        for iteration in range(iterations):
            # Each node updates position based on neighbors
            new_positions = {}
            
            for node in self.nodes:
                if node.node_id in self.anchor_indices:
                    # Anchors don't move
                    new_positions[node.node_id] = node.true_position
                else:
                    # Get current neighbor positions
                    neighbor_positions = {
                        nid: self.nodes[nid].estimated_position 
                        for nid in node.neighbors
                    }
                    
                    # Update position
                    new_pos = node.estimate_position_from_neighbors(
                        neighbor_positions, anchor_positions
                    )
                    new_positions[node.node_id] = new_pos
            
            # Apply updates
            for node_id, pos in new_positions.items():
                self.nodes[node_id].estimated_position = pos
            
            # Calculate error
            total_error = 0
            for node in self.nodes:
                if node.node_id not in self.anchor_indices:
                    error = np.linalg.norm(node.estimated_position - node.true_position)
                    total_error += error**2
            
            rmse = np.sqrt(total_error / (self.num_nodes - len(self.anchor_indices)))
            errors.append(rmse)
            
            if iteration % 10 == 0:
                print(f"  Iteration {iteration}: RMSE = {rmse:.4f}m")
        
        # Return final positions
        positions = np.array([node.estimated_position for node in self.nodes])
        return positions
    
    def get_statistics(self) -> Dict:
        """
        Get network-wide statistics
        
        Returns:
            Dictionary of statistics
        """
        all_stats = [node.get_ranging_statistics() for node in self.nodes]
        
        # Aggregate statistics
        total_ps_links = sum(s['num_ps_links'] for s in all_stats)
        total_ns_links = sum(s['num_ns_links'] for s in all_stats)
        total_links = total_ps_links + total_ns_links
        
        # Position errors
        position_errors = []
        for node in self.nodes:
            if node.node_id not in self.anchor_indices:
                error = np.linalg.norm(node.estimated_position - node.true_position)
                position_errors.append(error)
        
        stats = {
            'num_nodes': self.num_nodes,
            'num_anchors': len(self.anchor_indices),
            'total_links': total_links,
            'ps_links': total_ps_links,
            'ns_links': total_ns_links,
            'ps_ratio': total_ps_links / total_links if total_links > 0 else 0,
            'position_rmse': np.sqrt(np.mean(np.array(position_errors)**2)) if position_errors else 0,
            'position_mean': np.mean(position_errors) if position_errors else 0,
            'position_std': np.std(position_errors) if position_errors else 0,
            'position_max': np.max(position_errors) if position_errors else 0,
            'graph_connected': nx.is_connected(self.graph),
            'average_degree': np.mean([d for n, d in self.graph.degree()])
        }
        
        # Add per-node stats
        if all_stats:
            ps_errors = [s.get('ps_ranging_error_mean', 0) for s in all_stats if 'ps_ranging_error_mean' in s]
            ns_errors = [s.get('ns_ranging_error_mean', 0) for s in all_stats if 'ns_ranging_error_mean' in s]
            
            if ps_errors:
                stats['ps_ranging_error'] = np.mean(ps_errors)
            if ns_errors:
                stats['ns_ranging_error'] = np.mean(ns_errors)
                
        return stats


# Example usage
if __name__ == "__main__":
    print("Testing Nearest-Neighbor FTL")
    print("="*50)
    
    # Create small test network
    np.random.seed(42)
    positions = np.random.uniform(0, 20, (10, 2))
    
    # Make first 3 nodes anchors
    network = NearestNeighborNetwork(positions, anchor_indices=[0, 1, 2])
    
    # Discover topology
    network.discover_topology()
    
    # Perform ranging
    network.perform_all_ranging()
    
    # Run consensus
    network.run_consensus(iterations=30)
    
    # Get statistics
    stats = network.get_statistics()
    
    print("\nFinal Statistics:")
    print(f"  PS-precision links: {stats['ps_links']} ({stats['ps_ratio']:.1%})")
    print(f"  NS-precision links: {stats['ns_links']}")
    if 'ps_ranging_error' in stats:
        print(f"  PS ranging error: {stats['ps_ranging_error']*1000:.1f}mm")
    if 'ns_ranging_error' in stats:
        print(f"  NS ranging error: {stats['ns_ranging_error']:.3f}m")
    print(f"  Position RMSE: {stats['position_rmse']:.3f}m")