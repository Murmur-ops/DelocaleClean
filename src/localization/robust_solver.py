"""
Robust Distributed Localization Solver
Implements Levenberg-Marquardt with Huber loss for NLOS robustness
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MeasurementEdge:
    """Represents a ranging measurement between two nodes"""
    node_i: int
    node_j: int
    distance: float
    quality: float  # 0-1 quality score
    variance: float  # Measurement variance
    multipath_severity: float = 0.0  # 0=no multipath, 1=severe
    whitening_applied: bool = False  # Whether whitening was applied
    residual_bias: float = 0.0  # Expected bias after processing


class RobustLocalizer:
    """Robust localization using Levenberg-Marquardt with Huber loss"""
    
    def __init__(self, dimension: int = 2, huber_delta: float = 1.0):
        """
        Initialize robust localizer
        
        Args:
            dimension: 2D or 3D localization
            huber_delta: Threshold for Huber loss function
        """
        self.d = dimension
        self.huber_delta = huber_delta
        self.lambda_lm = 0.01  # Levenberg-Marquardt damping parameter
        self.max_iterations = 100
        self.convergence_threshold = 1e-6
        self.epsilon = 1e-10  # For numerical stability
        self.adaptive_huber = True  # Adapt Huber delta based on multipath
        
    def huber_loss(self, residual: float) -> float:
        """
        Huber loss function for robust estimation
        
        Args:
            residual: Error value
            
        Returns:
            Huber loss value
        """
        abs_r = abs(residual)
        if abs_r <= self.huber_delta:
            return 0.5 * residual**2
        else:
            return self.huber_delta * (abs_r - 0.5 * self.huber_delta)
    
    def huber_weight(self, residual: float) -> float:
        """
        Weight function for Huber loss (derivative of rho(r)/r)
        
        Args:
            residual: Error value
            
        Returns:
            Weight for this residual
        """
        abs_r = abs(residual)
        if abs_r <= self.huber_delta:
            return 1.0
        else:
            return self.huber_delta / abs_r if abs_r > 0 else 1.0
    
    def adapt_huber_delta(self, measurements: List[MeasurementEdge]) -> float:
        """
        Adapt Huber delta based on multipath severity and whitening status
        
        Args:
            measurements: List of measurements with multipath info
            
        Returns:
            Adapted Huber delta
        """
        if not self.adaptive_huber:
            return self.huber_delta
            
        # Compute average multipath severity
        multipath_severities = [m.multipath_severity for m in measurements]
        avg_severity = np.mean(multipath_severities) if multipath_severities else 0.0
        
        # Count whitened measurements
        whitened_count = sum(1 for m in measurements if m.whitening_applied)
        whitening_ratio = whitened_count / len(measurements) if measurements else 0.0
        
        # Adapt Huber delta
        if whitening_ratio > 0.8:
            # Most measurements whitened - can be more aggressive
            adapted_delta = self.huber_delta * (0.1 + 0.9 * (1 - avg_severity))
        elif whitening_ratio > 0.5:
            # Partial whitening - moderate approach
            adapted_delta = self.huber_delta * (0.3 + 0.7 * (1 - avg_severity))
        else:
            # Little/no whitening - conservative approach
            adapted_delta = self.huber_delta * (0.5 + 0.5 * (1 - avg_severity))
            
        return adapted_delta
    
    def compute_measurement_weight(self, edge: MeasurementEdge) -> float:
        """
        Compute weight for measurement based on quality and multipath
        
        Args:
            edge: Measurement edge with metadata
            
        Returns:
            Weight for this measurement
        """
        # Base weight from quality score
        base_weight = edge.quality
        
        # Adjust for multipath
        if edge.whitening_applied:
            # Whitening applied - trust more
            multipath_factor = 1.0 - 0.3 * edge.multipath_severity
        else:
            # No whitening - trust less with multipath
            multipath_factor = 1.0 - 0.7 * edge.multipath_severity
            
        # Account for expected residual bias
        bias_factor = 1.0 / (1.0 + abs(edge.residual_bias))
        
        # Combined weight
        weight = base_weight * multipath_factor * bias_factor
        
        return max(weight, 0.1)  # Minimum weight to avoid numerical issues
    
    def compute_residuals(self, positions: np.ndarray, 
                         measurements: List[MeasurementEdge],
                         anchor_positions: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Compute residuals between estimated and measured distances
        
        Args:
            positions: Current position estimates for unknown nodes
            measurements: List of range measurements
            anchor_positions: Known anchor positions
            
        Returns:
            Array of residuals
        """
        residuals = []
        
        for edge in measurements:
            # Get positions
            if edge.node_i in anchor_positions:
                pos_i = anchor_positions[edge.node_i]
            else:
                # Map unknown node ID to position index
                # Assumes unknown nodes are numbered starting from max(anchor_ids)+1
                unknown_idx = edge.node_i - len(anchor_positions)
                if unknown_idx >= 0 and unknown_idx * self.d + self.d <= len(positions):
                    pos_i = positions[unknown_idx * self.d:(unknown_idx + 1) * self.d]
                else:
                    # Fallback for single unknown
                    pos_i = positions[:self.d] if len(positions) >= self.d else positions
            
            if edge.node_j in anchor_positions:
                pos_j = anchor_positions[edge.node_j]
            else:
                # Map unknown node ID to position index
                unknown_idx = edge.node_j - len(anchor_positions)
                if unknown_idx >= 0 and unknown_idx * self.d + self.d <= len(positions):
                    pos_j = positions[unknown_idx * self.d:(unknown_idx + 1) * self.d]
                else:
                    # Fallback for single unknown
                    pos_j = positions[:self.d] if len(positions) >= self.d else positions
            
            # Compute estimated distance
            est_distance = np.linalg.norm(pos_i - pos_j)
            
            # Handle zero-distance edge case
            if est_distance < self.epsilon:
                continue
            
            # Weighted residual
            weight = np.sqrt(edge.quality / edge.variance)
            residual = weight * (est_distance - edge.distance)
            residuals.append(residual)
        
        return np.array(residuals)
    
    def compute_jacobian(self, positions: np.ndarray,
                         measurements: List[MeasurementEdge],
                         anchor_positions: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Compute Jacobian matrix for Levenberg-Marquardt
        
        Args:
            positions: Current position estimates
            measurements: List of range measurements
            anchor_positions: Known anchor positions
            
        Returns:
            Jacobian matrix
        """
        n_unknowns = len(positions) // self.d
        n_measurements = len(measurements)
        J = np.zeros((n_measurements, len(positions)))
        
        for k, edge in enumerate(measurements):
            # Get positions
            if edge.node_i in anchor_positions:
                pos_i = anchor_positions[edge.node_i]
                idx_i = None
            else:
                # Map unknown node ID to position index
                idx_i = edge.node_i - len(anchor_positions)
                if idx_i >= 0 and idx_i * self.d + self.d <= len(positions):
                    pos_i = positions[idx_i * self.d:(idx_i + 1) * self.d]
                else:
                    idx_i = 0
                    pos_i = positions[:self.d] if len(positions) >= self.d else positions
            
            if edge.node_j in anchor_positions:
                pos_j = anchor_positions[edge.node_j]
                idx_j = None
            else:
                # Map unknown node ID to position index
                idx_j = edge.node_j - len(anchor_positions)
                if idx_j >= 0 and idx_j * self.d + self.d <= len(positions):
                    pos_j = positions[idx_j * self.d:(idx_j + 1) * self.d]
                else:
                    idx_j = 0
                    pos_j = positions[:self.d] if len(positions) >= self.d else positions
            
            # Compute gradient
            diff = pos_i - pos_j
            dist = np.linalg.norm(diff)
            
            # Handle zero-distance edge case with epsilon
            if dist > self.epsilon:
                gradient = diff / dist
                weight = np.sqrt(edge.quality / edge.variance)
                
                # Set Jacobian entries
                if idx_i is not None:
                    J[k, idx_i * self.d:(idx_i + 1) * self.d] = weight * gradient
                if idx_j is not None:
                    J[k, idx_j * self.d:(idx_j + 1) * self.d] = -weight * gradient
        
        return J
    
    def solve(self, initial_positions: np.ndarray,
              measurements: List[MeasurementEdge],
              anchor_positions: Dict[int, np.ndarray],
              unknown_node_ids: List[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Solve localization problem using robust Levenberg-Marquardt
        
        Args:
            initial_positions: Initial guess for unknown node positions
            measurements: List of range measurements
            anchor_positions: Known anchor positions
            
        Returns:
            Optimized positions and convergence info
        """
        positions = initial_positions.copy()
        
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            # Compute residuals
            residuals = self.compute_residuals(positions, measurements, anchor_positions)
            
            # Apply Huber weighting
            weights = np.array([self.huber_weight(r) for r in residuals])
            weighted_residuals = weights * residuals
            
            # Compute cost
            cost = sum([self.huber_loss(r) for r in residuals])
            convergence_history.append(cost)
            
            # Check convergence
            if iteration > 0 and len(convergence_history) >= 2:
                if abs(float(convergence_history[-1]) - float(convergence_history[-2])) < float(self.convergence_threshold):
                    break
            
            # Compute Jacobian
            J = self.compute_jacobian(positions, measurements, anchor_positions)
            
            # Apply Huber weights to Jacobian
            W = np.diag(weights)
            J_weighted = W @ J
            
            # Levenberg-Marquardt update
            H = J_weighted.T @ J_weighted
            g = J_weighted.T @ weighted_residuals
            
            # Add damping
            H_damped = H + self.lambda_lm * np.eye(len(positions))
            
            try:
                # Solve for update
                delta = -np.linalg.solve(H_damped, g)
                
                # Update positions
                new_positions = positions + delta
                
                # Compute new cost
                new_residuals = self.compute_residuals(new_positions, measurements, anchor_positions)
                new_cost = sum([self.huber_loss(r) for r in new_residuals])
                
                # Accept or reject update
                if new_cost < cost:
                    positions = new_positions
                    self.lambda_lm *= 0.9  # Decrease damping
                else:
                    self.lambda_lm *= 10  # Increase damping
                    
            except np.linalg.LinAlgError:
                # Singular matrix, increase damping
                self.lambda_lm *= 10
                continue
        
        # Reshape positions to node format
        n_unknowns = len(positions) // self.d
        node_positions = {}
        for i in range(n_unknowns):
            node_positions[i + len(anchor_positions) + 1] = positions[i * self.d:(i + 1) * self.d]
        
        return positions, {
            'iterations': iteration + 1,
            'final_cost': convergence_history[-1] if convergence_history else float('inf'),
            'convergence_history': convergence_history,
            'converged': iteration < self.max_iterations - 1
        }


class DistributedLocalizer:
    """Distributed version using message passing between neighbors"""
    
    def __init__(self, node_id: int, neighbors: List[int], dimension: int = 2):
        """
        Initialize distributed localizer for a single node
        
        Args:
            node_id: This node's ID
            neighbors: List of neighbor node IDs
            dimension: 2D or 3D localization
        """
        self.node_id = node_id
        self.neighbors = neighbors
        self.d = dimension
        # Ensure float64 for numerical precision
        self.position = np.zeros(dimension, dtype=np.float64)
        self.gradient = np.zeros(dimension, dtype=np.float64)
        self.dual_variables = {n: 0.0 for n in neighbors}
        self.rho = 1.0  # ADMM penalty parameter
        
    def update_position(self, measurements: Dict[int, float],
                       neighbor_positions: Dict[int, np.ndarray],
                       qualities: Dict[int, float]) -> np.ndarray:
        """
        Update position based on local measurements and neighbor info
        
        Args:
            measurements: Distance measurements to neighbors
            neighbor_positions: Current position estimates of neighbors
            qualities: Quality scores for each measurement
            
        Returns:
            Updated position estimate
        """
        # Weighted least squares update
        A = np.zeros((self.d, self.d))
        b = np.zeros(self.d)
        
        for neighbor_id in self.neighbors:
            if neighbor_id in measurements and neighbor_id in neighbor_positions:
                # Get measurement and neighbor position
                dist = measurements[neighbor_id]
                neighbor_pos = neighbor_positions[neighbor_id]
                quality = qualities.get(neighbor_id, 0.5)
                
                # Compute gradient
                diff = self.position - neighbor_pos
                curr_dist = np.linalg.norm(diff)
                
                # Handle zero-distance edge case
                if curr_dist > 1e-10:
                    # Weighted update
                    weight = quality
                    gradient = diff / curr_dist
                    
                    # Add to normal equations
                    A += weight * np.outer(gradient, gradient)
                    b += weight * gradient * (curr_dist - dist)
        
        # Add ADMM consensus term (once, not per neighbor!)
        A += self.rho * np.eye(self.d)
        for neighbor_id in self.neighbors:
            if neighbor_id in neighbor_positions:
                b += self.rho * (neighbor_positions[neighbor_id] - self.dual_variables[neighbor_id])
        
        # Solve for position update
        try:
            if np.linalg.det(A) > 1e-6:
                delta = np.linalg.solve(A, b)
                self.position -= 0.5 * delta  # Damped update
        except np.linalg.LinAlgError:
            pass  # Keep current position if singular
        
        return self.position
    
    def update_dual_variables(self, neighbor_positions: Dict[int, np.ndarray]):
        """
        Update dual variables for ADMM consensus
        
        Args:
            neighbor_positions: Current position estimates of neighbors
        """
        for neighbor_id in self.neighbors:
            if neighbor_id in neighbor_positions:
                # Update dual variable
                self.dual_variables[neighbor_id] += self.rho * (
                    self.position - neighbor_positions[neighbor_id]
                )