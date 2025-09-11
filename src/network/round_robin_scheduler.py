"""
Round-Robin Scheduler for Collision-Free Network Sounding
Implements time-slotted transmission for decentralized ranging
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum


class NodeState(Enum):
    """Node states in round-robin schedule"""
    IDLE = "idle"
    TRANSMITTING = "transmitting"
    RECEIVING = "receiving"
    PROCESSING = "processing"


@dataclass
class FrameStructure:
    """
    Frame structure for S-band transmission
    All times in microseconds
    """
    guard_interval_us: float = 3.0     # Guard interval for multipath
    preamble_us: float = 33.0          # CAZAC/ZC sequence (4095 chips @ 250MS/s)
    pilots_us: float = 10.0            # Multitone pilots for CFO
    lfm_chirp_us: float = 10.0         # Wideband LFM for frequency sync
    data_us: float = 10.0               # Optional telemetry
    
    @property
    def total_frame_us(self) -> float:
        """Total frame duration"""
        return (self.guard_interval_us + self.preamble_us + 
                self.pilots_us + self.lfm_chirp_us + self.data_us)
    
    @property
    def slot_duration_us(self) -> float:
        """Total slot duration including processing margin"""
        return self.total_frame_us + 10.0  # Add 10μs processing margin


@dataclass
class ScheduleSlot:
    """Single slot in the round-robin schedule"""
    slot_number: int
    transmitter_id: int
    start_time_us: float
    duration_us: float
    receivers: List[int]  # List of receiving node IDs


class RoundRobinScheduler:
    """
    Manages collision-free round-robin transmission schedule
    Ensures each node gets dedicated TX slot while others listen
    """
    
    def __init__(self, 
                 node_ids: List[int],
                 frame_structure: Optional[FrameStructure] = None,
                 epoch_rate_hz: float = 10.0):
        """
        Initialize round-robin scheduler
        
        Args:
            node_ids: List of participating node IDs
            frame_structure: Frame timing parameters
            epoch_rate_hz: How often to complete full network sounding
        """
        self.node_ids = sorted(node_ids)
        self.num_nodes = len(node_ids)
        self.frame = frame_structure or FrameStructure()
        self.epoch_rate_hz = epoch_rate_hz
        
        # Build schedule
        self.schedule = self._build_schedule()
        
        # Node states
        self.node_states = {nid: NodeState.IDLE for nid in node_ids}
        
        # Timing
        self.epoch_duration_us = self.frame.slot_duration_us * self.num_nodes
        self.current_time_us = 0.0
        self.current_slot = 0
        
        # Statistics
        self.transmission_count = {nid: 0 for nid in node_ids}
        self.reception_count = {nid: {other: 0 for other in node_ids if other != nid} 
                               for nid in node_ids}
        
    def _build_schedule(self) -> List[ScheduleSlot]:
        """
        Build complete round-robin schedule for one epoch
        
        Returns:
            List of schedule slots
        """
        schedule = []
        
        for slot_num, tx_node in enumerate(self.node_ids):
            # All other nodes are receivers
            rx_nodes = [nid for nid in self.node_ids if nid != tx_node]
            
            slot = ScheduleSlot(
                slot_number=slot_num,
                transmitter_id=tx_node,
                start_time_us=slot_num * self.frame.slot_duration_us,
                duration_us=self.frame.slot_duration_us,
                receivers=rx_nodes
            )
            schedule.append(slot)
            
        return schedule
    
    def get_current_slot(self, time_us: float) -> Optional[ScheduleSlot]:
        """
        Get the active slot at given time
        
        Args:
            time_us: Current time in microseconds
            
        Returns:
            Active schedule slot or None if in inter-epoch gap
        """
        # Find position within epoch
        epoch_time = time_us % (self.epoch_duration_us * 1.1)  # 10% inter-epoch gap
        
        if epoch_time >= self.epoch_duration_us:
            # In inter-epoch gap
            return None
            
        # Find active slot
        slot_index = int(epoch_time / self.frame.slot_duration_us)
        
        if slot_index < len(self.schedule):
            return self.schedule[slot_index]
        
        return None
    
    def get_node_action(self, node_id: int, time_us: float) -> Tuple[NodeState, Optional[Dict]]:
        """
        Determine what a node should be doing at given time
        
        Args:
            node_id: Node ID to query
            time_us: Current time in microseconds
            
        Returns:
            (node_state, action_params)
        """
        slot = self.get_current_slot(time_us)
        
        if slot is None:
            # Inter-epoch gap - process accumulated data
            return NodeState.PROCESSING, None
            
        # Time within slot
        slot_time = time_us - slot.start_time_us
        
        if node_id == slot.transmitter_id:
            # This node is transmitting
            if slot_time < self.frame.guard_interval_us:
                return NodeState.IDLE, None  # Guard interval
            else:
                return NodeState.TRANSMITTING, {
                    'slot': slot,
                    'frame_phase': self._get_frame_phase(slot_time)
                }
        elif node_id in slot.receivers:
            # This node is receiving
            return NodeState.RECEIVING, {
                'slot': slot,
                'transmitter': slot.transmitter_id,
                'expected_signal': self._get_frame_phase(slot_time)
            }
        else:
            return NodeState.IDLE, None
    
    def _get_frame_phase(self, slot_time_us: float) -> str:
        """
        Determine which part of frame is active
        
        Args:
            slot_time_us: Time within current slot
            
        Returns:
            Frame phase name
        """
        t = slot_time_us
        
        if t < self.frame.guard_interval_us:
            return "guard"
        
        t -= self.frame.guard_interval_us
        
        if t < self.frame.preamble_us:
            return "preamble"
        
        t -= self.frame.preamble_us
        
        if t < self.frame.pilots_us:
            return "pilots"
            
        t -= self.frame.pilots_us
        
        if t < self.frame.lfm_chirp_us:
            return "lfm_chirp"
            
        t -= self.frame.lfm_chirp_us
        
        if t < self.frame.data_us:
            return "data"
            
        return "processing"
    
    def advance_time(self, delta_us: float) -> List[Dict]:
        """
        Advance scheduler time and return any events
        
        Args:
            delta_us: Time increment in microseconds
            
        Returns:
            List of schedule events that occurred
        """
        events = []
        
        old_slot = self.get_current_slot(self.current_time_us)
        self.current_time_us += delta_us
        new_slot = self.get_current_slot(self.current_time_us)
        
        # Check for slot transition
        if old_slot != new_slot:
            if old_slot is not None:
                events.append({
                    'type': 'slot_end',
                    'slot': old_slot,
                    'time': self.current_time_us
                })
                
            if new_slot is not None:
                events.append({
                    'type': 'slot_start',
                    'slot': new_slot,
                    'time': self.current_time_us
                })
                
                # Update statistics
                self.transmission_count[new_slot.transmitter_id] += 1
                for rx_id in new_slot.receivers:
                    self.reception_count[rx_id][new_slot.transmitter_id] += 1
                    
        # Check for epoch completion
        if self.current_time_us % (self.epoch_duration_us * 1.1) < delta_us:
            events.append({
                'type': 'epoch_complete',
                'time': self.current_time_us,
                'stats': self.get_statistics()
            })
            
        return events
    
    def get_statistics(self) -> Dict:
        """
        Get scheduler statistics
        
        Returns:
            Dictionary of scheduling statistics
        """
        total_tx = sum(self.transmission_count.values())
        total_rx = sum(sum(rx_dict.values()) for rx_dict in self.reception_count.values())
        
        return {
            'epoch_duration_us': self.epoch_duration_us,
            'slot_duration_us': self.frame.slot_duration_us,
            'num_slots': self.num_nodes,
            'total_transmissions': total_tx,
            'total_receptions': total_rx,
            'tx_per_node': self.transmission_count,
            'rx_per_node': self.reception_count,
            'efficiency': total_rx / (self.num_nodes * (self.num_nodes - 1)) if self.num_nodes > 1 else 1.0
        }
    
    def get_link_schedule(self, node_i: int, node_j: int) -> List[float]:
        """
        Get transmission times for specific link
        
        Args:
            node_i: Transmitter node ID
            node_j: Receiver node ID
            
        Returns:
            List of transmission times in current epoch
        """
        times = []
        
        for slot in self.schedule:
            if slot.transmitter_id == node_i and node_j in slot.receivers:
                times.append(slot.start_time_us)
                
        return times
    
    def optimize_schedule(self, link_quality: Dict[Tuple[int, int], float]) -> List[ScheduleSlot]:
        """
        Optimize schedule based on link quality metrics
        Prioritize weak links with more slots
        
        Args:
            link_quality: Dictionary of (tx, rx) -> quality scores
            
        Returns:
            Optimized schedule
        """
        # Sort nodes by average link quality (ascending - worst first)
        node_scores = {}
        
        for node in self.node_ids:
            # Average quality of links from this node
            outgoing_quality = [link_quality.get((node, other), 1.0) 
                               for other in self.node_ids if other != node]
            node_scores[node] = np.mean(outgoing_quality) if outgoing_quality else 1.0
            
        # Reorder schedule - nodes with poor links get earlier slots
        sorted_nodes = sorted(self.node_ids, key=lambda n: node_scores[n])
        
        optimized_schedule = []
        for slot_num, tx_node in enumerate(sorted_nodes):
            rx_nodes = [nid for nid in self.node_ids if nid != tx_node]
            
            slot = ScheduleSlot(
                slot_number=slot_num,
                transmitter_id=tx_node,
                start_time_us=slot_num * self.frame.slot_duration_us,
                duration_us=self.frame.slot_duration_us,
                receivers=rx_nodes
            )
            optimized_schedule.append(slot)
            
        self.schedule = optimized_schedule
        return optimized_schedule


class AdaptiveScheduler(RoundRobinScheduler):
    """
    Adaptive scheduler that adjusts based on network conditions
    """
    
    def __init__(self, node_ids: List[int], **kwargs):
        super().__init__(node_ids, **kwargs)
        self.link_success_rate = {}
        self.adaptive_slots = {}
        
    def update_link_success(self, tx_id: int, rx_id: int, success: bool):
        """
        Update link success rate for adaptation
        
        Args:
            tx_id: Transmitter node ID
            rx_id: Receiver node ID  
            success: Whether ranging was successful
        """
        link = (tx_id, rx_id)
        
        if link not in self.link_success_rate:
            self.link_success_rate[link] = []
            
        self.link_success_rate[link].append(1.0 if success else 0.0)
        
        # Keep only recent history
        if len(self.link_success_rate[link]) > 100:
            self.link_success_rate[link].pop(0)
            
    def adapt_frame_parameters(self, multipath_severity: float):
        """
        Adapt frame parameters based on channel conditions
        
        Args:
            multipath_severity: Estimated multipath severity (0-1)
        """
        if multipath_severity > 0.7:
            # Severe multipath - increase guard interval
            self.frame.guard_interval_us = 10.0
            self.frame.preamble_us = 50.0  # Longer preamble for better estimation
        elif multipath_severity > 0.3:
            # Moderate multipath
            self.frame.guard_interval_us = 5.0
            self.frame.preamble_us = 33.0
        else:
            # Low multipath - can use shorter intervals
            self.frame.guard_interval_us = 2.0
            self.frame.preamble_us = 20.0
            
        # Rebuild schedule with new frame parameters
        self.schedule = self._build_schedule()
        self.epoch_duration_us = self.frame.slot_duration_us * self.num_nodes


class DistributedScheduleCoordinator:
    """
    Coordinates distributed agreement on schedule without central authority
    """
    
    def __init__(self, node_id: int, initial_node_list: List[int]):
        """
        Initialize distributed coordinator
        
        Args:
            node_id: This node's ID
            initial_node_list: Initial known nodes
        """
        self.node_id = node_id
        self.known_nodes = set(initial_node_list)
        self.scheduler = None
        self.schedule_version = 0
        self.consensus_achieved = False
        
    def propose_schedule_update(self, new_nodes: List[int]) -> Dict:
        """
        Propose schedule update for new nodes joining
        
        Args:
            new_nodes: New nodes to add
            
        Returns:
            Schedule proposal message
        """
        proposed_nodes = sorted(list(self.known_nodes.union(set(new_nodes))))
        
        return {
            'type': 'schedule_proposal',
            'proposer': self.node_id,
            'version': self.schedule_version + 1,
            'node_list': proposed_nodes,
            'timestamp': self._get_timestamp()
        }
    
    def handle_schedule_proposal(self, proposal: Dict) -> bool:
        """
        Handle schedule proposal from another node
        
        Args:
            proposal: Schedule proposal message
            
        Returns:
            Whether to accept the proposal
        """
        # Simple rule: accept if newer version or same version from lower ID
        if proposal['version'] > self.schedule_version:
            return True
        elif proposal['version'] == self.schedule_version:
            return proposal['proposer'] < self.node_id
        return False
    
    def achieve_consensus(self, proposals: List[Dict]) -> bool:
        """
        Achieve consensus on schedule from multiple proposals
        
        Args:
            proposals: List of schedule proposals
            
        Returns:
            Whether consensus was achieved
        """
        if not proposals:
            return False
            
        # Sort by version then proposer ID
        proposals.sort(key=lambda p: (p['version'], p['proposer']))
        
        # Take highest version (tie-break by lowest proposer ID)
        accepted = proposals[-1]
        
        # Update local schedule
        self.known_nodes = set(accepted['node_list'])
        self.schedule_version = accepted['version']
        self.scheduler = RoundRobinScheduler(list(self.known_nodes))
        self.consensus_achieved = True
        
        return True
    
    def _get_timestamp(self) -> float:
        """Get current timestamp in microseconds"""
        import time
        return time.time() * 1e6


# Example usage
if __name__ == "__main__":
    # Create scheduler for 8-node network
    node_ids = list(range(8))
    scheduler = RoundRobinScheduler(node_ids, epoch_rate_hz=10)
    
    print(f"Round-Robin Schedule for {len(node_ids)} nodes")
    print(f"Epoch duration: {scheduler.epoch_duration_us:.1f} μs")
    print(f"Slot duration: {scheduler.frame.slot_duration_us:.1f} μs")
    print(f"Frame breakdown:")
    print(f"  Guard: {scheduler.frame.guard_interval_us} μs")
    print(f"  Preamble: {scheduler.frame.preamble_us} μs")
    print(f"  Pilots: {scheduler.frame.pilots_us} μs")
    print(f"  LFM: {scheduler.frame.lfm_chirp_us} μs")
    print(f"  Data: {scheduler.frame.data_us} μs")
    print()
    
    # Show schedule
    print("Transmission Schedule:")
    for slot in scheduler.schedule:
        print(f"  Slot {slot.slot_number}: Node {slot.transmitter_id} TX @ {slot.start_time_us:.1f} μs")
        print(f"    Receivers: {slot.receivers}")
    print()
    
    # Simulate one epoch
    print("Simulating one epoch...")
    time_us = 0
    step_us = 10
    
    while time_us < scheduler.epoch_duration_us * 1.1:
        events = scheduler.advance_time(step_us)
        
        for event in events:
            if event['type'] == 'slot_start':
                print(f"  t={event['time']:.1f}: Slot {event['slot'].slot_number} starts "
                     f"(Node {event['slot'].transmitter_id} TX)")
            elif event['type'] == 'epoch_complete':
                print(f"  t={event['time']:.1f}: Epoch complete")
                stats = event['stats']
                print(f"    Total TX: {stats['total_transmissions']}")
                print(f"    Total RX: {stats['total_receptions']}")
                print(f"    Efficiency: {stats['efficiency']:.1%}")
                
        time_us += step_us