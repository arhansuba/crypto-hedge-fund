# src/cognition/memory.py
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import logging
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class MemoryEntry:
    """Single memory entry."""
    timestamp: str
    type: str  # 'trade', 'analysis', 'error', 'learning'
    data: Dict[str, Any]
    importance: float  # 0 to 1
    metadata: Optional[Dict] = None

class MemorySystem:
    """Advanced memory system for autonomous trading agent."""
    
    def __init__(
        self,
        max_size: int = 1000,
        importance_threshold: float = 0.5,
        consolidation_interval: int = 100
    ):
        self.max_size = max_size
        self.importance_threshold = importance_threshold
        self.consolidation_interval = consolidation_interval
        
        # Memory storage
        self.short_term = deque(maxlen=100)  # Recent memories
        self.long_term = deque(maxlen=max_size)  # Important memories
        
        # Performance metrics
        self.metrics = {
            'trades': [],
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'total_trades': 0
        }
        
    async def add(self, entry_type: str, data: Dict[str, Any], metadata: Optional[Dict] = None) -> None:
        """Add new memory entry."""
        try:
            # Calculate importance
            importance = self._calculate_importance(entry_type, data)
            
            entry = MemoryEntry(
                timestamp=datetime.now().isoformat(),
                type=entry_type,
                data=data,
                importance=importance,
                metadata=metadata
            )
            
            # Add to short-term memory
            self.short_term.append(entry)
            
            # Add to long-term if important
            if importance >= self.importance_threshold:
                self.long_term.append(entry)
            
            # Update metrics if trade memory
            if entry_type == 'trade':
                self._update_metrics(data)
            
            # Consolidate if needed
            if len(self.short_term) >= self.consolidation_interval:
                await self._consolidate_memories()
                
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            
    def get_relevant_memories(
        self,
        context: Dict[str, Any],
        limit: int = 5,
        memory_types: Optional[List[str]] = None
    ) -> List[MemoryEntry]:
        """Get memories relevant to current context."""
        scored_memories = []
        
        # Search both memory stores
        memories = list(self.short_term) + list(self.long_term)
        
        for memory in memories:
            if memory_types and memory.type not in memory_types:
                continue
                
            relevance = self._calculate_relevance(memory, context)
            scored_memories.append((relevance, memory))
            
        # Sort by relevance and deduplicate
        sorted_memories = sorted(scored_memories, key=lambda x: x[0], reverse=True)
        unique_memories = []
        seen = set()
        
        for _, memory in sorted_memories:
            memory_key = f"{memory.timestamp}_{memory.type}"
            if memory_key not in seen:
                unique_memories.append(memory)
                seen.add(memory_key)
                if len(unique_memories) >= limit:
                    break
                    
        return unique_memories
        
    def get_recent_memories(self, limit: int = 10) -> List[MemoryEntry]:
        """Get most recent memories."""
        return list(self.short_term)[-limit:]
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return self.metrics
        
    def _calculate_importance(self, entry_type: str, data: Dict[str, Any]) -> float:
        """Calculate memory importance score."""
        importance = 0.5  # Base importance
        
        if entry_type == 'trade':
            # Important if large trade or significant profit/loss
            trade_size = abs(float(data.get('size', 0)))
            profit = float(data.get('profit', 0))
            importance += min(0.3, trade_size / 10000)  # Size factor
            importance += min(0.2, abs(profit) / 1000)  # Profit factor
            
        elif entry_type == 'error':
            # Errors are usually important
            importance += 0.3
            
        elif entry_type == 'analysis':
            # Important if high confidence or risk
            confidence = float(data.get('confidence', 0))
            risk_score = float(data.get('risk_score', 0))
            importance += 0.2 * confidence
            importance += 0.2 * risk_score
            
        return min(1.0, importance)
        
    def _calculate_relevance(self, memory: MemoryEntry, context: Dict[str, Any]) -> float:
        """Calculate memory relevance to current context."""
        relevance = 0.0
        
        # Time relevance (more recent = more relevant)
        time_diff = (datetime.now() - datetime.fromisoformat(memory.timestamp)).total_seconds()
        time_factor = max(0, 1 - (time_diff / (24 * 3600)))  # 24 hour scale
        relevance += 0.3 * time_factor
        
        # Context matching
        if 'token' in context and context['token'] == memory.data.get('token'):
            relevance += 0.3
            
        if 'type' in context and context['type'] == memory.type:
            relevance += 0.2
            
        # Market condition similarity
        if 'market_conditions' in context and 'market_conditions' in memory.data:
            condition_match = self._compare_market_conditions(
                context['market_conditions'],
                memory.data['market_conditions']
            )
            relevance += 0.2 * condition_match
            
        return min(1.0, relevance)
        
    async def _consolidate_memories(self):
        """Consolidate and clean up memories."""
        try:
            # Group similar memories
            consolidated = []
            groups = {}
            
            for memory in self.short_term:
                key = f"{memory.type}_{memory.data.get('token', '')}"
                if key not in groups:
                    groups[key] = []
                groups[key].append(memory)
                
            # Create consolidated memories
            for group in groups.values():
                if len(group) > 1:
                    consolidated_data = self._merge_memory_data(group)
                    importance = max(m.importance for m in group)
                    
                    consolidated.append(MemoryEntry(
                        timestamp=datetime.now().isoformat(),
                        type=group[0].type,
                        data=consolidated_data,
                        importance=importance
                    ))
                    
            # Update memory stores
            self.short_term.clear()
            self.short_term.extend(consolidated)
            
        except Exception as e:
            logger.error(f"Error consolidating memories: {e}")
            
    def _merge_memory_data(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Merge similar memory data."""
        merged = {}
        for memory in memories:
            for key, value in memory.data.items():
                if key not in merged:
                    merged[key] = []
                if isinstance(value, (int, float)):
                    merged[key].append(value)
                else:
                    merged[key] = value
                    
        # Average numerical values
        for key, value in merged.items():
            if isinstance(value, list) and value and isinstance(value[0], (int, float)):
                merged[key] = sum(value) / len(value)
                
        return merged
        
    def _update_metrics(self, trade_data: Dict[str, Any]):
        """Update performance metrics."""
        self.metrics['total_trades'] += 1
        self.metrics['trades'].append(trade_data)
        
        # Calculate win rate
        if 'profit' in trade_data:
            profit = float(trade_data['profit'])
            if profit > 0:
                self.metrics['win_rate'] = (
                    (self.metrics['win_rate'] * (self.metrics['total_trades'] - 1) + 1) /
                    self.metrics['total_trades']
                )
            else:
                self.metrics['win_rate'] = (
                    self.metrics['win_rate'] * (self.metrics['total_trades'] - 1) /
                    self.metrics['total_trades']
                )
                
        # Update average profit
        profits = [float(t.get('profit', 0)) for t in self.metrics['trades']]
        self.metrics['avg_profit'] = sum(profits) / len(profits) if profits else 0
        
    @staticmethod
    def _compare_market_conditions(cond1: Dict[str, Any], cond2: Dict[str, Any]) -> float:
        """Compare similarity of market conditions."""
        try:
            # Compare key metrics
            metrics = ['trend', 'volatility', 'volume', 'sentiment']
            matches = sum(
                1 for m in metrics
                if m in cond1 and m in cond2 and cond1[m] == cond2[m]
            )
            return matches / len(metrics)
        except Exception:
            return 0.0