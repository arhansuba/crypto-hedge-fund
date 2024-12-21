# src/agents/base.py
from typing import Dict, List, Optional
from datetime import datetime
import logging
from llm_client import GaiaLLM  # Ensure GaiaLLM is imported

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base agent with core capabilities."""
    
    def __init__(
        self,
        llm_config: Optional[Dict] = None,
        memory_size: int = 1000,
        objectives: List[str] = None
    ):
        """Initialize base agent.
        
        Args:
            llm_config: Configuration for LLM. If None, uses defaults.
        """
        self.llm_config = llm_config or {}
        self.llm = GaiaLLM()  # Initialize GaiaLLM without arguments
        self.memory = MemoryState(size=memory_size)
        self.objectives = objectives or []
        self.last_thought = None
        
    async def think(self, context: Dict) -> Dict:
        """Core thinking process."""
        try:
            # Format messages for LLM
            messages = [
                {
                    "role": "system",
                    "content": """You are an expert crypto trading AI assistant. 
                    Analyze market data and provide clear, actionable insights focused on:
                    - Technical analysis
                    - Risk assessment
                    - Market sentiment
                    - Trading opportunities"""
                },
                {
                    "role": "user",
                    "content": f"Analyze the following market context and provide insights:\n{context}"
                }
            ]
            
            # Get LLM response
            response = await self.llm.chat_completion(
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            try:
                thought = response["choices"][0]["message"]["content"]
                return {
                    "thought": thought,
                    "timestamp": datetime.now().isoformat()
                }
            except (KeyError, IndexError) as e:
                logger.error(f"Error parsing LLM response: {e}")
                return {
                    "error": "Failed to parse LLM response",
                    "raw_response": response,
                    "timestamp": datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error in thinking process: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    async def close(self):
        """Cleanup resources."""
        if hasattr(self, 'llm'):
            await self.llm.close()

    async def learn(self, experience: Dict):
        """Learn from experience and update memory."""
        await self.memory.add({
            'timestamp': datetime.now().isoformat(),
            'type': 'experience',
            'data': experience
        })
        
        # Analyze experience for learning
        analysis = await self.think({
            'type': 'learning',
            'experience': experience
        })
        
        # Update objectives if needed
        if analysis.get('update_objectives'):
            self.objectives = self._update_objectives(analysis['update_objectives'])
            
    def _update_objectives(self, updates: List[str]) -> List[str]:
        """Update agent objectives based on learning."""
        current = set(self.objectives)
        new = set(updates)
        
        # Keep important objectives, add new ones
        return list(current.union(new))[:5]  # Keep top 5 objectives

class LLM:
    """Language Model interface for generating responses."""
    def __init__(self, model: str, api_key: str, temperature: float = 0.7):
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

    async def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        # Simulate an API call to a language model
        # Replace this with actual API call logic
        return "Simulated response based on the prompt."

class MemoryState:
    """Memory state management for the agent."""
    def __init__(self, size: int = 1000):
        self.size = size
        self.memory = []

    async def add(self, entry: Dict):
        """Add a new entry to memory."""
        if len(self.memory) >= self.size:
            self.memory.pop(0)  # Remove the oldest entry if memory is full
        self.memory.append(entry)

    def get_recent(self, n: int = 5) -> List[Dict]:
        """Get the most recent n entries from memory."""
        return self.memory[-n:]