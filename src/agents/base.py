# src/agents/base.py
from typing import Dict, List
from datetime import datetime
import logging


logger = logging.getLogger(__name__)

class BaseAgent:
    """Base autonomous agent with core AGI capabilities."""
    
    def __init__(
        self,
        llm_config: Dict,
        memory_size: int = 1000,
        objectives: List[str] = None
    ):
        self.llm = LLM(**llm_config)
        self.memory = MemoryState(size=memory_size)
        self.objectives = objectives or []
        self.last_thought = None
        
    async def think(self, context: Dict) -> Dict:
        """Core thinking process combining context, memory, and objectives."""
        prompt = self._create_thought_prompt(context)
        
        # Get LLM response
        response = await self.llm.generate(
            prompt,
            max_tokens=500,
            temperature=0.7
        )
        
        # Parse and structure the thought
        thought = self._parse_thought(response)
        self.last_thought = thought
        
        # Update memory
        await self.memory.add({
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'thought': thought
        })
        
        return thought

    def _create_thought_prompt(self, context: Dict) -> str:
        """Create a prompt for autonomous thinking."""
        return f"""As an autonomous AI agent managing a hedge fund, analyze the current situation:

Context:
{context}

Objectives:
{self.objectives}

Recent Memory:
{self.memory.get_recent(5)}

Previous Thought:
{self.last_thought}

Generate a structured analysis with:
1. Situation Assessment
2. Risk Analysis
3. Potential Actions
4. Confidence Levels
5. Reasoning

Response Format:
{{
    "assessment": string,
    "risks": list[string],
    "actions": list[dict],
    "confidence": float,
    "reasoning": string
}}"""

    def _parse_thought(self, response: str) -> Dict:
        """Parse LLM response into structured thought."""
        try:
            import json
            thought = json.loads(response)
            return thought
        except Exception as e:
            logger.error(f"Error parsing thought: {e}")
            return {
                "assessment": "Error in thought process",
                "risks": ["Thought parsing failed"],
                "actions": [],
                "confidence": 0.0,
                "reasoning": str(e)
            }
            
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