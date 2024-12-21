# src/llm_client.py
import os
import json
import aiohttp
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

class GaiaLLM:
    """GaiaNet LLM client for Meta Llama 3."""
    
    def __init__(self, config_url: Optional[str] = None):
        self.config_url = config_url or "https://raw.gaianet.ai/llama-3-8b-instruct/config.json"
        self.model = "Meta-Llama-3-8B-Instruct-Q5_K_M"
        self.context_size = 409616
        self.nodes = []
        self.session = None
        
    async def initialize(self):
        """Initialize GaiaNet configuration."""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        try:
            # Load GaiaNet configuration
            async with self.session.get(self.config_url) as response:
                self.config = await response.json()
                
            # Get available nodes
            self.nodes = await self._get_active_nodes()
            logger.info(f"Initialized with {len(self.nodes)} active nodes")
            
        except Exception as e:
            logger.error(f"Error initializing GaiaNet: {e}")
            raise
            
    async def _get_active_nodes(self) -> List[str]:
        """Get list of active GaiaNet nodes."""
        try:
            # Use the node with highest throughput
            return [
                "0x4a9a395e9b969605c51d5e655bcaf60d1558ff7a.us.gaianet.network",
                "0xfeef0c75dcf512a9882fbb6ba929f965ba5a3462.us.gaianet.network"
            ]
        except Exception as e:
            logger.error(f"Error getting active nodes: {e}")
            return []
            
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate text using Llama 3."""
        if not self.nodes:
            await self.initialize()
            
        try:
            # Select best node based on throughput
            node = self.nodes[0]
            
            # Prepare request
            url = f"https://{node}/v1/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {os.getenv('GAIANET_API_KEY')}"
            }
            
            data = {
                "model": self.model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Make request
            async with self.session.post(url, headers=headers, json=data) as response:
                result = await response.json()
                
                if response.status == 200:
                    return result['choices'][0]['text']
                else:
                    raise Exception(f"GaiaNet error: {result.get('error', 'Unknown error')}")
                    
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
            
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Chat completion using Llama 3."""
        # Format messages into Llama chat format
        prompt = self._format_chat_messages(messages)
        return await self.generate(prompt, temperature=temperature)
        
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama 3 chat."""
        formatted = []
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted.append(f"<|system|>{content}</s>")
            elif role == 'user':
                formatted.append(f"<|user|>{content}</s>")
            elif role == 'assistant':
                formatted.append(f"<|assistant|>{content}</s>")
                
        return "\n".join(formatted)
        
    async def close(self):
        """Close client session."""
        if self.session:
            await self.session.close()