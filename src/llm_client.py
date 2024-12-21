# src/llm_client.py
import aiohttp
import logging
from typing import Dict, List, Optional, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GaiaLLM:
    """GaiaNet LLM client for Llama 3."""
    
    def __init__(self):
        self.api_base = "https://0xe7d21e1bd35163c0bcdc6d5ea8c23f3c277f2d17.us.gaianet.network/v1"
        self.session = None
        self.retry_attempts = 3
        self.default_system_message = """You are an expert crypto trading AI assistant. 
        Analyze market data and provide clear, actionable insights. Focus on:
        - Technical analysis
        - Risk assessment
        - Market sentiment
        - Trading opportunities"""

    async def ensure_session(self):
        """Initialize aiohttp session with proper headers."""
        if not self.session:
            self.session = aiohttp.ClientSession(headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            })

    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Send a chat completion request to GaiaNet node."""
        try:
            await self.ensure_session()
            
            # Ensure system message is present
            if not any(msg.get('role') == 'system' for msg in messages):
                messages.insert(0, {
                    "role": "system",
                    "content": self.default_system_message
                })
            
            payload = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "model": "Meta-Llama-3-8B-Instruct-Q5_K_M"
            }
            
            url = f"{self.api_base}/chat/completions"
            
            for attempt in range(self.retry_attempts):
                try:
                    async with self.session.post(url, json=payload) as response:
                        if response.status == 200:
                            return await response.json()
                        elif response.status == 404:
                            # Fallback to simple completion if chat endpoint fails
                            return await self._fallback_completion(messages)
                        else:
                            error_text = await response.text()
                            logger.error(f"API error: {response.status} - {error_text}")
                            
                            if attempt == self.retry_attempts - 1:
                                raise Exception(f"API error: {response.status}")
                                
                except aiohttp.ClientError as e:
                    if attempt == self.retry_attempts - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}")
                    
            raise Exception("Max retries exceeded")

        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise

    async def _fallback_completion(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Fallback to basic completion if chat fails."""
        try:
            # Combine messages into a single prompt
            prompt = "\n".join(msg['content'] for msg in messages)
            
            url = f"{self.api_base}/completions"
            
            payload = {
                "prompt": prompt,
                "max_tokens": 1000,
                "temperature": 0.7,
                "model": "Meta-Llama-3-8B-Instruct-Q5_K_M"
            }
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    completion = await response.json()
                    # Convert to chat format
                    return {
                        "choices": [{
                            "message": {
                                "role": "assistant",
                                "content": completion['choices'][0]['text']
                            }
                        }]
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Fallback completion error: {response.status} - {error_text}")
                    raise Exception(f"Fallback completion failed: {response.status}")

        except Exception as e:
            logger.error(f"Fallback completion error: {e}")
            raise
            
    def _validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """Validate message format."""
        if not messages:
            return False
            
        required_keys = {'role', 'content'}
        valid_roles = {'system', 'user', 'assistant'}
        
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if not all(key in msg for key in required_keys):
                return False
            if msg['role'] not in valid_roles:
                return False
            if not isinstance(msg['content'], str):
                return False
                
        return True