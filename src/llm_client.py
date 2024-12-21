# src/llm_client.py
import aiohttp
import logging
from typing import Dict, List, Optional
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GaiaLLM:
    """GaiaNet LLM client for Llama 3."""
    
    def __init__(self):
        # Use the specific GaiaNet node URL
        self.base_url = "https://0xe7d21e1bd35163c0bcdc6d5ea8c23f3c277f2d17.us.gaianet.network/v1"
        self.model = "Meta-Llama-3-8B-Instruct-Q5_K_M"
        self.session = None

    async def ensure_session(self):
        """Initialize aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """Generate text using single prompt."""
        messages = [
            {"role": "system", "content": "You are an expert AI trading assistant specializing in cryptocurrency markets."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.chat(messages, temperature)
        return response

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Chat completion using GaiaNet node."""
        try:
            await self.ensure_session()

            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 1000
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }

            async with self.session.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    logger.error(f"GaiaNet API error: {response.status} - {error_text}")
                    raise Exception(f"GaiaNet API error: {response.status}")

        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise

    async def analyze_market(self, market_data: Dict) -> Dict:
        """Analyze market data using LLM."""
        try:
            prompt = self._create_market_analysis_prompt(market_data)
            response = await self.generate(prompt, temperature=0.7)
            return self._parse_analysis_response(response)
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def generate_trading_decision(
        self,
        analysis: Dict,
        portfolio: Dict,
        risk_params: Dict
    ) -> Dict:
        """Generate trading decision based on analysis."""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert crypto trading AI focused on generating precise trading decisions with careful risk management."
                },
                {
                    "role": "user",
                    "content": self._create_trading_decision_prompt(analysis, portfolio, risk_params)
                }
            ]
            
            response = await self.chat(messages, temperature=0.5)
            return self._parse_trading_decision(response)
            
        except Exception as e:
            logger.error(f"Trading decision generation error: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def _create_market_analysis_prompt(self, market_data: Dict) -> str:
        """Create market analysis prompt."""
        return f"""Analyze the following market data and provide detailed insights:

Market Data:
{json.dumps(market_data, indent=2)}

Provide analysis covering:
1. Market Conditions
2. Technical Indicators
3. Risk Factors
4. Trading Opportunities

Format your response as JSON with the following structure:
{{
    "market_conditions": string,
    "technical_analysis": {{
        "trend": string,
        "momentum": string,
        "volatility": float
    }},
    "risks": [string],
    "opportunities": [{{
        "token": string,
        "action": string,
        "reason": string,
        "confidence": float
    }}],
    "overall_sentiment": string
}}"""

    def _create_trading_decision_prompt(
        self,
        analysis: Dict,
        portfolio: Dict,
        risk_params: Dict
    ) -> str:
        """Create trading decision prompt."""
        return f"""Based on the following data, generate specific trading decisions:

Analysis:
{json.dumps(analysis, indent=2)}

Current Portfolio:
{json.dumps(portfolio, indent=2)}

Risk Parameters:
{json.dumps(risk_params, indent=2)}

Generate trading decisions in JSON format:
{{
    "trades": [
        {{
            "token": string,
            "action": "buy" or "sell",
            "size": float,
            "price_limit": float,
            "confidence": float,
            "reasoning": string
        }}
    ],
    "risk_assessment": {{
        "portfolio_risk": float,
        "market_risk": float,
        "risk_factors": [string]
    }}
}}"""

    def _parse_analysis_response(self, response: str) -> Dict:
        """Parse LLM analysis response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse analysis response as JSON")
            return {
                "error": "Invalid response format",
                "raw_response": response,
                "timestamp": datetime.now().isoformat()
            }

    def _parse_trading_decision(self, response: str) -> Dict:
        """Parse LLM trading decision response."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error("Failed to parse trading decision as JSON")
            return {
                "error": "Invalid decision format",
                "raw_response": response,
                "timestamp": datetime.now().isoformat()
            }