from typing import Dict, List
from dataclasses import dataclass
from langchain_openai.chat_models import ChatOpenAI
from typing import Dict, List, Optional
import asyncio
import json
from tools import (
    CryptoDataTools,
    CryptoTechnicalAnalysis,
    LiquidityAnalysis
)
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional
import asyncio

from config import Config
from tools import CryptoDataTools, CryptoTechnicalAnalysis, LiquidityAnalysis

# Import LLM providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

@dataclass
class TradeSignal:
    chain_id: str
    token: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    confidence: float
    reasoning: str

class ChainAgnosticAgent:
    def __init__(
        self,
        chains: List[str],
        llm_provider: str = "openai",
        trading_pairs: Optional[Dict[str, List[str]]] = None,
        initial_capital: float = 10000
    ):
        # Load configurations
        self.config = Config()
        self.trading_config = self.config.get_trading_config()
        
        # Initialize chain configs
        self.chains = {
            chain: self.config.get_chain_config(chain)
            for chain in chains
        }
        
        # Set up LLM
        llm_config = self.config.get_llm_config(llm_provider)
        if not llm_config or not llm_config.api_key:
            raise ValueError(f"Missing API key for {llm_provider}")
            
        self.llm = self._initialize_llm(llm_config)
        
        # Initialize tools for each chain
        self.tools = {
            chain: {
                'data': CryptoDataTools(chain_config.rpc_url),
                'analysis': CryptoTechnicalAnalysis(),
                'liquidity': LiquidityAnalysis()
            }
            for chain, chain_config in self.chains.items()
        }
        
        # Set trading pairs per chain
        self.trading_pairs = trading_pairs or {chain: [] for chain in chains}
        self.initial_capital = initial_capital

    def _initialize_llm(self, llm_config):
        """Initialize the appropriate LLM based on provider."""
        if llm_config.provider == "openai":
            return ChatOpenAI(
                model=llm_config.model,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature
            )
        elif llm_config.provider == "anthropic":
            return ChatAnthropic(
                model=llm_config.model,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature
            )
        elif llm_config.provider == "groq":
            return ChatGroq(
                model=llm_config.model,
                api_key=llm_config.api_key,
                temperature=llm_config.temperature
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_config.provider}")

    async def analyze_market(self, chain_id: str, token: str):
        """Analyze market conditions for a specific token on a chain."""
        chain_tools = self.tools[chain_id]
        
        # Get market data
        metrics = await chain_tools['data'].get_token_metrics(token)
        history = await chain_tools['data'].get_historical_prices(
            token,
            limit=100
        )
        liquidity = await chain_tools['liquidity'].analyze_pool_depth(
            token,
            self.chains[chain_id].native_token,
            ""  # pool address if needed
        )
        
        # Perform technical analysis
        analysis = chain_tools['analysis'].analyze_token(
            history,
            metrics,
            liquidity
        )
        
        return {
            'metrics': metrics,
            'analysis': analysis,
            'liquidity': liquidity
        }

    async def generate_trade_signals(self) -> List[TradeSignal]:
        """Generate trade signals for all tokens across all chains."""
        signals = []
        
        for chain_id, tokens in self.trading_pairs.items():
            for token in tokens:
                # Analyze market
                analysis = await self.analyze_market(chain_id, token)
                
                # Generate signal using LLM
                signal = await self._generate_signal_with_llm(
                    chain_id,
                    token,
                    analysis
                )
                
                if signal:
                    signals.append(signal)
        
        return signals

    async def _generate_signal_with_llm(
        self,
        chain_id: str,
        token: str,
        analysis: Dict
    ) -> Optional[TradeSignal]:
        """Use LLM to generate trading signal from analysis."""
        prompt = self._create_analysis_prompt(chain_id, token, analysis)
        response = await self.llm.ainvoke(prompt)
        
        try:
            signal_data = self._parse_llm_response(response.content)
            return TradeSignal(
                chain_id=chain_id,
                token=token,
                action=signal_data['action'],
                quantity=signal_data['quantity'],
                price=analysis['metrics'].price,
                confidence=signal_data['confidence'],
                reasoning=signal_data['reasoning']
            )
        except Exception as e:
            print(f"Error parsing LLM response for {token}: {e}")
            return None

    def _create_analysis_prompt(self, chain_id: str, token: str, analysis: Dict) -> str:
        """Create prompt for LLM analysis."""
        return f"""Analyze the following market data for {token} on {chain_id}:

Technical Analysis:
{analysis['analysis']}

Market Metrics:
Price: {analysis['metrics'].price}
Volume: {analysis['metrics'].volume}
Liquidity: {analysis['liquidity']}

Generate a trading signal with the following format:
{{
    "action": "BUY" or "SELL" or "HOLD",
    "quantity": float (amount to trade),
    "confidence": float (0-1),
    "reasoning": string (explanation)
}}"""

    def _parse_llm_response(self, response: str) -> Dict:
        """Parse LLM response into structured data."""
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AI trading agent')
    parser.add_argument(
        '--chains',
        type=str,
        nargs='+',
        default=['solana'],
        help='Chains to trade on (e.g., solana ethereum)'
    )
    parser.add_argument(
        '--trading_pairs',
        type=str,
        nargs='+',
        help='Trading pairs for each chain'
    )
    parser.add_argument(
        '--llm_provider',
        type=str,
        default='openai',
        choices=['openai', 'anthropic', 'groq'],
        help='LLM provider to use'
    )
    parser.add_argument(
        '--initial_capital',
        type=float,
        default=10000,
        help='Initial capital in USD'
    )
    
    args = parser.parse_args()
    
    # Organize trading pairs by chain
    pairs_by_chain = {}
    if args.trading_pairs:
        # Simple distribution - assign all pairs to first chain
        # You might want to implement a more sophisticated way to specify pairs per chain
        pairs_by_chain[args.chains[0]] = args.trading_pairs
    
    # Initialize and run agent
    agent = ChainAgnosticAgent(
        chains=args.chains,
        llm_provider=args.llm_provider,
        trading_pairs=pairs_by_chain,
        initial_capital=args.initial_capital
    )
    
    # Run trading cycle
    signals = asyncio.run(agent.generate_trade_signals())
    
    # Print results
    print("\nTrading Signals:")
    for signal in signals:
        print(f"\nChain: {signal.chain_id}")
        print(f"Token: {signal.token}")
        print(f"Action: {signal.action}")
        print(f"Quantity: {signal.quantity}")
        print(f"Confidence: {signal.confidence:.2f}")
        print(f"Reasoning: {signal.reasoning}")