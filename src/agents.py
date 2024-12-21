import argparse
import asyncio
from datetime import datetime
from decimal import Decimal
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

from config import Config
from tools import CryptoDataTools, CryptoTechnicalAnalysis, LiquidityAnalysis
from llm_client import GaiaLLM
from tools import MarketAnalyzer
from executors.jupiter import JupiterExecutor

# Import LLM providers
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TradeSignal:
    chain_id: str
    token: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    confidence: float
    reasoning: str

@dataclass
class AgentState:
    """Agent's internal state."""
    cash: Decimal
    positions: Dict[str, Decimal]
    total_value: Decimal
    last_trade_time: Optional[datetime] = None
    trade_count: int = 0
    wins: int = 0
    losses: int = 0

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

class AutoHedgeFund:
    """Autonomous hedge fund agent using GaiaNet LLM."""
    
    def __init__(
        self,
        initial_capital: float,
        trading_pairs: List[str],
        risk_tolerance: float = 0.7,
        max_position_size: float = 0.2,  # 20% of portfolio
        min_trade_interval: int = 60  # seconds
    ):
        self.state = AgentState(
            cash=Decimal(str(initial_capital)),
            positions={},
            total_value=Decimal(str(initial_capital))
        )
        
        self.trading_pairs = trading_pairs
        self.risk_tolerance = risk_tolerance
        self.max_position_size = max_position_size
        self.min_trade_interval = min_trade_interval
        
        # Initialize components
        self.llm = GaiaLLM()
        self.market = MarketAnalyzer()
        self.executor = JupiterExecutor()
        
        # Memory system
        self.memory = []
        self.max_memory = 1000
        
    async def initialize(self):
        """Initialize and validate components."""
        try:
            await self.llm.initialize()
            logger.info("LLM initialized successfully")
            
            # Validate trading pairs
            for pair in self.trading_pairs:
                price = await self.market.get_current_price(pair)
                if not price:
                    logger.warning(f"Could not get price for {pair}")
            
            # Initialize memory with market state
            await self._update_memory({
                'type': 'initialization',
                'timestamp': datetime.now().isoformat(),
                'trading_pairs': self.trading_pairs,
                'initial_capital': float(self.state.cash)
            })
            
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    async def run(self):
        """Main agent loop with enhanced autonomy."""
        logger.info("Starting autonomous trading agent...")
        
        while True:
            try:
                # Check if we should trade
                if not self._should_trade():
                    await asyncio.sleep(5)
                    continue
                
                # Analyze market
                analysis = await self.analyze_market()
                await self._update_memory({
                    'type': 'analysis',
                    'data': analysis,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Generate and validate trades
                trades = await self.generate_trades(analysis)
                if trades:
                    valid_trades = self._validate_trades(trades)
                    if valid_trades:
                        results = await self.execute_trades(valid_trades)
                        self._process_results(results)
                
                # Adaptive sleep based on market conditions
                sleep_time = self._calculate_sleep_time(analysis)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await self._handle_error(e)
                await asyncio.sleep(30)  # Error cooldown

    def _should_trade(self) -> bool:
        """Determine if we should trade based on various factors."""
        now = datetime.now()
        
        # Check time since last trade
        if (self.state.last_trade_time and 
            (now - self.state.last_trade_time).seconds < self.min_trade_interval):
            return False
            
        # Check if we have enough capital
        if self.state.cash < Decimal('10'):  # Minimum trade size
            return False
            
        # Check win/loss ratio
        if self.state.trade_count > 10:
            win_rate = self.state.wins / self.state.trade_count
            if win_rate < 0.4:  # Below 40% win rate
                logger.warning("Trading paused due to low win rate")
                return False
                
        return True

    async def _update_memory(self, data: Dict):
        """Update agent's memory system."""
        self.memory.append(data)
        
        # Trim memory if needed
        if len(self.memory) > self.max_memory:
            self.memory = self.memory[-self.max_memory:]

    def _get_relevant_memories(self, context: str, limit: int = 5) -> List[Dict]:
        """Get relevant memories for current context."""
        # Simple relevance scoring
        scored_memories = []
        for memory in self.memory:
            score = 0
            if memory['type'] == context:
                score += 2
            if 'data' in memory:
                score += 1
            scored_memories.append((score, memory))
            
        # Return top memories
        sorted_memories = sorted(scored_memories, key=lambda x: x[0], reverse=True)
        return [m[1] for m in sorted_memories[:limit]]

    def _calculate_sleep_time(self, analysis: Dict) -> int:
        """Calculate adaptive sleep time based on market conditions."""
        base_time = self.min_trade_interval
        
        # Adjust based on volatility
        volatility = analysis.get('volatility', 0.5)
        if volatility > 0.8:  # High volatility
            return max(10, base_time // 2)  # More frequent trading
        elif volatility < 0.2:  # Low volatility
            return base_time * 2  # Less frequent trading
            
        return base_time

    async def _handle_error(self, error: Exception):
        """Handle errors and adjust strategy."""
        await self._update_memory({
            'type': 'error',
            'error': str(error),
            'timestamp': datetime.now().isoformat()
        })
        
        # Adjust risk tolerance on repeated errors
        error_count = sum(1 for m in self.memory[-10:] if m['type'] == 'error')
        if error_count > 3:
            self.risk_tolerance *= 0.8  # Reduce risk
            logger.warning("Reducing risk tolerance due to errors")

    def _process_results(self, results: Dict):
        """Process and learn from trade results."""
        for token, result in results.items():
            if result['success']:
                profit = result.get('profit', 0)
                if profit > 0:
                    self.state.wins += 1
                elif profit < 0:
                    self.state.losses += 1
                    
        self.state.trade_count += 1
        self.state.last_trade_time = datetime.now()

    def _validate_trades(self, trades: List[Dict]) -> List[Dict]:
        """Validate trades against current conditions."""
        valid_trades = []
        for trade in trades:
            # Calculate position size
            position_value = Decimal(str(trade['size'])) * Decimal(str(trade['price']))
            max_position = self.state.total_value * Decimal(str(self.max_position_size))
            
            # Validate size
            if position_value > max_position:
                logger.warning(f"Trade size too large for {trade['token']}")
                continue
                
            # Check if we have enough cash for buys
            if trade['action'] == 'buy' and position_value > self.state.cash:
                logger.warning(f"Insufficient funds for {trade['token']}")
                continue
                
            valid_trades.append(trade)
            
        return valid_trades

    async def analyze_market(self) -> Dict:
        """Analyze market conditions using Llama 3."""
        # Get market data
        market_data = {}
        for token in self.trading_pairs:
            metrics = await self.market.get_token_metrics(token)
            market_data[token] = metrics
            
        # Create analysis prompt
        prompt = self._create_analysis_prompt(market_data)
        
        # Get LLM analysis
        analysis = await self.llm.generate(prompt)
        
        # Parse and structure the analysis
        return self._parse_analysis(analysis)
        
    async def generate_trades(self, analysis: Dict) -> List[Dict]:
        """Generate trading decisions based on analysis."""
        # Create trading prompt
        prompt = self._create_trading_prompt(analysis)
        
        # Get LLM trading decisions
        decisions = await self.llm.generate(prompt)
        
        # Parse and validate trades
        trades = self._parse_trades(decisions)
        return self._validate_trades(trades)
        
    async def execute_trades(self, trades: List[Dict]) -> Dict:
        """Execute validated trades."""
        results = {}
        for trade in trades:
            try:
                # Execute trade through Jupiter
                result = await self.executor.execute_trade(trade)
                
                if result['success']:
                    # Update portfolio
                    self._update_portfolio(trade, result)
                    
                results[trade['token']] = result
                
            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                results[trade['token']] = {
                    'success': False,
                    'error': str(e)
                }
                
        return results
        
    def _create_analysis_prompt(self, market_data: Dict) -> str:
        """Create market analysis prompt for Llama 3."""
        return f"""Analyze the following market data as a hedge fund manager:

Market Data:
{json.dumps(market_data, indent=2)}

Portfolio:
{json.dumps(self.portfolio, indent=2)}

Provide analysis in the following format:
1. Market Conditions
2. Risk Assessment
3. Opportunities
4. Strategy Recommendations

Focus on:
- Technical analysis
- Market sentiment
- Risk factors
- Trading opportunities

Response Format:
{{
    "market_conditions": string,
    "risks": list[string],
    "opportunities": list[dict],
    "strategy": string,
    "confidence": float
}}"""

    def _create_trading_prompt(self, analysis: Dict) -> str:
        """Create trading decisions prompt."""
        return f"""Based on the following analysis, generate specific trading decisions:

Analysis:
{json.dumps(analysis, indent=2)}

Portfolio:
{json.dumps(self.portfolio, indent=2)}

Risk Tolerance: {self.risk_tolerance}

Generate trading decisions in the following format:
{{
    "trades": [
        {{
            "token": string,
            "action": "buy" or "sell",
            "size": float,
            "reasoning": string,
            "confidence": float
        }}
    ]
}}"""

    def _update_portfolio(self, trade: Dict, result: Dict):
        """Update portfolio after trade execution."""
        token = trade['token']
        amount = float(trade['size'])
        price = float(result['executed_price'])
        
        if trade['action'] == 'buy':
            self.portfolio['cash'] -= amount * price
            self.portfolio['positions'][token] = self.portfolio['positions'].get(token, 0) + amount
        else:
            self.portfolio['cash'] += amount * price
            self.portfolio['positions'][token] = self.portfolio['positions'].get(token, 0) - amount
            
        # Update total value
        self._calculate_total_value()
        
    def _calculate_total_value(self):
        """Calculate total portfolio value."""
        total = self.portfolio['cash']
        
        for token, amount in self.portfolio['positions'].items():
            price = float(self.market.get_current_price(token))
            total += amount * price
            
        self.portfolio['total_value'] = total

async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='AI Hedge Fund')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--pairs', nargs='+', default=['SOL', 'BONK'], help='Trading pairs')
    parser.add_argument('--risk', type=float, default=0.7, help='Risk tolerance (0-1)')
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = AutoHedgeFund(
        initial_capital=args.capital,
        trading_pairs=args.pairs,
        risk_tolerance=args.risk
    )
    
    await agent.initialize()
    
    try:
        while True:
            # Analyze market
            analysis = await agent.analyze_market()
            logger.info("Market Analysis:", analysis)
            
            # Generate trades
            trades = await agent.generate_trades(analysis)
            if trades:
                logger.info("Generated Trades:", trades)
                
                # Execute trades
                results = await agent.execute_trades(trades)
                logger.info("Trade Results:", results)
            
            # Wait before next cycle
            await asyncio.sleep(60)  # 1-minute cycle
            
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await agent.llm.close()

if __name__ == "__main__":
    asyncio.run(main())