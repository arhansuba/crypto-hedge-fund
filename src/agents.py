import argparse
import asyncio
from datetime import datetime
from decimal import Decimal
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import aiohttp
from dotenv import load_dotenv

from llm_client import GaiaLLM
from tools import MarketAnalyzer

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

class JupiterClient:
    """Jupiter Protocol API client."""
    
    def __init__(self, api_version: str = "v6"):
        self.base_url = f"https://quote-api.jup.ag/{api_version}"
        self.session = None
        
    async def ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 50,
        swap_mode: str = "ExactIn",
        only_direct_routes: bool = False,
        restrict_intermediate_tokens: bool = True
    ) -> Dict:
        """Get a swap quote from Jupiter."""
        try:
            await self.ensure_session()
            
            url = f"{self.base_url}/quote"
            params = {
                "inputMint": input_mint,
                "outputMint": output_mint,
                "amount": str(amount),
                "slippageBps": slippage_bps,
                "swapMode": swap_mode,
                "onlyDirectRoutes": only_direct_routes,
                "restrictIntermediateTokens": restrict_intermediate_tokens
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Successfully got quote for {input_mint} -> {output_mint}")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter quote error: {response.status} - {error_text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting Jupiter quote: {e}")
            return {}

    async def get_swap_instruction(
        self,
        quote_response: Dict,
        user_public_key: str,
        wrap_unwrap_sol: bool = True,
        use_shared_accounts: bool = True,
        compute_unit_price_micro_lamports: Optional[int] = None
    ) -> Dict:
        """Get swap instructions from Jupiter."""
        try:
            await self.ensure_session()
            
            url = f"{self.base_url}/swap-instructions"
            payload = {
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": wrap_unwrap_sol,
                "useSharedAccounts": use_shared_accounts,
                "quoteResponse": quote_response
            }
            
            if compute_unit_price_micro_lamports:
                payload["computeUnitPriceMicroLamports"] = compute_unit_price_micro_lamports
            
            async with self.session.post(url, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info("Successfully got swap instructions")
                    return data
                else:
                    error_text = await response.text()
                    logger.error(f"Jupiter swap instruction error: {response.status} - {error_text}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting swap instructions: {e}")
            return {}

    async def get_price(
        self,
        input_mint: str,
        output_mint: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        amount: int = 1_000_000  # 1 unit of input token
    ) -> Optional[Decimal]:
        """Get token price in terms of USDC."""
        try:
            quote = await self.get_quote(
                input_mint=input_mint,
                output_mint=output_mint,
                amount=amount
            )
            
            if quote and 'outAmount' in quote:
                price = Decimal(quote['outAmount']) / Decimal(amount)
                return price
            return None
            
        except Exception as e:
            logger.error(f"Error getting price: {e}")
            return None

    async def get_market_depth(
        self,
        input_mint: str,
        output_mint: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",  # USDC
        test_sizes: list = [1000, 10000, 100000, 1000000]  # USDC amounts
    ) -> Dict:
        """Get market depth by testing different trade sizes."""
        depth_data = {}
        
        for size in test_sizes:
            try:
                quote = await self.get_quote(
                    input_mint=input_mint,
                    output_mint=output_mint,
                    amount=size,
                    swap_mode="ExactOut"
                )
                
                if quote:
                    depth_data[size] = {
                        'price': Decimal(quote['inAmount']) / Decimal(size),
                        'price_impact': float(quote.get('priceImpactPct', 0))
                    }
                    
            except Exception as e:
                logger.error(f"Error getting depth for size {size}: {e}")
                continue
                
        return depth_data

    async def close(self):
        """Close the client session."""
        if self.session:
            await self.session.close()
            self.session = None

class JupiterExecutor:
    """Jupiter Protocol trade execution handler."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session = None
        self.slippage_bps = self.config.get('slippage_bps', 50)  # 0.5%
        self.max_retries = self.config.get('max_retries', 3)
        
    async def ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def execute_trade(
        self,
        input_token: str,
        output_token: str,
        amount: Union[int, float, str],
        user_public_key: str,
        exact_out: bool = False
    ) -> Dict:
        """Execute a trade through Jupiter."""
        try:
            await self.ensure_session()
            
            # Get quote
            quote = await self.get_quote(
                input_token=input_token,
                output_token=output_token,
                amount=str(amount),
                slippage_bps=self.slippage_bps,
                exact_out=exact_out
            )
            
            if not quote:
                raise Exception("Failed to get quote")
                
            # Get swap transaction
            swap_tx = await self.get_swap_transaction(
                quote_response=quote,
                user_public_key=user_public_key
            )
            
            if not swap_tx:
                raise Exception("Failed to get swap transaction")
                
            # Execute swap
            result = await self.execute_swap(swap_tx)
            
            return {
                'success': True,
                'input_token': input_token,
                'output_token': output_token,
                'amount_in': amount,
                'amount_out': quote['outAmount'],
                'price_impact': quote.get('priceImpactPct', '0'),
                'tx_hash': result.get('txid'),
                'executed_price': Decimal(quote['outAmount']) / Decimal(amount)
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_token': input_token,
                'output_token': output_token,
                'amount_in': amount
            }

    async def get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: str,
        slippage_bps: int = 50,
        exact_out: bool = False
    ) -> Optional[Dict]:
        """Get quote from Jupiter."""
        try:
            params = {
                'inputMint': input_token,
                'outputMint': output_token,
                'amount': amount,
                'slippageBps': slippage_bps,
                'swapMode': 'ExactOut' if exact_out else 'ExactIn'
            }
            
            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Quote error: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None

    async def get_swap_transaction(
        self,
        quote_response: Dict,
        user_public_key: str
    ) -> Optional[Dict]:
        """Get swap transaction from Jupiter."""
        try:
            payload = {
                'quoteResponse': quote_response,
                'userPublicKey': user_public_key,
                'wrapUnwrapSOL': True,
                'useSharedAccounts': True,
                'dynamicComputeUnitLimit': True,
                'prioritizationFeeLamports': 'auto'
            }
            
            async with self.session.post(
                f"{self.base_url}/swap",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Swap transaction error: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting swap transaction: {e}")
            return None

    async def execute_swap(self, swap_tx: Dict) -> Dict:
        """Execute swap transaction with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Simulate execution (replace with actual blockchain submission)
                tx_hash = "simulated_tx_hash"  # Replace with actual submission
                
                return {
                    'success': True,
                    'txid': tx_hash,
                    'attempt': attempt + 1
                }
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached for swap execution: {e}")
                    raise
                    
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} in {wait_time}s")
                await asyncio.sleep(wait_time)

    async def check_transaction_status(self, tx_hash: str) -> Dict:
        """Check status of a submitted transaction."""
        try:
            # Replace with actual transaction status check
            return {
                'status': 'confirmed',
                'confirmations': 32,
                'slot': 123456789
            }
        except Exception as e:
            logger.error(f"Error checking transaction status: {e}")
            return {
                'status': 'unknown',
                'error': str(e)
            }

    async def simulate_swap(
        self,
        input_token: str,
        output_token: str,
        amount: str,
        user_public_key: str
    ) -> Dict:
        """Simulate swap to estimate costs and outcomes."""
        try:
            # Get quote first
            quote = await self.get_quote(
                input_token=input_token,
                output_token=output_token,
                amount=amount
            )
            
            if not quote:
                raise Exception("Failed to get quote for simulation")
                
            # Get swap transaction
            swap_tx = await self.get_swap_transaction(
                quote_response=quote,
                user_public_key=user_public_key
            )
            
            if not swap_tx:
                raise Exception("Failed to get swap transaction for simulation")
                
            return {
                'success': True,
                'input_amount': amount,
                'output_amount': quote['outAmount'],
                'price_impact': quote.get('priceImpactPct', '0'),
                'minimum_output': quote.get('otherAmountThreshold', '0'),
                'estimated_fees': {
                    'network': swap_tx.get('prioritizationFeeLamports', 0),
                    'platform': quote.get('platformFee', {'amount': '0'})['amount']
                }
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _validate_amounts(
        self,
        amount_in: Union[int, float, str],
        min_amount: Union[int, float, str]
    ) -> bool:
        """Validate trade amounts."""
        try:
            amount_in_dec = Decimal(str(amount_in))
            min_amount_dec = Decimal(str(min_amount))
            
            if amount_in_dec <= 0 or min_amount_dec <= 0:
                return False
                
            if min_amount_dec > amount_in_dec:
                return False
                
            return True
            
        except Exception:
            return False

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
        self.memory = MemoryState(size=1000)
        
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
            await self.memory.add({
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
                await self.memory.add({
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
        await self.memory.add({
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