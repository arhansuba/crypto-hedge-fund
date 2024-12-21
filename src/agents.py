import argparse
import asyncio
from datetime import datetime
from decimal import Decimal
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import aiohttp
from dotenv import load_dotenv

from config import Config
from agents import HedgeFundAgent
from llm_client import GaiaLLM  # Import GaiaLLM

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
    llm_config = Config.get_llm_config("openai")
    gaianet_config = Config.get_gaianet_config()
    llm = GaiaLLM(config_url=gaianet_config.config_url)
    agent = HedgeFundAgent(
        llm_config=llm_config,
        initial_capital=args.capital,
        risk_tolerance=args.risk,
        chains=args.pairs
    )
    
    await agent.initialize()
    
    try:
        while True:
            # Analyze market
            analysis = await agent.analyze_market(args.pairs)
            logger.info("Market Analysis:", analysis)
            
            # Generate trades
            trades = analysis['trades']
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