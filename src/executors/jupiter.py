# src/executors/jupiter.py
import asyncio
import aiohttp
import logging
from typing import Dict, Optional, Union
from decimal import Decimal
import json

logger = logging.getLogger(__name__)

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