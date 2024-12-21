# src/executors/jupiter_client.py
import aiohttp
import logging
from typing import Dict, Optional, Union
from decimal import Decimal
import json

logger = logging.getLogger(__name__)

TOKEN_MINTS = {
    'SOL': 'So11111111111111111111111111111111111111112',
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'JUP': 'JUPyiwrYJFskUPiHa9toL3DeNMzPARXD7wqBqkSwkcj'
}

class JupiterClient:
    """Jupiter Protocol API client."""
    
    def __init__(self, use_mock: bool = True):
        """Initialize Jupiter client."""
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session = None
        self.use_mock = use_mock

    async def ensure_session(self):
        """Initialize aiohttp session."""
        if not self.session:
            self.session = aiohttp.ClientSession(
                headers={
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            )

    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()
            self.session = None

    def _get_token_mint(self, token: str) -> str:
        """Get token mint address."""
        return TOKEN_MINTS.get(token.upper(), token)

    async def get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: Union[int, float, str],
        slippage_bps: int = 50,
        exact_out: bool = False
    ) -> Optional[Dict]:
        """Get quote from Jupiter."""
        try:
            await self.ensure_session()
            
            params = {
                "inputMint": self._get_token_mint(input_token),
                "outputMint": self._get_token_mint(output_token),
                "amount": str(amount),
                "slippageBps": slippage_bps,
                "swapMode": "ExactOut" if exact_out else "ExactIn"
            }
            
            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Quote error: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None

    async def get_swap_tx(
        self,
        quote_response: Dict,
        user_public_key: str,
        options: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Get swap transaction from Jupiter."""
        try:
            await self.ensure_session()
            
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": True,
                "useSharedAccounts": True,
                "dynamicComputeUnitLimit": True,
                "skipUserAccountsRpcCalls": True,
                "prioritizationFeeLamports": "auto"
            }
            
            if options:
                payload.update(options)
            
            async with self.session.post(f"{self.base_url}/swap", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Swap error: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting swap tx: {e}")
            return None

    async def get_swap_instructions(
        self,
        quote_response: Dict,
        user_public_key: str,
        options: Optional[Dict] = None
    ) -> Optional[Dict]:
        """Get swap instructions from Jupiter."""
        try:
            await self.ensure_session()
            
            payload = {
                "quoteResponse": quote_response,
                "userPublicKey": user_public_key,
                "computeUnitPriceMicroLamports": "auto",
                "dynamicComputeUnitLimit": True
            }
            
            if options:
                payload.update(options)
            
            async with self.session.post(f"{self.base_url}/swap-instructions", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Instructions error: {response.status} - {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting instructions: {e}")
            return None

    async def execute_swap(
        self,
        input_token: str,
        output_token: str,
        amount: Union[int, float, str],
        user_public_key: str,
        slippage_bps: int = 50,
        exact_out: bool = False
    ) -> Dict:
        """Execute a swap through Jupiter."""
        try:
            # Get quote
            quote = await self.get_quote(
                input_token=input_token,
                output_token=output_token,
                amount=amount,
                slippage_bps=slippage_bps,
                exact_out=exact_out
            )
            
            if not quote:
                raise Exception("Failed to get quote")
                
            # Get swap transaction
            swap_tx = await self.get_swap_tx(
                quote_response=quote,
                user_public_key=user_public_key,
                options={
                    "dynamicSlippage": {
                        "minBps": max(10, slippage_bps // 2),
                        "maxBps": slippage_bps
                    }
                }
            )
            
            if not swap_tx:
                raise Exception("Failed to get swap transaction")
                
            return {
                'success': True,
                'input_token': input_token,
                'output_token': output_token,
                'amount_in': quote.get('inAmount'),
                'amount_out': quote.get('outAmount'),
                'price_impact': quote.get('priceImpactPct'),
                'slippage': slippage_bps / 10000,  # Convert to decimal
                'transaction': swap_tx
            }
            
        except Exception as e:
            logger.error(f"Swap execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_token': input_token,
                'output_token': output_token,
                'amount': amount
            }

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