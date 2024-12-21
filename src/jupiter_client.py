# src/jupiter_client.py
import aiohttp
import logging
from typing import Dict, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

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