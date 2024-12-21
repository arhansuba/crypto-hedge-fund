# src/executors/jupiter_client.py
import decimal
import aiohttp
import logging
from typing import Dict, List, Optional, Union
from decimal import Decimal
from dataclasses import dataclass

logger = logging.getLogger(__name__)

TOKEN_MINTS = {
    'SOL': 'So11111111111111111111111111111111111111112',
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'JUP': 'JUPyiwrYJFskUPiHa9toL3DeNMzPARXD7wqBqkSwkcj'
}

@dataclass
class TokenMetrics:
    """Token metrics data class."""
    price: float
    volume_24h: float
    liquidity: float
    holders: int
    transactions_24h: int
    error: Optional[str] = None

class TokenMetricsService:
    """Service for collecting token metrics."""
    
    def __init__(self, jupiter_client):
        """Initialize with a JupiterClient instance."""
        self.jupiter = jupiter_client
        self.default_quote_token = 'USDC'

    async def get_token_metrics(self, token: str) -> TokenMetrics:
        """Get comprehensive metrics for a token."""
        try:
            # Get price and derived metrics
            price = await self.jupiter.get_price(token, self.default_quote_token)
            if price is None:
                raise Exception("Failed to get token price")

            # Get mock/default metrics for now
            # TODO: Integrate with real data sources
            metrics = TokenMetrics(
                price=price,
                volume_24h=0.0,  # Replace with real volume data
                liquidity=0.0,   # Replace with real liquidity data
                holders=0,       # Replace with real holder count
                transactions_24h=0  # Replace with real transaction count
            )
            
            return metrics

        except Exception as e:
            return TokenMetrics(
                price=0.0,
                volume_24h=0.0,
                liquidity=0.0,
                holders=0,
                transactions_24h=0,
                error=str(e)
            )

    async def get_multiple_token_metrics(
        self, 
        tokens: List[str]
    ) -> Dict[str, TokenMetrics]:
        """Get metrics for multiple tokens."""
        metrics = {}
        for token in tokens:
            metrics[token] = await self.get_token_metrics(token)
        return metrics

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
        token: str,
        quote_token: str = 'USDC',
        amount: str = "1000000"  # 1 USDC with 6 decimals
    ) -> Optional[float]:
        """Get token price in terms of quote token."""
        try:
            if self.use_mock:
                # Return mock data for testing
                mock_prices = {
                    'SOL': 90.0,
                    'BONK': 0.000012,
                    'JUP': 1.20
                }
                return mock_prices.get(token.upper(), 0.0)

            # Get quote for exact amount of quote token
            quote = await self.get_quote(
                input_token=quote_token,
                output_token=token,
                amount=amount,
                slippage_bps=50
            )
            
            if not quote:
                logger.error(f"Failed to get price quote for {token}")
                return None
                
            # Calculate price from quote
            try:
                in_amount = Decimal(quote['inAmount'])
                out_amount = Decimal(quote['outAmount'])
                
                # Adjust for token decimals
                in_decimals = 6 if quote_token == 'USDC' else 9
                out_decimals = 9 if token == 'SOL' else 6
                
                price = float(in_amount / Decimal(10 ** in_decimals) / 
                            (out_amount / Decimal(10 ** out_decimals)))
                
                return price
                
            except (KeyError, ValueError, decimal.InvalidOperation) as e:
                logger.error(f"Error calculating price from quote: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting price for {token}: {e}")
            return None
            
    async def get_prices(
        self,
        tokens: List[str],
        quote_token: str = 'USDC'
    ) -> Dict[str, Optional[float]]:
        """Get prices for multiple tokens."""
        prices = {}
        for token in tokens:
            price = await self.get_price(token, quote_token)
            prices[token] = price
        return prices

    async def get_token_volume(
        self,
        token: str,
        quote_token: str = 'USDC'
    ) -> Optional[float]:
        """Get 24h trading volume for token."""
        # TODO: Implement real volume fetching
        if self.use_mock:
            mock_volumes = {
                'SOL': 150000000.0,
                'BONK': 25000000.0,
                'JUP': 5000000.0
            }
            return mock_volumes.get(token.upper(), 0.0)
        return 0.0

    async def get_token_liquidity(
        self,
        token: str,
        quote_token: str = 'USDC'
    ) -> Optional[float]:
        """Get total liquidity for token."""
        # TODO: Implement real liquidity fetching
        if self.use_mock:
            mock_liquidity = {
                'SOL': 500000000.0,
                'BONK': 50000000.0,
                'JUP': 10000000.0
            }
            return mock_liquidity.get(token.upper(), 0.0)
        return 0.0

    async def get_market_depth(
        self,
        input_token: str,
        output_token: str = "USDC",
        test_sizes: list = [1000, 10000, 100000, 1000000]  # USDC amounts
    ) -> Dict:
        """Get market depth by testing different trade sizes.
        
        Args:
            input_token: Token to get depth for (e.g., 'SOL', 'BONK')
            output_token: Quote token (defaults to USDC)
            test_sizes: List of amounts to test for depth
            
        Returns:
            Dictionary containing price and price impact for each test size
        """
        depth_data = {}
        
        for size in test_sizes:
            try:
                # Convert size to USDC decimals (6)
                size_in_decimals = str(int(size * 1_000_000))
                
                quote = await self.get_quote(
                    input_token=input_token,
                    output_token=output_token,
                    amount=size_in_decimals
                )
                
                if quote:
                    # Calculate effective price and impact
                    try:
                        in_amount = Decimal(quote['inAmount'])
                        out_amount = Decimal(quote['outAmount'])
                        
                        # Adjust for decimals
                        in_decimals = 6 if input_token == 'USDC' else 9
                        out_decimals = 6 if output_token == 'USDC' else 9
                        
                        effective_price = float(
                            (in_amount / Decimal(10 ** in_decimals)) /
                            (out_amount / Decimal(10 ** out_decimals))
                        )
                        
                        depth_data[size] = {
                            'price': effective_price,
                            'price_impact': float(quote.get('priceImpactPct', 0)),
                            'in_amount': str(in_amount),
                            'out_amount': str(out_amount)
                        }
                    except (decimal.InvalidOperation, KeyError) as e:
                        logger.error(f"Error calculating metrics for size {size}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error getting depth for size {size}: {e}")
                continue
                
        return depth_data

    async def get_token_metrics(
        self,
        token: str,
        quote_token: str = 'USDC'
    ) -> TokenMetrics:
        """Get comprehensive token metrics including price, volume, and liquidity."""
        try:
            # Get price
            price = await self.get_price(token, quote_token)
            if price is None:
                raise Exception("Failed to get token price")
                
            # Get market depth metrics
            depth = await self.get_market_depth(token, quote_token)
            
            # Calculate volume and liquidity from depth data
            total_volume = 0.0
            total_liquidity = 0.0
            
            if depth:
                # Use largest test size for liquidity estimate
                max_size = max(depth.keys())
                if max_size in depth:
                    max_depth = depth[max_size]
                    # Rough liquidity estimate based on max tested size
                    total_liquidity = float(max_depth['in_amount']) / 1_000_000  # Convert from USDC decimals
                    
                # Calculate approximate 24h volume (mock for now)
                total_volume = total_liquidity * 0.3  # Assume 30% daily turnover
            
            return TokenMetrics(
                price=price,
                volume_24h=total_volume,
                liquidity=total_liquidity,
                holders=0,  # Would need additional data source
                transactions_24h=0,  # Would need additional data source
                error=None
            )
            
        except Exception as e:
            logger.error(f"Error getting token metrics: {e}")
            return TokenMetrics(
                price=0.0,
                volume_24h=0.0,
                liquidity=0.0,
                holders=0,
                transactions_24h=0,
                error=str(e)
            )