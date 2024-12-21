# src/tools.py
import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass
import aiohttp
import pandas as pd
from datetime import datetime, timedelta
from .executors.jupiter_client import JupiterClient

logger = logging.getLogger(__name__)

@dataclass
class CryptoMetrics:
    price: float
    volume: float
    liquidity: float
    tvl: float
    holders: int
    transactions: int
    circulating_supply: float
    total_supply: float

class CryptoDataTools:
    def __init__(self, rpc_url: Optional[str] = None, config: Optional[Dict] = None):
        """Initialize CryptoDataTools with chain-specific configuration."""
        self.config = config or {}
        self.rpc_url = rpc_url or os.getenv("RPC_URL", "https://api.mainnet-beta.solana.com")
        self.helius_api = "https://api.helius.xyz/v0"
        self.helius_key = os.getenv("HELIUS_API_KEY")
        
        # Initialize Jupiter client
        self.jupiter = JupiterClient()
        
    async def __aenter__(self):
        # Initialize Jupiter client
        await self.jupiter.ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.jupiter.close()

    async def get_token_metrics(self, token: str) -> CryptoMetrics:
        """Get comprehensive token metrics."""
        try:
            # Get price and market data from Jupiter
            price = await self.jupiter.get_price(token)
            depth_data = await self.jupiter.get_market_depth(token)
            
            # Get on-chain data from Helius
            chain_data = await self.fetch_helius_metrics(token)
            
            # Calculate effective liquidity from market depth
            liquidity = self.calculate_effective_liquidity(depth_data)
            
            return CryptoMetrics(
                price=float(price) if price else 0.0,
                volume=depth_data.get(10000, {}).get('volume', 0.0),
                liquidity=liquidity,
                tvl=chain_data.get('tvl', 0.0),
                holders=chain_data.get('holders', 0),
                transactions=chain_data.get('transactions24h', 0),
                circulating_supply=chain_data.get('circulating_supply', 0.0),
                total_supply=chain_data.get('total_supply', 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error in get_token_metrics: {e}")
            raise

    def calculate_effective_liquidity(self, depth_data: Dict) -> float:
        """Calculate effective liquidity from market depth data."""
        if not depth_data:
            return 0.0
            
        # Use the largest test size that has less than 1% price impact
        for size in sorted(depth_data.keys(), reverse=True):
            if depth_data[size]['price_impact'] < 0.01:
                return float(size)
        
        return float(min(depth_data.keys()))

    async def get_historical_prices(
        self,
        token: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """Get historical price data using Jupiter quotes."""
        prices = []
        timestamps = []
        
        # Get current time
        end_time = datetime.now()
        time_step = timedelta(hours=1)
        
        for i in range(limit):
            timestamp = end_time - (i * time_step)
            try:
                price = await self.jupiter.get_price(token)
                if price:
                    prices.append(float(price))
                    timestamps.append(timestamp)
            except Exception as e:
                logger.error(f"Error fetching historical price for {timestamp}: {e}")
                continue
                
        df = pd.DataFrame({
            'price': prices,
            'timestamp': timestamps
        })
        df.set_index('timestamp', inplace=True)
        return df

    async def fetch_helius_metrics(self, token: str) -> Dict:
        """Fetch token metrics from Helius."""
        if not self.helius_key:
            logger.warning("No Helius API key provided")
            return {}
            
        try:
            headers = {
                "Authorization": f"Bearer {self.helius_key}",
                "Accept": "application/json"
            }
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.helius_api}/token-metrics"
                params = {"tokenAddress": token}
                
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info(f"Successfully fetched Helius metrics for {token}")
                        return data
                    else:
                        error_text = await response.text()
                        logger.error(f"Helius API error: {response.status} - {error_text}")
                        return {}
                    
        except Exception as e:
            logger.error(f"Error fetching Helius metrics: {e}")
            return {}