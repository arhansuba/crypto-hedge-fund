import os
from typing import Dict, List, Tuple, Optional
import pandas as pd
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass

@dataclass
class CryptoMetrics:
    price: float
    volume: float
    liquidity: float
    tvl: float
    holders: int
    transactions: int

class CryptoDataTools:
    def __init__(self):
        self.jupiter_api = "https://price.jup.ag/v4"
        self.helius_api = "https://api.helius.xyz/v0"
        self.helius_key = os.getenv("HELIUS_API_KEY")
        
    async def get_token_prices(
        self, 
        token_addresses: List[str], 
        vs_token: str = "USDC"
    ) -> Dict[str, float]:
        """Fetch current token prices from Jupiter."""
        prices = {}
        for address in token_addresses:
            url = f"{self.jupiter_api}/price?ids={address}&vsToken={vs_token}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                prices[address] = data.get("data", {}).get("price", 0)
        return prices

    async def get_historical_prices(
        self, 
        token_address: str, 
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical price data using Helius."""
        url = f"{self.helius_api}/token/prices"
        headers = {"Authorization": f"Bearer {self.helius_key}"}
        params = {
            "tokenAddress": token_address,
            "startTime": int(start_date.timestamp()),
            "endTime": int(end_date.timestamp())
        }
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching data: {response.status_code}")
            
        data = response.json()
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)
        return df

    async def get_token_metrics(self, token_address: str) -> CryptoMetrics:
        """Get comprehensive token metrics using Helius."""
        url = f"{self.helius_api}/token-metrics"
        headers = {"Authorization": f"Bearer {self.helius_key}"}
        params = {"tokenAddress": token_address}
        
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            raise Exception(f"Error fetching metrics: {response.status_code}")
            
        data = response.json()
        return CryptoMetrics(
            price=data.get("price", 0),
            volume=data.get("volume24h", 0),
            liquidity=data.get("liquidity", 0),
            tvl=data.get("tvl", 0),
            holders=data.get("holders", 0),
            transactions=data.get("transactions24h", 0)
        )

class CryptoTechnicalAnalysis:
    @staticmethod
    def calculate_confidence_level(signals: Dict[str, float]) -> float:
        """Calculate trading confidence based on multiple indicators."""
        # Weighted average of different signals
        weights = {
            'price_momentum': 0.3,
            'volume_trend': 0.2,
            'liquidity_depth': 0.2,
            'holder_change': 0.15,
            'social_sentiment': 0.15
        }
        
        confidence = sum(weights[k] * v for k, v in signals.items())
        return min(max(confidence, 0), 1)  # Normalize between 0 and 1

    @staticmethod
    def calculate_liquidity_depth(
        orderbook_data: Dict[str, List[float]]
    ) -> Tuple[float, float]:
        """Calculate liquidity depth from Jupiter orderbook."""
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])
        
        bid_depth = sum(bid[1] for bid in bids)
        ask_depth = sum(ask[1] for ask in asks)
        
        return bid_depth, ask_depth

    @staticmethod
    def calculate_volume_profile(df: pd.DataFrame, bins: int = 10) -> pd.Series:
        """Calculate Volume Profile (VP) for price levels."""
        price_bins = pd.cut(df['price'], bins=bins)
        volume_profile = df.groupby(price_bins)['volume'].sum()
        return volume_profile

    @staticmethod
    def calculate_token_correlation(
        token1_prices: pd.Series,
        token2_prices: pd.Series,
        window: int = 24
    ) -> pd.Series:
        """Calculate rolling correlation between two tokens."""
        return token1_prices.rolling(window).corr(token2_prices)

    @staticmethod
    def calculate_onchain_momentum(
        df: pd.DataFrame,
        volume_weight: float = 0.4,
        tx_weight: float = 0.3,
        holder_weight: float = 0.3
    ) -> pd.Series:
        """Calculate momentum based on on-chain metrics."""
        volume_change = df['volume'].pct_change()
        tx_change = df['transactions'].pct_change()
        holder_change = df['holders'].pct_change()
        
        momentum = (
            volume_change * volume_weight +
            tx_change * tx_weight +
            holder_change * holder_weight
        )
        return momentum

class LiquidityAnalysis:
    @staticmethod
    async def analyze_pool_depth(
        pool_address: str,
        token_a: str,
        token_b: str
    ) -> Dict[str, float]:
        """Analyze liquidity pool depth using Orca/Meteora data."""
        # Implementation using Orca/Meteora APIs
        pass

    @staticmethod
    def calculate_impermanent_loss(
        price_change: float,
        liquidity_share: float
    ) -> float:
        """Calculate potential impermanent loss for LP positions."""
        sqrt_price_ratio = (1 + price_change) ** 0.5
        il = 2 * sqrt_price_ratio / (1 + sqrt_price_ratio) - 1
        return il * liquidity_share

    @staticmethod
    def estimate_slippage(
        amount: float,
        liquidity: float,
        price: float
    ) -> float:
        """Estimate slippage for a given trade size."""
        return (amount * price) / liquidity