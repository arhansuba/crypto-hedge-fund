# src/tools.py
import os
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from executors.jupiter_client import JupiterClient
from typing import Tuple
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

class MarketAnalyzer:
    """Market analysis tools for crypto trading."""
    
    def __init__(self):
        self.data_tools = CryptoDataTools()
        
    async def get_token_metrics(self, token: str) -> Dict:
        """Get comprehensive token metrics with analysis."""
        try:
            # Get base metrics
            metrics = await self.data_tools.get_token_metrics(token)
            
            # Get historical data
            history = await self.data_tools.get_historical_prices(token)
            
            # Perform technical analysis
            analysis = self.analyze_market_data(history, metrics)
            
            return {
                'metrics': metrics.__dict__,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error analyzing token {token}: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def analyze_market_data(
        self,
        history: pd.DataFrame,
        metrics: CryptoMetrics
    ) -> Dict:
        """Perform technical analysis on market data."""
        analysis = {}
        
        # Calculate price trends
        if not history.empty:
            analysis['price_trends'] = {
                'sma_20': float(history['price'].rolling(20).mean().iloc[-1]),
                'sma_50': float(history['price'].rolling(50).mean().iloc[-1]),
                'current_price': float(history['price'].iloc[-1]),
                'price_change_24h': self.calculate_price_change(history)
            }
            
            # Add momentum indicators
            analysis['momentum'] = {
                'rsi': self.calculate_rsi(history['price']),
                'macd': self.calculate_macd(history['price']),
                'volatility': self.calculate_volatility(history['price'])
            }
        
        # Add market health metrics
        analysis['market_health'] = {
            'liquidity_ratio': metrics.liquidity / metrics.volume if metrics.volume > 0 else 0,
            'holder_concentration': self.calculate_holder_concentration(metrics),
            'volume_stability': self.calculate_volume_stability(history) if not history.empty else 0
        }
        
        return analysis
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1])
        except Exception:
            return 50.0  # Neutral RSI on error
            
    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26
    ) -> Dict[str, float]:
        """Calculate MACD indicators."""
        try:
            exp1 = prices.ewm(span=fast_period, adjust=False).mean()
            exp2 = prices.ewm(span=slow_period, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(macd.iloc[-1] - signal.iloc[-1])
            }
        except Exception:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
            
    def calculate_volatility(self, prices: pd.Series, window: int = 20) -> float:
        """Calculate price volatility."""
        try:
            returns = prices.pct_change()
            return float(returns.std() * np.sqrt(window))
        except Exception:
            return 0.0
            
    def calculate_price_change(self, history: pd.DataFrame) -> float:
        """Calculate 24-hour price change percentage."""
        try:
            if len(history) >= 24:
                current_price = history['price'].iloc[-1]
                past_price = history['price'].iloc[-24]
                return ((current_price - past_price) / past_price) * 100
            return 0.0
        except Exception:
            return 0.0
            
    def calculate_holder_concentration(self, metrics: CryptoMetrics) -> float:
        """Calculate holder concentration metric."""
        try:
            # Simplified concentration metric
            if metrics.circulating_supply > 0:
                return metrics.holders / metrics.circulating_supply
            return 0.0
        except Exception:
            return 0.0
            
    def calculate_volume_stability(self, history: pd.DataFrame) -> float:
        """Calculate volume stability metric."""
        try:
            if 'volume' in history.columns:
                volume_std = history['volume'].std()
                volume_mean = history['volume'].mean()
                return 1 - (volume_std / volume_mean) if volume_mean > 0 else 0
            return 0.0
        except Exception:
            return 0.0

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



def calculate_bollinger_bands(prices_df: pd.DataFrame, window: int = 20, num_std_dev: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    rolling_mean = prices_df['close'].rolling(window=window).mean()
    rolling_std = prices_df['close'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std_dev)
    lower_band = rolling_mean - (rolling_std * num_std_dev)
    return upper_band, lower_band

def calculate_intrinsic_value(free_cash_flow: float, growth_rate: float, discount_rate: float, terminal_growth_rate: float, num_years: int) -> float:
    """Calculate intrinsic value using Discounted Cash Flow (DCF) model."""
    present_value = 0
    for year in range(1, num_years + 1):
        present_value += free_cash_flow * ((1 + growth_rate) ** year) / ((1 + discount_rate) ** year)
    terminal_value = (free_cash_flow * (1 + growth_rate) ** num_years * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
    intrinsic_value = present_value + terminal_value / ((1 + discount_rate) ** num_years)
    return intrinsic_value

def calculate_macd(prices_df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    exp1 = prices_df['close'].ewm(span=fast_period, adjust=False).mean()
    exp2 = prices_df['close'].ewm(span=slow_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def calculate_obv(prices_df: pd.DataFrame) -> pd.Series:
    """Calculate On-Balance Volume (OBV)."""
    obv = (np.sign(prices_df['close'].diff()) * prices_df['volume']).fillna(0).cumsum()
    return obv

def calculate_rsi(prices_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = prices_df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def search_line_items(ticker: str, line_items: List[str], period: str = 'ttm', limit: int = 1) -> List[Dict]:
    """Search for specific financial line items."""
    # Placeholder implementation
    return [{"line_item": item, "value": 1000} for item in line_items]

def get_financial_metrics(ticker: str, report_period: str = 'ttm', period: str = 'ttm', limit: int = 1) -> List[Dict]:
    """Get financial metrics for a given ticker."""
    # Placeholder implementation
    return [{"return_on_equity": 0.15, "net_margin": 0.20, "operating_margin": 0.15, "revenue_growth": 0.10, "earnings_growth": 0.10, "book_value_growth": 0.10, "current_ratio": 1.5, "debt_to_equity": 0.5, "free_cash_flow_per_share": 5.0, "earnings_per_share": 6.0, "price_to_earnings_ratio": 25, "price_to_book_ratio": 3, "price_to_sales_ratio": 5}]

def get_insider_trades(ticker: str, end_date: str, limit: int = 5) -> List[Dict]:
    """Get insider trades for a given ticker."""
    # Placeholder implementation
    return [{"transaction_shares": 1000, "transaction_price": 50.0} for _ in range(limit)]

def get_market_cap(ticker: str) -> float:
    """Get market capitalization for a given ticker."""
    # Placeholder implementation
    return 1000000000.0

def get_prices(ticker: str, start_date: str, end_date: str) -> List[Dict]:
    """Get historical prices for a given ticker."""
    # Placeholder implementation
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return [{"date": date.strftime('%Y-%m-%d'), "close": 100.0, "volume": 1000} for date in dates]

def prices_to_df(prices: List[Dict]) -> pd.DataFrame:
    """Convert list of price dictionaries to DataFrame."""
    return pd.DataFrame(prices).set_index('date')