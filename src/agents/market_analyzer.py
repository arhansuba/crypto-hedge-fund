# src/market_analyzer.py
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from tools import CryptoDataTools
from executors.jupiter_client import JupiterClient

logger = logging.getLogger(__name__)

@dataclass
class MarketAnalysis:
    """Market analysis result structure."""
    token: str
    price: float
    volume_24h: float
    price_change_24h: float
    technical_indicators: Dict[str, float]
    risk_metrics: Dict[str, float]
    trading_signals: Dict[str, Union[str, float]]
    liquidity_metrics: Dict[str, float]
    timestamp: str

class MarketAnalyzer:
    """Advanced market analysis for crypto trading."""
    
    def __init__(
        self,
        config: Optional[Dict] = None,
        lookback_period: int = 100,
        risk_free_rate: float = 0.03
    ):
        self.config = config or {}
        self.lookback_period = lookback_period
        self.risk_free_rate = risk_free_rate
        self.data_tools = CryptoDataTools()
        self.jupiter = JupiterClient()
        
    async def analyze_token(self, token: str) -> MarketAnalysis:
        """Perform comprehensive token analysis."""
        try:
            # Get market data
            metrics = await self.data_tools.get_token_metrics(token)
            history = await self.data_tools.get_historical_prices(token, self.lookback_period)
            liquidity = await self.get_liquidity_metrics(token)
            
            # Technical analysis
            technical = self.calculate_technical_indicators(history)
            
            # Risk analysis
            risk = self.calculate_risk_metrics(history, metrics)
            
            # Generate trading signals
            signals = self.generate_trading_signals(
                technical,
                risk,
                liquidity
            )
            
            return MarketAnalysis(
                token=token,
                price=float(metrics.price),
                volume_24h=float(metrics.volume),
                price_change_24h=self.calculate_price_change(history),
                technical_indicators=technical,
                risk_metrics=risk,
                trading_signals=signals,
                liquidity_metrics=liquidity,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing token {token}: {e}")
            raise
            
    def calculate_technical_indicators(self, history: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical analysis indicators."""
        try:
            price_series = history['price']
            
            return {
                'rsi': self.calculate_rsi(price_series),
                'macd': self.calculate_macd(price_series)['histogram'],
                'bollinger_position': self.calculate_bollinger_position(price_series),
                'momentum': self.calculate_momentum(price_series),
                'volatility': self.calculate_volatility(price_series)
            }
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return {}

    def calculate_risk_metrics(
        self,
        history: pd.DataFrame,
        metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk metrics."""
        try:
            returns = history['price'].pct_change().dropna()
            
            var = self.calculate_value_at_risk(returns)
            sharp = self.calculate_sharpe_ratio(returns)
            
            return {
                'value_at_risk': var,
                'sharpe_ratio': sharp,
                'volatility': returns.std() * np.sqrt(252),  # Annualized
                'max_drawdown': self.calculate_max_drawdown(history['price']),
                'liquidity_risk': self.calculate_liquidity_risk(metrics)
            }
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}
            
    async def get_liquidity_metrics(self, token: str) -> Dict[str, float]:
        """Get liquidity-related metrics."""
        try:
            depth_data = await self.jupiter.get_market_depth(token)
            
            return {
                'depth_2_percent': depth_data.get('depth_2percent', 0),
                'slippage_impact': depth_data.get('price_impact', 0),
                'maker_volume': depth_data.get('maker_volume_24h', 0),
                'taker_volume': depth_data.get('taker_volume_24h', 0)
            }
        except Exception as e:
            logger.error(f"Error getting liquidity metrics: {e}")
            return {}

    def generate_trading_signals(
        self,
        technical: Dict[str, float],
        risk: Dict[str, float],
        liquidity: Dict[str, float]
    ) -> Dict[str, Union[str, float]]:
        """Generate trading signals from analysis."""
        try:
            # Combine indicators for signal
            signal_strength = 0.0
            
            # Technical factors (40% weight)
            if technical.get('rsi', 50) < 30:
                signal_strength += 0.4  # Oversold
            elif technical.get('rsi', 50) > 70:
                signal_strength -= 0.4  # Overbought
                
            if technical.get('macd', 0) > 0:
                signal_strength += 0.2
            else:
                signal_strength -= 0.2
                
            # Risk factors (30% weight)
            risk_score = 0.3 * (1 - min(1, risk.get('value_at_risk', 0) / 0.1))
            signal_strength += risk_score
            
            # Liquidity factors (30% weight)
            liquidity_score = 0.3 * min(1, liquidity.get('depth_2_percent', 0) / 100000)
            signal_strength += liquidity_score
            
            # Generate signal
            if signal_strength > 0.3:
                action = 'BUY'
            elif signal_strength < -0.3:
                action = 'SELL'
            else:
                action = 'HOLD'
                
            return {
                'action': action,
                'confidence': abs(signal_strength),
                'signal_strength': signal_strength,
                'risk_score': risk_score,
                'liquidity_score': liquidity_score
            }
            
        except Exception as e:
            logger.error(f"Error generating trading signals: {e}")
            return {
                'action': 'HOLD',
                'confidence': 0,
                'signal_strength': 0,
                'risk_score': 0,
                'liquidity_score': 0
            }

    # Technical Analysis Helper Methods
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
            return 50.0

    def calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        try:
            fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
            slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
            macd = fast_ema - slow_ema
            signal = macd.ewm(span=signal_period, adjust=False).mean()
            histogram = macd - signal
            
            return {
                'macd': float(macd.iloc[-1]),
                'signal': float(signal.iloc[-1]),
                'histogram': float(histogram.iloc[-1])
            }
        except Exception:
            return {'macd': 0, 'signal': 0, 'histogram': 0}

    def calculate_bollinger_position(
        self,
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> float:
        """Calculate relative position within Bollinger Bands."""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            
            # Position as percentage between bands
            position = (prices.iloc[-1] - lower_band.iloc[-1]) / (upper_band.iloc[-1] - lower_band.iloc[-1])
            return float(position)
        except Exception:
            return 0.5

    def calculate_momentum(
        self,
        prices: pd.Series,
        period: int = 14
    ) -> float:
        """Calculate price momentum."""
        try:
            return float(prices.pct_change(period).iloc[-1])
        except Exception:
            return 0.0

    # Risk Analysis Helper Methods
    def calculate_value_at_risk(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        try:
            return float(np.percentile(returns, (1 - confidence_level) * 100))
        except Exception:
            return 0.0

    def calculate_sharpe_ratio(
        self,
        returns: pd.Series
    ) -> float:
        """Calculate Sharpe Ratio."""
        try:
            excess_returns = returns - self.risk_free_rate/252
            return float(np.sqrt(252) * excess_returns.mean() / returns.std())
        except Exception:
            return 0.0

    def calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate Maximum Drawdown."""
        try:
            rolling_max = prices.expanding(min_periods=1).max()
            drawdown = (prices - rolling_max) / rolling_max
            return float(drawdown.min())
        except Exception:
            return 0.0

    def calculate_liquidity_risk(self, metrics: Dict[str, float]) -> float:
        """Calculate liquidity risk score."""
        try:
            volume = float(metrics.get('volume', 0))
            liquidity = float(metrics.get('liquidity', 0))
            
            if volume > 0:
                return min(1.0, liquidity / volume)
            return 1.0
        except Exception:
            return 1.0