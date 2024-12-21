# src/agents/hedge_fund.py
from typing import Dict, List
from datetime import datetime
import logging
from decimal import Decimal

from agents.base import BaseAgent
from executors.jupiter import JupiterExecutor
from tools import MarketAnalyzer

logger = logging.getLogger(__name__)

class HedgeFundAgent(BaseAgent):
    """Autonomous hedge fund agent with advanced trading capabilities."""
    
    def __init__(
        self,
        llm_config: Dict,
        initial_capital: float,
        risk_tolerance: float = 0.7,
        max_position_size: float = 0.2,  # 20% of portfolio
        chains: List[str] = ['solana']
    ):
        super().__init__(
            llm_config=llm_config,
            objectives=[
                "Maximize risk-adjusted returns",
                "Maintain portfolio diversification",
                "Control downside risk",
                "Adapt to market conditions"
            ]
        )
        
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},
            'total_value': initial_capital
        }
        
        self.risk_params = {
            'tolerance': risk_tolerance,
            'max_position_size': max_position_size
        }
        
        # Initialize components
        self.market = MarketAnalyzer()
        self.executor = JupiterExecutor()
        
    async def analyze_market(self, tokens: List[str]) -> Dict:
        """Analyze market conditions and generate trading insights."""
        # Gather market data
        market_data = await self.market.get_token_metrics(tokens)
        
        # Generate market thoughts
        thought = await self.think({
            'type': 'market_analysis',
            'data': market_data,
            'portfolio': self.portfolio
        })
        
        # Generate trade ideas
        trades = await self._generate_trades(thought, market_data)
        
        return {
            'analysis': thought,
            'trades': trades,
            'timestamp': datetime.now().isoformat()
        }
        
    async def execute_trades(self, trades: List[Dict]) -> Dict:
        """Execute trading decisions with risk management."""
        results = {}
        
        for trade in trades:
            # Validate trade against risk limits
            if not self._validate_trade(trade):
                continue
                
            try:
                # Execute trade
                result = await self.executor.execute_trade(trade)
                
                # Update portfolio
                if result['success']:
                    self._update_portfolio(trade, result)
                    
                results[trade['token']] = result
                
                # Learn from execution
                await self.learn({
                    'type': 'trade_execution',
                    'trade': trade,
                    'result': result
                })
                
            except Exception as e:
                logger.error(f"Trade execution error: {e}")
                results[trade['token']] = {
                    'success': False,
                    'error': str(e)
                }
                
        return results
        
    def _validate_trade(self, trade: Dict) -> bool:
        """Validate trade against risk parameters."""
        # Check position size
        position_value = Decimal(trade['amount']) * Decimal(trade['price'])
        max_position = Decimal(self.portfolio['total_value']) * Decimal(self.risk_params['max_position_size'])
        
        if position_value > max_position:
            logger.warning(f"Trade exceeds max position size: {trade}")
            return False
            
        # Check portfolio concentration
        token = trade['token']
        current_exposure = Decimal(self.portfolio['positions'].get(token, 0))
        
        if current_exposure + position_value > max_position:
            logger.warning(f"Trade would exceed concentration limit: {trade}")
            return False
            
        return True
        
    def _update_portfolio(self, trade: Dict, result: Dict):
        """Update portfolio after successful trade."""
        token = trade['token']
        amount = Decimal(trade['amount'])
        price = Decimal(result['executed_price'])
        
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
        total = Decimal(self.portfolio['cash'])
        
        for token, amount in self.portfolio['positions'].items():
            price = Decimal(self.market.get_current_price(token))
            total += Decimal(amount) * price
            
        self.portfolio['total_value'] = float(total)
        
    async def _generate_trades(self, thought: Dict, market_data: Dict) -> List[Dict]:
        """Generate trade decisions based on market analysis."""
        trades = []
        
        for action in thought.get('actions', []):
            if action.get('type') == 'trade':
                trade = {
                    'token': action['token'],
                    'action': action['direction'],  # buy/sell
                    'amount': self._calculate_position_size(
                        action['token'],
                        action.get('confidence', 0.5),
                        market_data
                    ),
                    'price': market_data[action['token']]['price'],
                    'reason': action.get('reasoning', '')
                }
                trades.append(trade)
                
        return trades
        
    def _calculate_position_size(self, token: str, confidence: float, market_data: Dict) -> float:
        """Calculate optimal position size based on confidence and risk parameters."""
        max_position = float(Decimal(self.portfolio['total_value']) * 
                           Decimal(self.risk_params['max_position_size']))
                           
        # Scale by confidence and market liquidity
        liquidity_factor = min(1.0, market_data[token]['liquidity'] / max_position)
        size = max_position * confidence * liquidity_factor * self.risk_params['tolerance']
        
        return size