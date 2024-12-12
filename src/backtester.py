from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class TradeExecution:
    timestamp: datetime
    token: str
    action: str
    quantity: float
    price: float
    slippage: float
    fees: float
    
@dataclass
class PortfolioState:
    cash: float
    token_balances: Dict[str, float]
    positions_value: float
    total_value: float
    timestamp: datetime

class CryptoBacktester:
    def __init__(
        self,
        agent,
        trading_pairs: List[str],
        start_date: datetime,
        end_date: datetime,
        initial_capital: float,
        data_tools,
        slippage_model="jupiter"
    ):
        self.agent = agent
        self.trading_pairs = trading_pairs
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.data_tools = data_tools
        self.slippage_model = slippage_model
        
        # Initialize portfolio
        self.portfolio = {
            "cash": initial_capital,
            "tokens": {pair: 0 for pair in trading_pairs}
        }
        self.portfolio_history: List[PortfolioState] = []
        self.trades_history: List[TradeExecution] = []
        
    async def estimate_slippage(
        self,
        token: str,
        quantity: float,
        price: float,
        action: str
    ) -> float:
        """Estimate slippage for a given trade using Jupiter data."""
        if self.slippage_model == "jupiter":
            # Fetch Jupiter route quote
            pool_data = await self.data_tools.get_token_metrics(token)
            slippage = self.data_tools.estimate_slippage(
                amount=quantity,
                liquidity=pool_data.liquidity,
                price=price
            )
            return min(slippage, 0.05)  # Cap at 5%
        return 0.001  # Default 0.1% slippage
        
    def calculate_fees(self, quantity: float, price: float) -> float:
        """Calculate trading fees including Jupiter fees."""
        trade_value = quantity * price
        jupiter_fee = trade_value * 0.0005  # 0.05% Jupiter fee
        return jupiter_fee

    async def execute_trade(
        self,
        token: str,
        action: str,
        quantity: float,
        current_price: float
    ) -> Optional[TradeExecution]:
        """Execute trade with slippage and fee considerations."""
        if quantity <= 0:
            return None
            
        slippage = await self.estimate_slippage(
            token, quantity, current_price, action
        )
        fees = self.calculate_fees(quantity, current_price)
        
        executed_price = (
            current_price * (1 + slippage) if action == "buy"
            else current_price * (1 - slippage)
        )
        total_cost = quantity * executed_price + fees
        
        if action == "buy":
            if total_cost <= self.portfolio["cash"]:
                self.portfolio["cash"] -= total_cost
                self.portfolio["tokens"][token] += quantity
                return TradeExecution(
                    timestamp=datetime.now(),
                    token=token,
                    action=action,
                    quantity=quantity,
                    price=executed_price,
                    slippage=slippage,
                    fees=fees
                )
        elif action == "sell":
            if quantity <= self.portfolio["tokens"][token]:
                self.portfolio["cash"] += total_cost - fees
                self.portfolio["tokens"][token] -= quantity
                return TradeExecution(
                    timestamp=datetime.now(),
                    token=token,
                    action=action,
                    quantity=quantity,
                    price=executed_price,
                    slippage=slippage,
                    fees=fees
                )
        return None

    async def run_backtest(self):
        """Run backtest simulation with market impact."""
        dates = pd.date_range(self.start_date, self.end_date, freq="1H")
        
        print("\nStarting crypto backtest...")
        print(f"{'Date':<20} {'Token':<10} {'Action':<6} {'Quantity':>10} "
              f"{'Price':>10} {'Slippage':>10} {'Fees':>10} {'Portfolio Value':>15}")
        print("-" * 95)

        for current_date in dates:
            # Get market state for all trading pairs
            market_state = {}
            for token in self.trading_pairs:
                metrics = await self.data_tools.get_token_metrics(token)
                lookback_data = await self.data_tools.get_historical_prices(
                    token,
                    current_date - timedelta(hours=24),
                    current_date
                )
                market_state[token] = {
                    "price": metrics.price,
                    "volume": metrics.volume,
                    "liquidity": metrics.liquidity,
                    "history": lookback_data
                }
            
            # Get agent's trading decisions
            decisions = await self.agent.generate_trading_signals(
                market_state=market_state,
                portfolio=self.portfolio
            )
            
            # Execute trades
            for token, decision in decisions.items():
                trade_execution = await self.execute_trade(
                    token=token,
                    action=decision["action"],
                    quantity=decision["quantity"],
                    current_price=market_state[token]["price"]
                )
                
                if trade_execution:
                    self.trades_history.append(trade_execution)
                    print(
                        f"{current_date:%Y-%m-%d %H:%M} {token:<10} "
                        f"{trade_execution.action:<6} {trade_execution.quantity:>10.4f} "
                        f"{trade_execution.price:>10.2f} {trade_execution.slippage:>10.4f} "
                        f"{trade_execution.fees:>10.2f}"
                    )
            
            # Update portfolio state
            total_value = self.portfolio["cash"]
            for token in self.trading_pairs:
                total_value += (
                    self.portfolio["tokens"][token] * market_state[token]["price"]
                )
            
            self.portfolio_history.append(
                PortfolioState(
                    cash=self.portfolio["cash"],
                    token_balances=self.portfolio["tokens"].copy(),
                    positions_value=total_value - self.portfolio["cash"],
                    total_value=total_value,
                    timestamp=current_date
                )
            )

    def analyze_performance(self):
        """Analyze trading performance with crypto-specific metrics."""
        performance_df = pd.DataFrame([
            {
                "Date": state.timestamp,
                "Portfolio Value": state.total_value,
                "Cash": state.cash,
                "Positions Value": state.positions_value
            }
            for state in self.portfolio_history
        ]).set_index("Date")
        
        # Calculate metrics
        total_return = (
            (performance_df["Portfolio Value"].iloc[-1] - self.initial_capital)
            / self.initial_capital
        )
        daily_returns = performance_df["Portfolio Value"].pct_change().dropna()
        
        # Risk metrics
        sharpe_ratio = (
            np.sqrt(365 * 24) * daily_returns.mean() / daily_returns.std()
        )
        sortino_ratio = (
            np.sqrt(365 * 24) * daily_returns.mean()
            / daily_returns[daily_returns < 0].std()
        )
        max_drawdown = (
            (performance_df["Portfolio Value"] 
             / performance_df["Portfolio Value"].cummax() - 1).min()
        )
        
        # Trading metrics
        winning_trades = len([t for t in self.trades_history 
                            if t.action == "sell" and t.price > t.price])
        total_trades = len(self.trades_history)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Print results
        print("\nPerformance Metrics:")
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Sortino Ratio: {sortino_ratio:.2f}")
        print(f"Maximum Drawdown: {max_drawdown:.2%}")
        print(f"Win Rate: {win_rate:.2%}")
        print(f"Total Trades: {total_trades}")
        
        # Plot results
        self.plot_performance(performance_df)
        return performance_df
        
    def plot_performance(self, performance_df: pd.DataFrame):
        """Plot portfolio performance and trading activity."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Portfolio value
        performance_df["Portfolio Value"].plot(ax=ax1)
        ax1.set_title("Portfolio Value Over Time")
        ax1.set_ylabel("Value (USDC)")
        
        # Asset allocation
        portfolio_allocation = pd.DataFrame([
            {**{"Date": state.timestamp}, 
             **state.token_balances}
            for state in self.portfolio_history
        ]).set_index("Date")
        portfolio_allocation.plot(ax=ax2, kind="area", stacked=True)
        ax2.set_title("Portfolio Allocation")
        ax2.set_ylabel("Token Amount")
        
        # Trading activity
        trades_df = pd.DataFrame([
            {
                "Date": t.timestamp,
                "Value": t.quantity * t.price * (1 if t.action == "buy" else -1)
            }
            for t in self.trades_history
        ]).set_index("Date")
        trades_df["Value"].plot(ax=ax3, kind="bar")
        ax3.set_title("Trading Activity")
        ax3.set_ylabel("Trade Value (USDC)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    import argparse
    from tools import CryptoDataTools  # Updated import statement
    from agents import Agent  # Import the Agent class
    
    parser = argparse.ArgumentParser(description='Run crypto backtesting simulation')
    parser.add_argument(
        '--trading_pairs',
        type=str,
        nargs='+',
        help='List of trading pairs (e.g., SOL BONK JUP)'
    )
    parser.add_argument(
        '--end_date',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--start_date',
        type=str,
        default=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--initial_capital',
        type=float,
        default=10000,
        help='Initial capital in USDC'
    )
    
    args = parser.parse_args()
    
    # Initialize tools and run backtest
    data_tools = CryptoDataTools()
    backtester = CryptoBacktester(
        agent=Agent(),  # Replace with your AI agent
        trading_pairs=args.trading_pairs,
        start_date=datetime.strptime(args.start_date, '%Y-%m-%d'),
        end_date=datetime.strptime(args.end_date, '%Y-%m-%d'),
        initial_capital=args.initial_capital,
        data_tools=data_tools
    )
    
    backtester.run_backtest()
    performance_df = backtester.analyze_performance()