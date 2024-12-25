import sys
import os
import argparse
import asyncio
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Sequence, TypedDict, List, Tuple

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math
import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.tools import calculate_bollinger_bands, calculate_intrinsic_value, calculate_macd, calculate_obv, calculate_rsi, search_line_items, get_financial_metrics, get_insider_trades, get_market_cap, get_prices, prices_to_df

class TradingAgent:
    def __init__(self, capital: float, trading_pairs: List[str], risk_factor: float, dry_run: bool, interval: int, show_reasoning: bool):
        self.capital = capital
        self.trading_pairs = trading_pairs
        self.risk_factor = risk_factor
        self.dry_run = dry_run
        self.interval = interval
        self.show_reasoning = show_reasoning

    async def run(self):
        # Implement the trading logic here
        pass

def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the trading agent."""
    parser = argparse.ArgumentParser(description='Crypto Trading Agent')
    
    # Required arguments
    parser.add_argument(
        '--pairs', 
        nargs='+', 
        required=True,
        help='Trading pairs to monitor (e.g., SOL BONK JUP)'
    )
    
    parser.add_argument(
        '--capital', 
        type=float, 
        required=True,
        help='Initial capital in USDC'
    )
    
    # Optional arguments
    parser.add_argument(
        '--risk', 
        type=float, 
        default=0.5,
        help='Risk factor (0.0-1.0)'
    )
    
    parser.add_argument(
        '--interval', 
        type=int, 
        default=300,
        help='Trading interval in seconds'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Run in simulation mode without real trades'
    )
    
    parser.add_argument(
        '--show-reasoning',
        action='store_true',
        help='Show AI reasoning for trades'
    )
    
    # Validate arguments
    args = parser.parse_args()
    validate_args(args)
    
    return args

def validate_args(args: argparse.Namespace):
    """Validate parsed arguments."""
    if args.capital <= 0:
        raise ValueError("Capital must be positive")
        
    if not 0 <= args.risk <= 1:
        raise ValueError("Risk must be between 0 and 1")
        
    if args.interval < 10:
        raise ValueError("Interval must be at least 10 seconds")
        
    for pair in args.pairs:
        if pair not in ['SOL', 'BONK', 'JUP']:
            raise ValueError(f"Unsupported trading pair: {pair}")

if __name__ == "__main__":
    args = parse_args()
    trading_agent = TradingAgent(
        capital=args.capital,
        trading_pairs=args.pairs,
        risk_factor=args.risk,
        dry_run=args.dry_run,
        interval=args.interval,
        show_reasoning=args.show_reasoning
    )
    asyncio.run(trading_agent.run())