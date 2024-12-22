import math
from typing import Annotated, Any, Dict, Sequence, TypedDict

import operator
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph

from .tools import calculate_bollinger_bands, calculate_intrinsic_value, calculate_macd, calculate_obv, calculate_rsi, search_line_items, get_financial_metrics, get_insider_trades, get_market_cap, get_prices, prices_to_df

import argparse
from datetime import datetime
import json
import ast
import argparse
import asyncio
from datetime import datetime
from decimal import Decimal
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import aiohttp
from dotenv import load_dotenv

from config import Config
from agents.hedge_fund import HedgeFundAgent
from llm_client import GaiaLLM  # Import GaiaLLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}
@dataclass
class TradeSignal:
    chain_id: str
    token: str
    action: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    confidence: float
    reasoning: str

@dataclass
class AgentState:
    """Agent's internal state."""
    cash: Decimal
    positions: Dict[str, Decimal]
    total_value: Decimal
    last_trade_time: Optional[datetime] = None
    trade_count: int = 0
    wins: int = 0
    losses: int = 0

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
                "amount": str(amount),  # Ensure amount is a string
                "slippageBps": str(slippage_bps),  # Ensure slippage_bps is a string
                "swapMode": swap_mode,
                "onlyDirectRoutes": str(only_direct_routes).lower(),  # Convert to string
                "restrictIntermediateTokens": str(restrict_intermediate_tokens).lower()  # Convert to string
            }
            
            async with self.session.get(url, params=params) as response:
                if (response.status == 200):
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

class JupiterExecutor:
    """Jupiter Protocol trade execution handler."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.base_url = "https://quote-api.jup.ag/v6"
        self.session = None
        self.slippage_bps = self.config.get('slippage_bps', 50)  # 0.5%
        self.max_retries = self.config.get('max_retries', 3)
        
    async def ensure_session(self):
        """Ensure aiohttp session is initialized."""
        if not self.session:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def execute_trade(
        self,
        input_token: str,
        output_token: str,
        amount: Union[int, float, str],
        user_public_key: str,
        exact_out: bool = False
    ) -> Dict:
        """Execute a trade through Jupiter."""
        try:
            await self.ensure_session()
            
            # Get quote
            quote = await self.get_quote(
                input_token=input_token,
                output_token=output_token,
                amount=str(amount),
                slippage_bps=self.slippage_bps,
                exact_out=exact_out
            )
            
            if not quote:
                raise Exception("Failed to get quote")
                
            # Get swap transaction
            swap_tx = await self.get_swap_transaction(
                quote_response=quote,
                user_public_key=user_public_key
            )
            
            if not swap_tx:
                raise Exception("Failed to get swap transaction")
                
            # Execute swap
            result = await self.execute_swap(swap_tx)
            
            return {
                'success': True,
                'input_token': input_token,
                'output_token': output_token,
                'amount_in': amount,
                'amount_out': quote['outAmount'],
                'price_impact': quote.get('priceImpactPct', '0'),
                'tx_hash': result.get('txid'),
                'executed_price': Decimal(quote['outAmount']) / Decimal(amount)
            }
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_token': input_token,
                'output_token': output_token,
                'amount_in': amount
            }

    async def get_quote(
        self,
        input_token: str,
        output_token: str,
        amount: str,
        slippage_bps: int = 50,
        exact_out: bool = False
    ) -> Optional[Dict]:
        """Get quote from Jupiter."""
        try:
            params = {
                'inputMint': input_token,
                'outputMint': output_token,
                'amount': amount,
                'slippageBps': slippage_bps,
                'swapMode': 'ExactOut' if exact_out else 'ExactIn'
            }
            
            async with self.session.get(f"{self.base_url}/quote", params=params) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Quote error: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting quote: {e}")
            return None

    async def get_swap_transaction(
        self,
        quote_response: Dict,
        user_public_key: str
    ) -> Optional[Dict]:
        """Get swap transaction from Jupiter."""
        try:
            payload = {
                'quoteResponse': quote_response,
                'userPublicKey': user_public_key,
                'wrapUnwrapSOL': True,
                'useSharedAccounts': True,
                'dynamicComputeUnitLimit': True,
                'prioritizationFeeLamports': 'auto'
            }
            
            async with self.session.post(
                f"{self.base_url}/swap",
                json=payload
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Swap transaction error: {error_text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting swap transaction: {e}")
            return None

    async def execute_swap(self, swap_tx: Dict) -> Dict:
        """Execute swap transaction with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Simulate execution (replace with actual blockchain submission)
                tx_hash = "simulated_tx_hash"  # Replace with actual submission
                
                return {
                    'success': True,
                    'txid': tx_hash,
                    'attempt': attempt + 1
                }
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Max retries reached for swap execution: {e}")
                    raise
                    
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{self.max_retries} in {wait_time}s")
                await asyncio.sleep(wait_time)

    async def check_transaction_status(self, tx_hash: str) -> Dict:
        """Check status of a submitted transaction."""
        try:
            # Replace with actual transaction status check
            return {
                'status': 'confirmed',
                'confirmations': 32,
                'slot': 123456789
            }
        except Exception as e:
            logger.error(f"Error checking transaction status: {e}")
            return {
                'status': 'unknown',
                'error': str(e)
            }

    async def simulate_swap(
        self,
        input_token: str,
        output_token: str,
        amount: str,
        user_public_key: str
    ) -> Dict:
        """Simulate swap to estimate costs and outcomes."""
        try:
            # Get quote first
            quote = await self.get_quote(
                input_token=input_token,
                output_token=output_token,
                amount=amount
            )
            
            if not quote:
                raise Exception("Failed to get quote for simulation")
                
            # Get swap transaction
            swap_tx = await self.get_swap_transaction(
                quote_response=quote,
                user_public_key=user_public_key
            )
            
            if not swap_tx:
                raise Exception("Failed to get swap transaction for simulation")
                
            return {
                'success': True,
                'input_amount': amount,
                'output_amount': quote['outAmount'],
                'price_impact': quote.get('priceImpactPct', '0'),
                'minimum_output': quote.get('otherAmountThreshold', '0'),
                'estimated_fees': {
                    'network': swap_tx.get('prioritizationFeeLamports', 0),
                    'platform': quote.get('platformFee', {'amount': '0'})['amount']
                }
            }
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _validate_amounts(
        self,
        amount_in: Union[int, float, str],
        min_amount: Union[int, float, str]
    ) -> bool:
        """Validate trade amounts."""
        try:
            amount_in_dec = Decimal(str(amount_in))
            min_amount_dec = Decimal(str(min_amount))
            
            if amount_in_dec <= 0 or min_amount_dec <= 0:
                return False
                
            if min_amount_dec > amount_in_dec:
                return False
                
            return True
            
        except Exception:
            return False

class MemoryState:
    """Memory state management for the agent."""
    def __init__(self, size: int = 1000):
        self.size = size
        self.memory = []

    async def add(self, entry: Dict):
        """Add a new entry to memory."""
        if len(self.memory) >= self.size:
            self.memory.pop(0)  # Remove the oldest entry if memory is full
        self.memory.append(entry)

    def get_recent(self, n: int = 5) -> List[Dict]:
        """Get the most recent n entries from memory."""
        return self.memory[-n:]
# Define agent state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]

##### Market Data Agent #####
def market_data_agent(state: AgentState):
    """Responsible for gathering and preprocessing market data"""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data["end_date"] or datetime.now().strftime('%Y-%m-%d')
    if not data["start_date"]:
        # Calculate 3 months before end_date
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        start_date = end_date_obj.replace(month=end_date_obj.month - 3) if end_date_obj.month > 3 else \
            end_date_obj.replace(year=end_date_obj.year - 1, month=end_date_obj.month + 9)
        start_date = start_date.strftime('%Y-%m-%d')
    else:
        start_date = data["start_date"]

    # Get the historical price data
    prices = get_prices(
        ticker=data["ticker"], 
        start_date=start_date, 
        end_date=end_date,
    )

    # Get the financial metrics
    financial_metrics = get_financial_metrics(
        ticker=data["ticker"], 
        report_period=end_date, 
        period='ttm', 
        limit=1,
    )

    # Get the insider trades
    insider_trades = get_insider_trades(
        ticker=data["ticker"], 
        end_date=end_date,
        limit=5,
    )

    # Get the market cap
    market_cap = get_market_cap(
        ticker=data["ticker"],
    )

    # Get the line_items
    financial_line_items = search_line_items(
        ticker=data["ticker"], 
        line_items=["free_cash_flow"],
        period='ttm',
        limit=1,
    )

    return {
        "messages": messages,
        "data": {
            **data, 
            "prices": prices, 
            "start_date": start_date, 
            "end_date": end_date,
            "financial_metrics": financial_metrics,
            "insider_trades": insider_trades,
            "market_cap": market_cap,
            "financial_line_items": financial_line_items,
        }
    }

##### Quantitative Agent #####
def quant_agent(state: AgentState):
    """Analyzes technical indicators and generates trading signals."""
    show_reasoning = state["metadata"]["show_reasoning"]

    data = state["data"]
    prices = data["prices"]
    prices_df = prices_to_df(prices)
    
    # Calculate indicators
    # 1. MACD (Moving Average Convergence Divergence)
    macd_line, signal_line = calculate_macd(prices_df)
    
    # 2. RSI (Relative Strength Index)
    rsi = calculate_rsi(prices_df)
    
    # 3. Bollinger Bands (Bollinger Bands)
    upper_band, lower_band = calculate_bollinger_bands(prices_df)
    
    # 4. OBV (On-Balance Volume)
    obv = calculate_obv(prices_df)
    
    # Generate individual signals
    signals = []
    
    # MACD signal
    if macd_line.iloc[-2] < signal_line.iloc[-2] and macd_line.iloc[-1] > signal_line.iloc[-1]:
        signals.append('bullish')
    elif macd_line.iloc[-2] > signal_line.iloc[-2] and macd_line.iloc[-1] < signal_line.iloc[-1]:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # RSI signal
    if rsi.iloc[-1] < 30:
        signals.append('bullish')
    elif rsi.iloc[-1] > 70:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # Bollinger Bands signal
    current_price = prices_df['close'].iloc[-1]
    if current_price < lower_band.iloc[-1]:
        signals.append('bullish')
    elif current_price > upper_band.iloc[-1]:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # OBV signal
    obv_slope = obv.diff().iloc[-5:].mean()
    if obv_slope > 0:
        signals.append('bullish')
    elif obv_slope < 0:
        signals.append('bearish')
    else:
        signals.append('neutral')
    
    # Add reasoning collection
    reasoning = {
        "MACD": {
            "signal": signals[0],
            "details": f"MACD Line crossed {'above' if signals[0] == 'bullish' else 'below' if signals[0] == 'bearish' else 'neither above nor below'} Signal Line"
        },
        "RSI": {
            "signal": signals[1],
            "details": f"RSI is {rsi.iloc[-1]:.2f} ({'oversold' if signals[1] == 'bullish' else 'overbought' if signals[1] == 'bearish' else 'neutral'})"
        },
        "Bollinger": {
            "signal": signals[2],
            "details": f"Price is {'below lower band' if signals[2] == 'bullish' else 'above upper band' if signals[2] == 'bearish' else 'within bands'}"
        },
        "OBV": {
            "signal": signals[3],
            "details": f"OBV slope is {obv_slope:.2f} ({signals[3]})"
        }
    }
    
    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level based on the proportion of indicators agreeing
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    # Generate the message content
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": {
            "MACD": reasoning["MACD"],
            "RSI": reasoning["RSI"],
            "Bollinger": reasoning["Bollinger"],
            "OBV": reasoning["OBV"]
        }
    }

    # Create the quant message
    message = HumanMessage(
        content=str(message_content),  # Convert dict to string for message content
        name="quant_agent",
    )

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Quant Agent")
    
    return {
        "messages": [message],
        "data": data,
    }

##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]
    metrics = data["financial_metrics"][0]
    financial_line_item = data["financial_line_items"][0]
    market_cap = data["market_cap"]

    # Initialize signals list for different fundamental aspects
    signals = []
    reasoning = {}
    
    # 1. Profitability Analysis
    profitability_score = 0
    if metrics["return_on_equity"] > 0.15:  # Strong ROE above 15%
        profitability_score += 1
    if metrics["net_margin"] > 0.20:  # Healthy profit margins
        profitability_score += 1
    if metrics["operating_margin"] > 0.15:  # Strong operating efficiency
        profitability_score += 1
        
    signals.append('bullish' if profitability_score >= 2 else 'bearish' if profitability_score == 0 else 'neutral')
    reasoning["Profitability"] = {
        "signal": signals[0],
        "details": f"ROE: {metrics['return_on_equity']:.2%}, Net Margin: {metrics['net_margin']:.2%}, Op Margin: {metrics['operating_margin']:.2%}"
    }
    
    # 2. Growth Analysis
    growth_score = 0
    if metrics["revenue_growth"] > 0.10:  # 10% revenue growth
        growth_score += 1
    if metrics["earnings_growth"] > 0.10:  # 10% earnings growth
        growth_score += 1
    if metrics["book_value_growth"] > 0.10:  # 10% book value growth
        growth_score += 1
        
    signals.append('bullish' if growth_score >= 2 else 'bearish' if growth_score == 0 else 'neutral')
    reasoning["Growth"] = {
        "signal": signals[1],
        "details": f"Revenue Growth: {metrics['revenue_growth']:.2%}, Earnings Growth: {metrics['earnings_growth']:.2%}"
    }
    
    # 3. Financial Health
    health_score = 0
    if metrics["current_ratio"] > 1.5:  # Strong liquidity
        health_score += 1
    if metrics["debt_to_equity"] < 0.5:  # Conservative debt levels
        health_score += 1
    if metrics["free_cash_flow_per_share"] > metrics["earnings_per_share"] * 0.8:  # Strong FCF conversion
        health_score += 1
        
    signals.append('bullish' if health_score >= 2 else 'bearish' if health_score == 0 else 'neutral')
    reasoning["Financial_Health"] = {
        "signal": signals[2],
        "details": f"Current Ratio: {metrics['current_ratio']:.2f}, D/E: {metrics['debt_to_equity']:.2f}"
    }
    
    # 4. Price to X ratios
    pe_ratio = metrics["price_to_earnings_ratio"]
    pb_ratio = metrics["price_to_book_ratio"]
    ps_ratio = metrics["price_to_sales_ratio"]
    
    price_ratio_score = 0
    if pe_ratio < 25:  # Reasonable P/E ratio
        price_ratio_score += 1
    if pb_ratio < 3:  # Reasonable P/B ratio
        price_ratio_score += 1
    if ps_ratio < 5:  # Reasonable P/S ratio
        price_ratio_score += 1
        
    signals.append('bullish' if price_ratio_score >= 2 else 'bearish' if price_ratio_score == 0 else 'neutral')
    reasoning["Price_Ratios"] = {
        "signal": signals[3],
        "details": f"P/E: {pe_ratio:.2f}, P/B: {pb_ratio:.2f}, P/S: {ps_ratio:.2f}"
    }

    # 5. Calculate intrinsic value and compare to market cap
    free_cash_flow = financial_line_item.get('free_cash_flow')
    intrinsic_value = calculate_intrinsic_value(
        free_cash_flow=free_cash_flow,
        growth_rate=metrics["earnings_growth"],
        discount_rate=0.10,
        terminal_growth_rate=0.03,
        num_years=5,
    )
    if market_cap < intrinsic_value:
        signals.append('bullish')
    else:
        signals.append('bearish')

    reasoning["Intrinsic_Value"] = {
        "signal": signals[4],
        "details": f"Intrinsic Value: ${intrinsic_value:,.2f}, Market Cap: ${market_cap:,.2f}"
    }
    
    # Determine overall signal
    bullish_signals = signals.count('bullish')
    bearish_signals = signals.count('bearish')
    
    if bullish_signals > bearish_signals:
        overall_signal = 'bullish'
    elif bearish_signals > bullish_signals:
        overall_signal = 'bearish'
    else:
        overall_signal = 'neutral'
    
    # Calculate confidence level
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals
    
    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": reasoning
    }
    
    # Create the fundamental analysis message
    message = HumanMessage(
        content=str(message_content),
        name="fundamentals_agent",
    )
    
    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Fundamental Analysis Agent")
    
    return {
        "messages": [message],
        "data": data,
    }

##### Sentiment Agent #####
def sentiment_agent(state: AgentState):
    """Analyzes market sentiment and generates trading signals."""
    data = state["data"]
    insider_trades = data["insider_trades"]
    show_reasoning = state["metadata"]["show_reasoning"]

    # Loop through the insider trades, if transaction_shares is negative, then it is a sell, which is bearish, if positive, then it is a buy, which is bullish
    signals = []
    for trade in insider_trades:
        if trade["transaction_shares"] < 0:
            signals.append("bearish")
        else:
            signals.append("bullish")

    # Determine overall signal
    bullish_signals = signals.count("bullish")
    bearish_signals = signals.count("bearish")
    if bullish_signals > bearish_signals:
        overall_signal = "bullish"
    elif bearish_signals > bullish_signals:
        overall_signal = "bearish"
    else:
        overall_signal = "neutral"

    # Calculate confidence level based on the proportion of indicators agreeing
    total_signals = len(signals)
    confidence = max(bullish_signals, bearish_signals) / total_signals

    message_content = {
        "signal": overall_signal,
        "confidence": f"{round(confidence * 100)}%",
        "reasoning": f"Bullish signals: {bullish_signals}, Bearish signals: {bearish_signals}"
    }

    # Print the reasoning if the flag is set
    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")

    # Create the sentiment message
    message = HumanMessage(
        content=str(message_content),
        name="sentiment_agent",
    )

    return {
        "messages": [message],
        "data": data,
    }

##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Evaluates portfolio risk and sets position limits based on comprehensive risk analysis."""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]
    data = state["data"]

    prices_df = prices_to_df(data["prices"])

    # Fetch messages from other agents
    quant_message = next(msg for msg in state["messages"] if msg.name == "quant_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(msg for msg in state["messages"] if msg.name == "sentiment_agent")

    try:
        fundamental_signals = json.loads(fundamentals_message.content)
        technical_signals = json.loads(quant_message.content)
        sentiment_signals = json.loads(sentiment_message.content)
    except Exception as e:
        fundamental_signals = ast.literal_eval(fundamentals_message.content)
        technical_signals = ast.literal_eval(quant_message.content)
        sentiment_signals = ast.literal_eval(sentiment_message.content)


    print(f"fundamental_signals: {fundamental_signals}")
    print(f"technical_signals: {technical_signals}")
    print(f"sentiment_signals: {sentiment_signals}")

    agent_signals = {
        "fundamental": fundamental_signals,
        "technical": technical_signals,
        "sentiment": sentiment_signals
    }

    # 1. Calculate Risk Metrics
    returns = prices_df['close'].pct_change().dropna()
    daily_vol = returns.std()
    volatility = daily_vol * (252 ** 0.5)  # Annualized volatility approximation
    var_95 = returns.quantile(0.05)         # Simple historical VaR at 95% confidence
    max_drawdown = (prices_df['close'] / prices_df['close'].cummax() - 1).min()

    # 2. Market Risk Assessment
    market_risk_score = 0

    # Volatility scoring
    if volatility > 0.30:     # High volatility
        market_risk_score += 2
    elif volatility > 0.20:   # Moderate volatility
        market_risk_score += 1

    # VaR scoring
    # Note: var_95 is typically negative. The more negative, the worse.
    if var_95 < -0.03:
        market_risk_score += 2
    elif var_95 < -0.02:
        market_risk_score += 1

    # Max Drawdown scoring
    if max_drawdown < -0.20:  # Severe drawdown
        market_risk_score += 2
    elif max_drawdown < -0.10:
        market_risk_score += 1

    # 3. Position Size Limits
    # Consider total portfolio value, not just cash
    current_stock_value = portfolio['stock'] * prices_df['close'].iloc[-1]
    total_portfolio_value = portfolio['cash'] + current_stock_value

    base_position_size = total_portfolio_value * 0.25  # Start with 25% max position of total portfolio
    
    if market_risk_score >= 4:
        # Reduce position for high risk
        max_position_size = base_position_size * 0.5
    elif market_risk_score >= 2:
        # Slightly reduce for moderate risk
        max_position_size = base_position_size * 0.75
    else:
        # Keep base size for low risk
        max_position_size = base_position_size

    # 4. Stress Testing
    stress_test_scenarios = {
        "market_crash": -0.20,
        "moderate_decline": -0.10,
        "slight_decline": -0.05
    }

    stress_test_results = {}
    current_position_value = current_stock_value

    for scenario, decline in stress_test_scenarios.items():
        potential_loss = current_position_value * decline
        portfolio_impact = potential_loss / (portfolio['cash'] + current_position_value) if (portfolio['cash'] + current_position_value) != 0 else math.nan
        stress_test_results[scenario] = {
            "potential_loss": potential_loss,
            "portfolio_impact": portfolio_impact
        }

    # 5. Risk-Adjusted Signals Analysis
    # Convert all confidences to numeric for proper comparison
    def parse_confidence(conf_str):
        return float(conf_str.replace('%', '')) / 100.0
    low_confidence = any(parse_confidence(signal['confidence']) < 0.30 for signal in agent_signals.values())

    # Check the diversity of signals. If all three differ, add to risk score
    # (signal divergence can be seen as increased uncertainty)
    unique_signals = set(signal['signal'] for signal in agent_signals.values())
    signal_divergence = (2 if len(unique_signals) == 3 else 0)

    risk_score = (market_risk_score * 2)  # Market risk contributes up to ~6 points total when doubled
    if low_confidence:
        risk_score += 4  # Add penalty if any signal confidence < 30%
    risk_score += signal_divergence

    # Cap risk score at 10
    risk_score = min(round(risk_score), 10)

    # 6. Generate Trading Action
    # If risk is very high, hold. If moderately high, consider reducing.
    # Else, follow fundamental signal as a baseline.
    if risk_score >= 8:
        trading_action = "hold"
    elif risk_score >= 6:
        trading_action = "reduce"
    else:
        trading_action = agent_signals['fundamental']['signal']

    message_content = {
        "max_position_size": float(max_position_size),
        "risk_score": risk_score,
        "trading_action": trading_action,
        "risk_metrics": {
            "volatility": float(volatility),
            "value_at_risk_95": float(var_95),
            "max_drawdown": float(max_drawdown),
            "market_risk_score": market_risk_score,
            "stress_test_results": stress_test_results
        },
        "reasoning": f"Risk Score {risk_score}/10: Market Risk={market_risk_score}, "
                     f"Volatility={volatility:.2%}, VaR={var_95:.2%}, "
                     f"Max Drawdown={max_drawdown:.2%}"
    }

    # Create the risk management message
    message = HumanMessage(
        content=json.dumps(message_content),
        name="risk_management_agent",
    )

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")

    return {"messages": state["messages"] + [message]}


##### Portfolio Management Agent #####
def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions and generates orders"""
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = state["data"]["portfolio"]

    # Get the quant agent, fundamentals agent, and risk management agent messages
    quant_message = next(msg for msg in state["messages"] if msg.name == "quant_agent")
    fundamentals_message = next(msg for msg in state["messages"] if msg.name == "fundamentals_agent")
    sentiment_message = next(msg for msg in state["messages"] if msg.name == "sentiment_agent")
    risk_message = next(msg for msg in state["messages"] if msg.name == "risk_management_agent")

    # Create the prompt template
    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a portfolio manager making final trading decisions.
                Your job is to make a trading decision based on the team's analysis while strictly adhering
                to risk management constraints.

                RISK MANAGEMENT CONSTRAINTS:
                - You MUST NOT exceed the max_position_size specified by the risk manager
                - You MUST follow the trading_action (buy/sell/hold) recommended by risk management
                - These are hard constraints that cannot be overridden by other signals

                When weighing the different signals for direction and timing:
                1. Fundamental Analysis (50% weight)
                   - Primary driver of trading decisions
                   - Should determine overall direction
                
                2. Technical/Quant Analysis (35% weight)
                   - Secondary confirmation
                   - Helps with entry/exit timing
                
                3. Sentiment Analysis (15% weight)
                   - Final consideration
                   - Can influence sizing within risk limits
                
                The decision process should be:
                1. First check risk management constraints
                2. Then evaluate fundamental outlook
                3. Use technical analysis for timing
                4. Consider sentiment for final adjustment
                
                Provide the following in your output:
                - "action": "buy" | "sell" | "hold",
                - "quantity": <positive integer>
                - "confidence": <float between 0 and 1>
                - "agent_signals": <list of agent signals including agent name, signal (bullish | bearish | neutral), and their confidence>
                - "reasoning": <concise explanation of the decision including how you weighted the signals>

                Trading Rules:
                - Never exceed risk management position limits
                - Only buy if you have available cash
                - Only sell if you have shares to sell
                - Quantity must be ≤ current position for sells
                - Quantity must be ≤ max_position_size from risk management"""
            ),
            (
                "human",
                """Based on the team's analysis below, make your trading decision.

                Quant Analysis Trading Signal: {quant_message}
                Fundamental Analysis Trading Signal: {fundamentals_message}
                Sentiment Analysis Trading Signal: {sentiment_message}
                Risk Management Trading Signal: {risk_message}

                Here is the current portfolio:
                Portfolio:
                Cash: {portfolio_cash}
                Current Position: {portfolio_stock} shares

                Only include the action, quantity, reasoning, confidence, and agent_signals in your output as JSON.  Do not include any JSON markdown.

                Remember, the action must be either buy, sell, or hold.
                You can only buy if you have available cash.
                You can only sell if you have shares in the portfolio to sell.
                """
            ),
        ]
    )

    # Generate the prompt
    prompt = template.invoke(
        {
            "quant_message": quant_message.content, 
            "fundamentals_message": fundamentals_message.content,
            "sentiment_message": sentiment_message.content,
            "risk_message": risk_message.content,
            "portfolio_cash": f"{portfolio['cash']:.2f}",
            "portfolio_stock": portfolio["stock"]
        }
    )
    # Invoke the LLM
    llm = GaiaLLM()  # Initialize the LLM client
    result = llm.invoke(prompt)

    # Create the portfolio management message
    message = HumanMessage(
        content=result.content,
        name="portfolio_management",
    )

    # Print the decision if the flag is set
    if show_reasoning:
        show_agent_reasoning(message.content, "Portfolio Management Agent")

    return {"messages": state["messages"] + [message]}

def show_agent_reasoning(output, agent_name):
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    if isinstance(output, (dict, list)):
        # If output is already a dictionary or list, just pretty print it
        print(json.dumps(output, indent=2))
    else:
        try:
            # Parse the string as JSON and pretty print it
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            # Fallback to original string if not valid JSON
            print(output)
    print("=" * 48)

##### Run the Hedge Fund #####
def run_hedge_fund(ticker: str, start_date: str, end_date: str, portfolio: dict, show_reasoning: bool = False):
    final_state = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="Make a trading decision based on the provided data.",
                )
            ],
            "data": {
                "ticker": ticker,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
            },
            "metadata": {
                "show_reasoning": show_reasoning,
            }
        },
    )
    return final_state["messages"][-1].content

# Define the new workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("market_data_agent", market_data_agent)
workflow.add_node("quant_agent", quant_agent)
workflow.add_node("fundamentals_agent", fundamentals_agent)
workflow.add_node("sentiment_agent", sentiment_agent)
workflow.add_node("risk_management_agent", risk_management_agent)
workflow.add_node("portfolio_management_agent", portfolio_management_agent)

# Define the workflow
workflow.set_entry_point("market_data_agent")
workflow.add_edge("market_data_agent", "quant_agent")
workflow.add_edge("market_data_agent", "fundamentals_agent")
workflow.add_edge("market_data_agent", "sentiment_agent")
workflow.add_edge("quant_agent", "risk_management_agent")
workflow.add_edge("fundamentals_agent", "risk_management_agent")
workflow.add_edge("sentiment_agent", "risk_management_agent")
workflow.add_edge("risk_management_agent", "portfolio_management_agent")
workflow.add_edge("portfolio_management_agent", END)

app = workflow.compile()

# Add this at the bottom of the file
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the hedge fund trading system')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD). Defaults to 3 months before end date')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD). Defaults to today')
    parser.add_argument('--show-reasoning', action='store_true', help='Show reasoning from each agent')
    
    args = parser.parse_args()
    
    # Validate dates if provided
    if args.start_date:
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("Start date must be in YYYY-MM-DD format")
    
    if args.end_date:
        try:
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            raise ValueError("End date must be in YYYY-MM-DD format")
    
    # Sample portfolio - you might want to make this configurable too
    portfolio = {
        "cash": 100000.0,  # $100,000 initial cash
        "stock": 0         # No initial stock position
    }
    async def main():
        """Main entry point."""
        # Load environment variables
        load_dotenv()
        
        # Parse arguments
        parser = argparse.ArgumentParser(description='AI Hedge Fund')
        parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
        parser.add_argument('--pairs', nargs='+', default=['SOL', 'BONK'], help='Trading pairs')
        parser.add_argument('--risk', type=float, default=0.7, help='Risk tolerance (0-1)')
        parser.add_argument('--dry-run', action='store_true', help='Run without executing trades')
        parser.add_argument('--interval', type=int, default=60, help='Trading interval in seconds')
        
        args = parser.parse_args()
        
        # LLM configuration
        llm_config = {
            "model": "Meta-Llama-3-8B-Instruct-Q5_K_M",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        
        # Initialize agent
        agent = None
        try:
            agent = HedgeFundAgent(
                initial_capital=args.capital,
                trading_pairs=args.pairs,
                risk_tolerance=args.risk,
                llm_config=llm_config
            )
            
            while True:
                try:
                    # Analyze market
                    logger.info(f"\nAnalyzing market for {args.pairs}...")
                    analysis = await agent.analyze_market(args.pairs)
                    
                    # Log market data
                    logger.info("\nMarket Analysis:")
                    for token, data in analysis['market_data'].items():
                        logger.info(f"\n{token}:")
                        for key, value in data.items():
                            if key != 'error':
                                logger.info(f"  {key}: {value}")
                            else:
                                logger.warning(f"  {key}: {value}")
                    
                    # Handle trades
                    trades = analysis.get('trades', [])
                    if trades:
                        logger.info("\nGenerated Trades:")
                        for trade in trades:
                            logger.info(
                                f"{trade['action'].upper()} {trade['token']}: "
                                f"Amount: {trade.get('amount', 0):.2f}, "
                                f"Confidence: {trade.get('confidence', 0):.2f}"
                            )
                        
                        # Execute trades if not dry run
                        if not args.dry_run:
                            results = await agent.execute_trades(trades)
                            
                            logger.info("\nTrade Results:")
                            for token, result in results.items():
                                if result.get('success'):
                                    logger.info(
                                        f"{token}: Success - "
                                        f"Price: ${result.get('executed_price', 0):.4f}, "
                                        f"Amount: {result.get('amount_out', 0):.4f}"
                                    )
                                else:
                                    logger.warning(f"{token}: Failed - {result.get('error', 'Unknown error')}")
                    else:
                        logger.info("\nNo trades generated this cycle")
                    
                    # Portfolio update
                    logger.info("\nPortfolio Status:")
                    logger.info(f"Cash: ${agent.portfolio['cash']:.2f}")
                    logger.info(f"Total Value: ${agent.portfolio['total_value']:.2f}")
                    logger.info("Positions:")
                    for token, amount in agent.portfolio['positions'].items():
                        try:
                            price = await agent.get_current_price(token)
                            value = amount * price
                            logger.info(f"  {token}: {amount:.4f} (${value:.2f})")
                        except Exception as e:
                            logger.error(f"Error getting price for {token}: {e}")
                    
                    # Wait before next cycle
                    await asyncio.sleep(args.interval)
                    
                except Exception as e:
                    logger.error(f"Error in trading cycle: {e}")
                    await asyncio.sleep(5)  # Short sleep on error
                    
        except KeyboardInterrupt:
            logger.info("\nShutting down...")
        except Exception as e:
            logger.error(f"Fatal error: {e}")
        finally:
            if agent:
                await agent.close()

    result = run_hedge_fund(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        portfolio=portfolio,
        show_reasoning=args.show_reasoning
    )
    print("\nFinal Result:")
    print(result)
if __name__ == "__main__":
    asyncio.run(main())