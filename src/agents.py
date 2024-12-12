from typing import Annotated, Any, Dict, List, Sequence, TypedDict
from datetime import datetime, timedelta
import json
import operator
from dataclasses import dataclass
from arch import arch_model
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, StateGraph
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import logging
import json
from tools import (
    CryptoDataTools,
    CryptoTechnicalAnalysis,
    LiquidityAnalysis,
    calculate_bollinger_bands,
    calculate_macd,
    calculate_obv,
    calculate_rsi,
    get_financial_metrics,
    get_insider_trades,
    get_prices,
    prices_to_df
)

# Initialize AI model
llm = ChatOpenAI(model="gpt-4")
data_tools = CryptoDataTools()
tech_analysis = CryptoTechnicalAnalysis()
liquidity_analysis = LiquidityAnalysis()

def merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    return {**a, **b}

@dataclass
class TokenMetrics:
    price: float
    volume_24h: float
    liquidity: float
    holders: int
    transactions_24h: int
    market_cap: float
    fully_diluted_val: float
    circulating_supply: float
    total_supply: float

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[Dict[str, Any], merge_dicts]
    metadata: Annotated[Dict[str, Any], merge_dicts]

async def market_data_agent(state: AgentState):
    """Gathers and preprocesses crypto market data from multiple sources"""
    messages = state["messages"]
    data = state["data"]

    # Set default dates
    end_date = data.get("end_date") or datetime.now()
    if not data.get("start_date"):
        start_date = end_date - timedelta(days=30)
    else:
        start_date = datetime.strptime(data["start_date"], '%Y-%m-%d')

    token_metrics = {}
    historical_prices = {}
    liquidity_data = {}

    # Gather data for each trading pair
    for token in data["trading_pairs"]:
        # Get current metrics using Helius
        metrics = await data_tools.get_token_metrics(token)
        token_metrics[token] = TokenMetrics(
            price=metrics.price,
            volume_24h=metrics.volume,
            liquidity=metrics.liquidity,
            holders=metrics.holders,
            transactions_24h=metrics.transactions,
            market_cap=metrics.price * metrics.circulating_supply,
            fully_diluted_val=metrics.price * metrics.total_supply,
            circulating_supply=metrics.circulating_supply,
            total_supply=metrics.total_supply
        )

        # Get historical price data from Jupiter
        historical_prices[token] = await data_tools.get_historical_prices(
            token,
            start_date=start_date,
            end_date=end_date
        )

        # Get liquidity pool data from Orca/Meteora
        liquidity_data[token] = await liquidity_analysis.analyze_pool_depth(
            token,
            "USDC",
            token
        )

        # Get financial metrics
        financial_metrics = await get_financial_metrics(token)
        token_metrics[token].update(financial_metrics)

        # Get insider trades
        insider_trades = await get_insider_trades(token)
        token_metrics[token].update(insider_trades)

        # Get prices and convert to DataFrame
        prices = await get_prices(token)
        prices_df = prices_to_df(prices)
        historical_prices[token] = prices_df

    # Get on-chain metrics using Helius
    onchain_metrics = {
        token: await data_tools.get_token_on_chain_metrics(token)
        for token in data["trading_pairs"]
    }

    return {
        "messages": messages,
        "data": {
            **data,
            "token_metrics": token_metrics,
            "historical_prices": historical_prices,
            "liquidity_data": liquidity_data,
            "onchain_metrics": onchain_metrics,
            "start_date": start_date,
            "end_date": end_date
        }
    }

def show_agent_reasoning(output: Dict[str, Any], agent_name: str):
    """Display agent's reasoning in a formatted way"""
    print(f"\n{'=' * 10} {agent_name.center(28)} {'=' * 10}")
    if isinstance(output, (dict, list)):
        print(json.dumps(output, indent=2))
    else:
        try:
            parsed_output = json.loads(output)
            print(json.dumps(parsed_output, indent=2))
        except json.JSONDecodeError:
            print(output)
    print("=" * 48)

# Initialize core components
async def initialize_trading_environment(
    trading_pairs: List[str],
    initial_portfolio: Dict[str, float]
) -> Dict[str, Any]:
    """Initialize the trading environment with necessary components"""
    return {
        "trading_pairs": trading_pairs,
        "portfolio": initial_portfolio,
        "data_tools": data_tools,
        "tech_analysis": tech_analysis,
        "liquidity_analysis": liquidity_analysis
    }
async def quant_agent(state: AgentState):
    """Analyzes technical indicators and on-chain metrics for trading signals."""
    show_reasoning = state["metadata"]["show_reasoning"]
    data = state["data"]

    # Initialize signals dictionary for each trading pair
    signals = {}
    reasoning = {}

    for token in data["trading_pairs"]:
        prices_df = data["historical_prices"][token]
        metrics = data["token_metrics"][token]
        liquidity = data["liquidity_data"][token]
        onchain = data["onchain_metrics"][token]

        # Calculate technical indicators
        technical_signals = await calculate_technical_signals(
            prices_df=prices_df,
            token_metrics=metrics
        )

        # Calculate on-chain signals
        onchain_signals = await calculate_onchain_signals(
            onchain_metrics=onchain,
            token_metrics=metrics
        )

        # Calculate liquidity signals
        liquidity_signals = await calculate_liquidity_signals(
            liquidity_data=liquidity,
            token_metrics=metrics
        )

        # Combine all signals for final analysis
        token_analysis = combine_signals(
            technical=technical_signals,
            onchain=onchain_signals,
            liquidity=liquidity_signals
        )

        signals[token] = token_analysis["signal"]
        reasoning[token] = token_analysis["reasoning"]

    message_content = {
        "signals": signals,
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat()
    }

    if show_reasoning:
        show_agent_reasoning(message_content, "Quant Agent")

    return {
        "messages": [HumanMessage(content=str(message_content), name="quant_agent")],
        "data": data,
    }

async def calculate_technical_signals(
    prices_df: pd.DataFrame,
    token_metrics: TokenMetrics
) -> Dict[str, Any]:
    """Calculate technical analysis signals for crypto."""
    
    # Price action analysis
    sma_20 = prices_df['close'].rolling(window=20).mean()
    sma_50 = prices_df['close'].rolling(window=50).mean()
    ema_12 = prices_df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = prices_df['close'].ewm(span=26, adjust=False).mean()
    
    # Volume analysis
    volume_sma = prices_df['volume'].rolling(window=20).mean()
    current_volume = prices_df['volume'].iloc[-1]
    
    # Momentum indicators
    rsi = calculate_rsi(prices_df, period=14)
    macd, macd_signal = calculate_macd(prices_df)
    obv = calculate_obv(prices_df)
    
    # Volatility analysis
    atr = tech_analysis.calculate_atr(prices_df, period=14)
    bollinger = calculate_bollinger_bands(prices_df)

    current_price = prices_df['close'].iloc[-1]
    signals = []
    details = {}

    # Trend analysis
    if current_price > sma_20.iloc[-1] and sma_20.iloc[-1] > sma_50.iloc[-1]:
        signals.append('bullish')
        details['trend'] = 'Upward trend: Price above both SMAs'
    elif current_price < sma_20.iloc[-1] and sma_20.iloc[-1] < sma_50.iloc[-1]:
        signals.append('bearish')
        details['trend'] = 'Downward trend: Price below both SMAs'
    else:
        signals.append('neutral')
        details['trend'] = 'No clear trend'

    # Volume analysis
    if current_volume > volume_sma.iloc[-1] * 1.5:
        signals.append('bullish')
        details['volume'] = 'High volume: Above average'
    elif current_volume < volume_sma.iloc[-1] * 0.5:
        signals.append('bearish')
        details['volume'] = 'Low volume: Below average'
    else:
        signals.append('neutral')
        details['volume'] = 'Normal volume levels'

    # Momentum analysis
    if rsi.iloc[-1] < 30:
        signals.append('bullish')
        details['momentum'] = 'Oversold: RSI below 30'
    elif rsi.iloc[-1] > 70:
        signals.append('bearish')
        details['momentum'] = 'Overbought: RSI above 70'
    else:
        signals.append('neutral')
        details['momentum'] = 'Normal momentum'

    # OBV analysis
    if obv.iloc[-1] > obv.mean():
        signals.append('bullish')
        details['obv'] = 'Positive OBV trend'
    else:
        signals.append('neutral')
        details['obv'] = 'Neutral OBV trend'

    return {
        "signals": signals,
        "details": details
    }

async def calculate_onchain_signals(
    onchain_metrics: Dict[str, Any],
    token_metrics: TokenMetrics
) -> Dict[str, Any]:
    """Analyze on-chain metrics for trading signals."""
    
    signals = []
    details = {}

    # Holder analysis
    holder_change = onchain_metrics['holder_change_24h']
    if holder_change > 0.05:  # 5% increase in holders
        signals.append('bullish')
        details['holders'] = f'Strong holder growth: {holder_change:.1%} increase'
    elif holder_change < -0.05:
        signals.append('bearish')
        details['holders'] = f'Holder decline: {holder_change:.1%} decrease'
    else:
        signals.append('neutral')
        details['holders'] = 'Stable holder base'

    # Transaction analysis
    tx_volume = onchain_metrics['transaction_volume_24h']
    avg_tx_volume = onchain_metrics['avg_transaction_volume_7d']
    if tx_volume > avg_tx_volume * 1.5:
        signals.append('bullish')
        details['transactions'] = 'High transaction activity'
    elif tx_volume < avg_tx_volume * 0.5:
        signals.append('bearish')
        details['transactions'] = 'Low transaction activity'
    else:
        signals.append('neutral')
        details['transactions'] = 'Normal transaction activity'

    # Whale analysis
    whale_accumulation = onchain_metrics['whale_accumulation_24h']
    if whale_accumulation > 0:
        signals.append('bullish')
        details['whales'] = 'Whales accumulating'
    elif whale_accumulation < 0:
        signals.append('bearish')
        details['whales'] = 'Whales distributing'
    else:
        signals.append('neutral')
        details['whales'] = 'No significant whale activity'

    return {
        "signals": signals,
        "details": details
    }

async def calculate_liquidity_signals(
    liquidity_data: Dict[str, Any],
    token_metrics: TokenMetrics
) -> Dict[str, Any]:
    """Analyze DEX liquidity metrics for trading signals."""
    
    signals = []
    details = {}

    # Liquidity depth analysis
    depth_ratio = liquidity_data['depth_2percent'] / token_metrics.volume_24h
    if depth_ratio > 0.3:
        signals.append('bullish')
        details['liquidity'] = 'Deep liquidity relative to volume'
    elif depth_ratio < 0.1:
        signals.append('bearish')
        details['liquidity'] = 'Shallow liquidity relative to volume'
    else:
        signals.append('neutral')
        details['liquidity'] = 'Adequate liquidity'

    # Liquidity concentration
    concentration = liquidity_data['concentration_score']
    if concentration < 0.3:
        signals.append('bullish')
        details['concentration'] = 'Well-distributed liquidity'
    elif concentration > 0.7:
        signals.append('bearish')
        details['concentration'] = 'Highly concentrated liquidity'
    else:
        signals.append('neutral')
        details['concentration'] = 'Moderate liquidity concentration'

    # Impermanent loss risk
    il_risk = liquidity_data['impermanent_loss_risk']
    if il_risk < 0.1:
        signals.append('bullish')
        details['il_risk'] = 'Low impermanent loss risk'
    elif il_risk > 0.3:
        signals.append('bearish')
        details['il_risk'] = 'High impermanent loss risk'
    else:
        signals.append('neutral')
        details['il_risk'] = 'Moderate impermanent loss risk'

    return {
        "signals": signals,
        "details": details
    }

def combine_signals(
    technical: Dict[str, Any],
    onchain: Dict[str, Any],
    liquidity: Dict[str, Any]
) -> Dict[str, Any]:
    """Combine different signal types into overall trading signal."""
    
    all_signals = (
        technical["signals"] +
        onchain["signals"] +
        liquidity["signals"]
    )
    
    bullish_count = all_signals.count('bullish')
    bearish_count = all_signals.count('bearish')
    total_signals = len(all_signals)

    # Calculate overall signal
    if bullish_count > bearish_count:
        overall_signal = 'bullish'
        confidence = bullish_count / total_signals
    elif bearish_count > bullish_count:
        overall_signal = 'bearish'
        confidence = bearish_count / total_signals
    else:
        overall_signal = 'neutral'
        confidence = 0.5

    return {
        "signal": overall_signal,
        "confidence": confidence,
        "reasoning": {
            "technical": technical["details"],
            "onchain": onchain["details"],
            "liquidity": liquidity["details"]
        }
    }
async def sentiment_agent(state: AgentState):
    """Analyzes market sentiment from social media, news, and on-chain metrics."""
    data = state["data"]
    show_reasoning = state["metadata"]["show_reasoning"]

    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a crypto market sentiment analyst specializing in social and on-chain metrics.
            Analyze the following data points to determine market sentiment:
            - Social media engagement and sentiment
            - On-chain activity patterns
            - Whale wallet movements
            - Developer activity
            - Market maker behavior
            
            Provide your analysis in JSON format with:
            {
                "sentiment": "bullish" | "bearish" | "neutral",
                "confidence": <float between 0 and 1>,
                "reasoning": {
                    "social": <social media analysis>,
                    "onchain": <on-chain behavior analysis>,
                    "development": <developer activity analysis>,
                    "market_making": <market maker behavior analysis>
                }
            }"""
        ),
        (
            "human",
            """Analyze the sentiment for these tokens based on the provided metrics:
            Token Metrics: {token_metrics}
            On-chain Data: {onchain_data}
            Social Metrics: {social_metrics}
            """
        ),
    ])

    token_sentiments = {}
    
    for token in data["trading_pairs"]:
        # Gather sentiment data points
        social_metrics = await data_tools.get_social_metrics(token)
        github_activity = await data_tools.get_developer_activity(token)
        whale_activity = data["onchain_metrics"][token]["whale_movements"]
        
        prompt = template.invoke({
            "token_metrics": json.dumps(data["token_metrics"][token].__dict__),
            "onchain_data": json.dumps({
                "whale_activity": whale_activity,
                "developer_activity": github_activity
            }),
            "social_metrics": json.dumps(social_metrics)
        })
        
        result = await llm.ainvoke(prompt)
        token_sentiments[token] = json.loads(result.content)

    message_content = {
        "sentiments": token_sentiments,
        "timestamp": datetime.now().isoformat()
    }

    if show_reasoning:
        show_agent_reasoning(message_content, "Sentiment Analysis Agent")

    return {
        "messages": [HumanMessage(content=str(message_content), name="sentiment_agent")],
        "data": data
    }

async def risk_management_agent(state: AgentState):
    """Evaluates portfolio risk and sets position limits for crypto assets."""
    data = state["data"]
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = data["portfolio"]

    # Get messages from previous agents
    quant_message = next(msg for msg in state["messages"] if msg.name == "quant_agent")
    sentiment_message = next(msg for msg in state["messages"] if msg.name == "sentiment_agent")

    risk_assessments = {}
    
    for token in data["trading_pairs"]:
        # Calculate risk metrics
        volatility = calculate_token_volatility(data["historical_prices"][token])
        liquidity_risk = assess_liquidity_risk(data["liquidity_data"][token])
        correlation_risk = calculate_correlation_risk(
            data["historical_prices"],
            token,
            portfolio["tokens"]
        )
        #smart_contract_risk = assess_smart_contract_risk(token)
        
        # Determine position limits
        max_position = calculate_position_limit(
            token=token,
            portfolio_value=calculate_portfolio_value(portfolio, data["token_metrics"]),
            volatility=volatility,
            liquidity=data["liquidity_data"][token],
            correlation=correlation_risk
        )
        
        risk_assessments[token] = {
            "risk_score": calculate_risk_score(
                volatility,
                liquidity_risk,
                correlation_risk,
                #smart_contract_risk
            ),
            "max_position_size": max_position,
            "risk_factors": {
                "volatility": volatility,
                "liquidity_risk": liquidity_risk,
                "correlation_risk": correlation_risk,
                #"smart_contract_risk": smart_contract_risk
            }
        }

    message_content = {
        "risk_assessments": risk_assessments,
        "portfolio_recommendations": generate_portfolio_recommendations(
            risk_assessments,
            portfolio
        )
    }

    if show_reasoning:
        show_agent_reasoning(message_content, "Risk Management Agent")

    return {
        "messages": state["messages"] + [
            HumanMessage(content=str(message_content), name="risk_management_agent")
        ]
    }

async def portfolio_management_agent(state: AgentState):
    """Makes final trading decisions considering all signals and risk parameters."""
    data = state["data"]
    show_reasoning = state["metadata"]["show_reasoning"]
    portfolio = data["portfolio"]

    # Collect all agent signals
    quant_signals = json.loads(
        next(msg for msg in state["messages"] if msg.name == "quant_agent").content
    )
    sentiment_signals = json.loads(
        next(msg for msg in state["messages"] if msg.name == "sentiment_agent").content
    )
    risk_assessment = json.loads(
        next(msg for msg in state["messages"] if msg.name == "risk_management_agent").content
    )

    trading_decisions = {}
    
    for token in data["trading_pairs"]:
        # Generate optimal trade size
        trade_size = calculate_optimal_trade_size(
            token=token,
            portfolio=portfolio,
            signals=quant_signals["signals"][token],
            sentiment=sentiment_signals["sentiments"][token],
            risk_assessment=risk_assessment["risk_assessments"][token],
            liquidity=data["liquidity_data"][token]
        )
        
        # Determine best execution strategy
        execution_strategy = determine_execution_strategy(
            token=token,
            trade_size=trade_size,
            liquidity=data["liquidity_data"][token]
        )
        
        trading_decisions[token] = {
            "action": determine_trade_action(
                quant_signals["signals"][token],
                sentiment_signals["sentiments"][token],
                risk_assessment["risk_assessments"][token]
            ),
            "quantity": trade_size,
            "execution": execution_strategy,
            "reasoning": compile_trading_reasoning(
                token,
                quant_signals,
                sentiment_signals,
                risk_assessment
            )
        }

    message_content = {
        "trading_decisions": trading_decisions,
        "portfolio_update": calculate_expected_portfolio(
            portfolio,
            trading_decisions,
            data["token_metrics"]
        )
    }

    if show_reasoning:
        show_agent_reasoning(message_content, "Portfolio Management Agent")

    return {
        "messages": state["messages"] + [
            HumanMessage(content=str(message_content), name="portfolio_management")
        ]
    }

def determine_trade_action(
    quant_signal: str,
    sentiment: Dict[str, Any],
    risk_assessment: Dict[str, Any]
) -> str:
    """Determine the trade action based on signals, sentiment, and risk assessment."""
    if quant_signal == "bullish" and sentiment["sentiment"] == "bullish":
        return "buy"
    elif quant_signal == "bearish" and sentiment["sentiment"] == "bearish":
        return "sell"
    else:
        return "hold"

def calculate_execution_duration(self, trade_size: float, liquidity: Dict[str, Any]) -> float:
    """Calculate the duration for trade execution based on trade size and liquidity."""
    base_duration = 1  # Base duration in hours
    liquidity_factor = liquidity["depth_2percent"] / trade_size
    return base_duration / liquidity_factor

def compile_trading_reasoning(
    token: str,
    quant_signals: Dict[str, Any],
    sentiment_signals: Dict[str, Any],
    risk_assessment: Dict[str, Any]
) -> Dict[str, Any]:
    """Compile reasoning for trading decisions."""
    return {
        "quant_signals": quant_signals["signals"][token],
        "sentiment_signals": sentiment_signals["sentiments"][token],
        "risk_assessment": risk_assessment["risk_assessments"][token]
    }

def calculate_portfolio_value(portfolio: Dict[str, Any], token_metrics: Dict[str, TokenMetrics]) -> float:
    """Calculate the total value of the portfolio."""
    total_value = portfolio["cash"]
    for token, quantity in portfolio["tokens"].items():
        if quantity > 0:
            total_value += quantity * token_metrics[token].price
    return total_value

# Helper functions for risk and portfolio management

def calculate_parkinson_volatility(price_history: pd.DataFrame) -> float:
    """Calculate Parkinson volatility for a given price history."""
    log_high_low = np.log(price_history['high'] / price_history['low'])
    return np.sqrt((1 / (4 * len(log_high_low) * np.log(2))) * np.sum(log_high_low ** 2))
def calculate_garch_volatility(returns: pd.Series) -> float:
    """Calculate GARCH volatility for a given series of returns."""
    
    model = arch_model(returns, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp='off')
    return model_fit.conditional_volatility[-1]

def calculate_position_limit(
    token: str,
    portfolio_value: float,
    volatility: Dict[str, float],
    liquidity: Dict[str, Any],
    correlation: float
) -> float:
    """Calculate the maximum position limit for a token."""
    base_limit = portfolio_value * 0.1  # 10% of portfolio value
    volatility_adjustment = 1 / (1 + volatility["daily_vol"])
    liquidity_adjustment = liquidity["depth_2percent"] / 100000
    correlation_adjustment = 1 / (1 + correlation)
    
    return base_limit * volatility_adjustment * liquidity_adjustment * correlation_adjustment
    return {
        "daily_vol": returns.std() * np.sqrt(24),
        "parkinson_vol": calculate_parkinson_volatility(price_history),
        "garch_vol": calculate_garch_volatility(returns)
    }

def calculate_token_volatility(price_history: pd.DataFrame) -> Dict[str, float]:
    """Calculate token volatility using advanced metrics."""
    returns = price_history['close'].pct_change().dropna()
    return {
        "daily_vol": returns.std() * np.sqrt(24),
        "parkinson_vol": calculate_parkinson_volatility(price_history),
        "garch_vol": calculate_garch_volatility(returns)
    }

def assess_liquidity_risk(liquidity_data: Dict[str, Any]) -> Dict[str, float]:
    """Assess liquidity risk across multiple dimensions."""
    return {
        "depth_risk": 1 - min(1, liquidity_data['depth_2percent'] / 100000),
        "concentration_risk": liquidity_data['concentration_score'],
        "slippage_risk": 0.0  # Placeholder value, define estimate_slippage_risk function if needed
    }

def calculate_correlation_risk(
    price_histories: Dict[str, pd.DataFrame],
    token: str,
    portfolio_holdings: Dict[str, float]
) -> float:
    """Calculate portfolio correlation risk for a token."""
    correlations = {}
    token_returns = price_histories[token]['close'].pct_change().dropna()
    
    for other_token, holding in portfolio_holdings.items():
        if other_token != token and holding > 0:
            other_returns = price_histories[other_token]['close'].pct_change().dropna()
            correlations[other_token] = token_returns.corr(other_returns)
    
    return np.mean(list(correlations.values())) if correlations else 0


def calculate_risk_score(
    volatility: Dict[str, float],
    liquidity_risk: Dict[str, float],
    correlation_risk: float,
    smart_contract_risk: Dict[str, float]
) -> float:
    """Calculate overall risk score combining multiple risk factors."""
    weights = {
        "volatility": 0.3,
        "liquidity": 0.3,
        "correlation": 0.2,
        "smart_contract": 0.2
    }
    
    vol_score = (volatility["daily_vol"] + volatility["parkinson_vol"]) / 2
    liq_score = (liquidity_risk["depth_risk"] + liquidity_risk["concentration_risk"]) / 2
    sc_score = (
        smart_contract_risk["audit_score"] +
        smart_contract_risk["code_quality"]
    ) / 2
    
    return (
        weights["volatility"] * vol_score +
        weights["liquidity"] * liq_score +
        weights["correlation"] * correlation_risk +
        weights["smart_contract"] * sc_score
    )

def calculate_optimal_trade_size(
    token: str,
    portfolio: Dict[str, Any],
    signals: Dict[str, Any],
    sentiment: Dict[str, Any],
    risk_assessment: Dict[str, Any],
    liquidity: Dict[str, Any]
) -> float:
    """Calculate optimal trade size considering multiple factors."""
    base_size = min(
        risk_assessment["max_position_size"],
        liquidity["depth_2percent"] * 0.1
    )
    
    # Adjust size based on signals and sentiment
    confidence = (
        float(signals["confidence"]) +
        float(sentiment["confidence"])
    ) / 2
    
    return base_size * confidence

def determine_execution_strategy(
    token: str,
    trade_size: float,
    liquidity: Dict[str, Any]
) -> Dict[str, Any]:
    """Determine best execution strategy for a trade."""
    return {
        "method": "jupiter" if trade_size < liquidity["depth_2percent"] * 0.05 else "twap",
        "duration": calculate_execution_duration(trade_size, liquidity),
        "splits": calculate_optimal_splits(trade_size, liquidity),
        "routes": determine_optimal_routes(token, trade_size)
    }

def calculate_optimal_splits(trade_size: float, liquidity: Dict[str, Any]) -> List[float]:
    """Calculate optimal splits for a trade based on liquidity."""
    # Example implementation: split trade into equal parts
    num_splits = max(1, int(trade_size / (liquidity["depth_2percent"] * 0.1)))
    split_size = trade_size / num_splits
    return [split_size] * num_splits

def determine_optimal_routes(token: str, trade_size: float) -> List[Dict[str, Any]]:
    """Determine optimal routes for a trade."""
    # Example implementation: return a single route
    return [{"token": token, "size": trade_size, "route": "default"}]

class CryptoTradingSystem:
    def __init__(
        self,
        trading_pairs: List[str],
        initial_capital: float,
        risk_parameters: Optional[Dict[str, float]] = None,
        show_reasoning: bool = False
    ):
        self.trading_pairs = trading_pairs
        self.initial_capital = initial_capital
        self.show_reasoning = show_reasoning
        
        # Initialize default risk parameters if not provided
        self.risk_parameters = risk_parameters or {
            "max_position_size": 0.1,  # 10% of portfolio per position
            "max_slippage": 0.02,      # 2% maximum slippage
            "min_liquidity": 100000,   # Minimum liquidity requirement in USDC
            "rebalance_threshold": 0.05 # 5% deviation triggers rebalance
        }
        
        # Initialize components
        self.workflow = self._initialize_workflow()
        self.logger = self._setup_logger()
        
    def _initialize_workflow(self) -> StateGraph:
        """Initialize the agent workflow graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("market_data_agent", market_data_agent)
        workflow.add_node("quant_agent", quant_agent)
        workflow.add_node("sentiment_agent", sentiment_agent)
        workflow.add_node("risk_management_agent", risk_management_agent)
        workflow.add_node("portfolio_management_agent", portfolio_management_agent)
        
        # Define workflow
        workflow.set_entry_point("market_data_agent")
        workflow.add_edge("market_data_agent", "quant_agent")
        workflow.add_edge("market_data_agent", "sentiment_agent")
        workflow.add_edge("quant_agent", "risk_management_agent")
        workflow.add_edge("sentiment_agent", "risk_management_agent")
        workflow.add_edge("risk_management_agent", "portfolio_management_agent")
        workflow.add_edge("portfolio_management_agent", END)
        
        return workflow.compile()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("CryptoTrading")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def compile_trading_reasoning(
        self,
        token: str,
        quant_signals: Dict[str, Any],
        sentiment_signals: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile reasoning for trading decisions."""
        return {
            "quant_signals": quant_signals["signals"][token],
            "sentiment_signals": sentiment_signals["sentiments"][token],
            "risk_assessment": risk_assessment["risk_assessments"][token]
        }

    async def execute_trades(
        self,
        trading_decisions: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute trading decisions using Jupiter and track results."""
        execution_results = {}
        
        for token, decision in trading_decisions.items():
            try:
                if decision["action"] in ["buy", "sell"]:
                    # Prepare transaction parameters
                    tx_params = await self._prepare_transaction(token, decision)
                    
                    # Execute trade through Jupiter
                    result = await self._execute_jupiter_trade(tx_params)
                    
                    # Track execution results
                    execution_results[token] = {
                        "status": "success" if result["success"] else "failed",
                        "txid": result.get("signature"),
                        "executed_price": result.get("executed_price"),
                        "slippage": result.get("slippage"),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    self.logger.info(
                        f"Trade executed for {token}: {decision['action']} "
                        f"{decision['quantity']} @ {result.get('executed_price')}"
                    )
                else:
                    execution_results[token] = {
                        "status": "skipped",
                        "reason": "no action required",
                        "timestamp": datetime.now().isoformat()
                    }
            
            except Exception as e:
                self.logger.error(f"Trade execution failed for {token}: {str(e)}")
                execution_results[token] = {
                    "status": "failed",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
        
        return execution_results

    async def _prepare_transaction(
        self,
        token: str,
        decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare transaction parameters for Jupiter."""
        # Get current market data
        token_price = await self.data_tools.get_token_price(token)
        
        # Calculate optimal route
        route = await self._find_optimal_route(
            token=token,
            amount=decision["quantity"],
            side=decision["action"]
        )
        
        return {
            "token": token,
            "action": decision["action"],
            "quantity": decision["quantity"],
            "route": route,
            "max_slippage": self.risk_parameters["max_slippage"],
            "execution_strategy": decision["execution"]
        }

    async def run_trading_cycle(self) -> Dict[str, Any]:
        """Execute one complete trading cycle."""
        try:
            # Initialize state
            initial_state = {
                "messages": [],
                "data": {
                    "trading_pairs": self.trading_pairs,
                    "portfolio": {
                        "cash": self.initial_capital,
                        "tokens": {pair: 0 for pair in self.trading_pairs}
                    },
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "show_reasoning": self.show_reasoning,
                    "risk_parameters": self.risk_parameters
                }
            }
            
            # Run agent workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Extract trading decisions
            trading_decisions = json.loads(
                next(
                    msg.content for msg in final_state["messages"]
                    if msg.name == "portfolio_management"
                )
            )
            
            # Execute trades
            execution_results = await self.execute_trades(
                trading_decisions["trading_decisions"]
            )
            
            # Update portfolio state
            updated_portfolio = await self._update_portfolio_state(
                execution_results
            )
            
            return {
                "trading_decisions": trading_decisions,
                "execution_results": execution_results,
                "updated_portfolio": updated_portfolio,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Trading cycle failed: {str(e)}")
            raise

    async def _update_portfolio_state(
        self,
        execution_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Update portfolio state after trade execution."""
        portfolio = {
            "cash": self.initial_capital,
            "tokens": {pair: 0 for pair in self.trading_pairs},
            "total_value": 0,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update positions based on execution results
        for token, result in execution_results.items():
            if result["status"] == "success":
                quantity = float(result["executed_quantity"])
                price = float(result["executed_price"])
                
                if result.get("action") == "buy":
                    portfolio["tokens"][token] += quantity
                    portfolio["cash"] -= quantity * price
                else:
                    portfolio["tokens"][token] -= quantity
                    portfolio["cash"] += quantity * price
        
        # Calculate total portfolio value
        portfolio["total_value"] = portfolio["cash"]
        for token, quantity in portfolio["tokens"].items():
            if quantity > 0:
                current_price = await self.data_tools.get_token_price(token)
                portfolio["total_value"] += quantity * current_price
        
        return portfolio

def calculate_expected_portfolio(
    portfolio: Dict[str, Any],
    trading_decisions: Dict[str, Any],
    token_metrics: Dict[str, TokenMetrics]
) -> Dict[str, Any]:
    """Calculate the expected portfolio after executing trading decisions."""
    updated_portfolio = portfolio.copy()
    for token, decision in trading_decisions.items():
        if decision["action"] == "buy":
            updated_portfolio["tokens"][token] += decision["quantity"]
            updated_portfolio["cash"] -= decision["quantity"] * token_metrics[token].price
        elif decision["action"] == "sell":
            updated_portfolio["tokens"][token] -= decision["quantity"]
            updated_portfolio["cash"] += decision["quantity"] * token_metrics[token].price
    updated_portfolio["total_value"] = updated_portfolio["cash"]
    for token, quantity in updated_portfolio["tokens"].items():
        if quantity > 0:
            updated_portfolio["total_value"] += quantity * token_metrics[token].price
    return updated_portfolio

def generate_portfolio_recommendations(
    risk_assessments: Dict[str, Any],
    portfolio: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate portfolio recommendations based on risk assessments."""
    recommendations = {}
    for token, assessment in risk_assessments.items():
        max_position = assessment["max_position_size"]
        current_position = portfolio["tokens"].get(token, 0)
        if current_position > max_position:
            recommendations[token] = {
                "action": "reduce",
                "quantity": current_position - max_position
            }
        elif current_position < max_position:
            recommendations[token] = {
                "action": "increase",
                "quantity": max_position - current_position
            }
        else:
            recommendations[token] = {
                "action": "hold",
                "quantity": 0
            }
    return recommendations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run crypto trading system')
    parser.add_argument(
        '--trading_pairs',
        type=str,
        nargs='+',
        help='List of trading pairs (e.g., SOL BONK JUP)'
    )
    parser.add_argument(
        '--initial_capital',
        type=float,
        default=10000,
        help='Initial capital in USDC'
    )
    parser.add_argument(
        '--show_reasoning',
        action='store_true',
        help='Show detailed agent reasoning'
    )
    
    args = argparse.ArgumentParser()
    
    # Initialize and run trading system
    trading_system = CryptoTradingSystem(
        trading_pairs=args.trading_pairs,
        initial_capital=args.initial_capital,
        show_reasoning=args.show_reasoning
    )
    
    asyncio.run(trading_system.run_trading_cycle())