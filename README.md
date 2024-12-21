# AI Crypto Trading System

An intelligent trading system that combines artificial intelligence with Solana's blockchain infrastructure to execute data-driven trading strategies. The system leverages Jupiter's pricing engine and Helius's comprehensive blockchain data to make informed trading decisions.

## Overview

This system functions as an autonomous trading platform, utilizing multiple AI agents to analyze market conditions and execute trades on Solana-based tokens. By combining on-chain data analysis with technical indicators, the system provides a sophisticated approach to cryptocurrency trading.

## Core Infrastructure

The system is built on two primary data providers:

Jupiter Protocol provides:
- Real-time token pricing
- Order book depth analysis
- Price impact calculations
- Historical price data

Helius delivers:
- Comprehensive token metrics
- Transaction monitoring
- Holder statistics
- Network activity data

## System Architecture

The trading system operates through five specialized components:

The Market Data Agent gathers and processes data from Jupiter and Helius, creating a comprehensive market view for analysis.

The Quantitative Analysis Agent performs technical analysis using this data, calculating various indicators and identifying trading opportunities.

The Sentiment Analysis Agent evaluates market sentiment using on-chain metrics and holder behavior patterns.

The Risk Management Agent sets position limits and evaluates trading risks based on liquidity and market conditions.

The Portfolio Management Agent makes final trading decisions and manages overall portfolio allocation.

## Getting Started

Initial setup requires the following steps:

```bash
# Clone the repository
git clone https://github.com/arhansuba/crypto-hedge-fund.git
cd crypto-trading-system

# Set up Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
```

## Configuration

Create a `.env` file with your API credentials:

```env
HELIUS_API_KEY=your_helius_api_key
JUPITER_API_KEY=your_jupiter_api_key
OPENAI_API_KEY=your_openai_api_key
```

## Usage

Launch the trading system with specific parameters:

```bash
python src/agents.py --trading_pairs SOL BONK JUP --initial_capital 10000 --show_reasoning
python src/agents.py --capital 10000 --pairs SOL BONK --risk 0.7 --dry-run --interval 60
```

Command line parameters include:
- trading_pairs: Tokens to include in trading strategy
- initial_capital: Starting capital in USDC
- show_reasoning: Enable detailed analysis logging

## Data Analysis Tools

The system includes specialized tools for market analysis:

CryptoDataTools handles data retrieval and processing, interfacing with Jupiter and Helius APIs for comprehensive market information.

CryptoTechnicalAnalysis provides advanced technical indicators specifically calibrated for cryptocurrency markets.

LiquidityAnalysis evaluates market depth and trading impact, essential for optimal trade execution.

## Development

For local development and testing:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run test suite
pytest tests/

# Execute backtesting
python src/backtester.py --trading_pairs SOL --start_date 2024-01-01
```

## Project Structure

```
crypto-trading-system/
├── src/
│   ├── agents.py           # AI agent implementations
│   ├── backtester.py       # Strategy testing framework
│   └── tools.py            # Market analysis tools
├── tests/
│   └── test_agents.py      # Unit tests
└── config/
    └── .env.example        # Configuration template
```

## Security Considerations

Implement these security measures:
- Store API keys in secure environment variables
- Set appropriate position limits
- Monitor trading volumes against liquidity
- Maintain secure key management
- Regular system audits

## Risk Management

The system incorporates several risk control measures:
- Dynamic position sizing based on liquidity
- Slippage protection mechanisms
- Portfolio correlation analysis
- Continuous market monitoring

## Support and Maintenance

For technical support:
- Submit issues through GitHub
- Consult API documentation: 
  - Jupiter: https://station.jup.ag/docs
  - Helius: https://docs.helius.dev/

## Legal Considerations

This software is provided for educational purposes only. Cryptocurrency trading involves significant risks. Users should conduct thorough research and risk assessment before implementation.

## Contact Information

For technical queries:
- GitHub Issues


## Acknowledgments

Special thanks to:
- Jupiter Protocol for comprehensive pricing infrastructure
- Helius for robust blockchain data services
- The Solana Foundation for blockchain support

## Future Development

Planned enhancements include:
- Additional technical indicators
- Enhanced risk management features
- Expanded portfolio optimization tools
- Advanced backtesting capabilities