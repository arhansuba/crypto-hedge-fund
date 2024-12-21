# src/config.py
import os
from typing import Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class ChainConfig:
    chain_id: str
    name: str
    rpc_url: str
    native_token: str
    block_time: float
    explorer_url: str

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    temperature: float = 0.7
    max_tokens: int = 1000

@dataclass
class TradingConfig:
    max_position_size: float
    risk_tolerance: float
    min_confidence: float
    slippage_tolerance: float = 0.01

@dataclass
class GaiaNetConfig:
    config_url: str

class Config:
    # Chain configurations
    CHAIN_CONFIGS = {
        "solana": ChainConfig(
            chain_id="solana",
            name="Solana",
            rpc_url=os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"),
            native_token="SOL",
            block_time=0.4,
            explorer_url="https://solscan.io"
        ),
        "ethereum": ChainConfig(
            chain_id="ethereum",
            name="Ethereum",
            rpc_url=os.getenv("ETH_RPC_URL", ""),
            native_token="ETH",
            block_time=12,
            explorer_url="https://etherscan.io"
        )
    }

    # LLM configurations
    LLM_CONFIGS = {
        "openai": LLMConfig(
            provider="openai",
            model="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY", "")
        ),
        "anthropic": LLMConfig(
            provider="anthropic",
            model="claude-3-opus-20240229",
            api_key=os.getenv("ANTHROPIC_API_KEY", "")
        ),
        "groq": LLMConfig(
            provider="groq",
            model="mixtral-8x7b-32768",
            api_key=os.getenv("GROQ_API_KEY", "")
        )
    }

    # DEX configurations by chain
    DEX_CONFIGS = {
        "solana": {
            "jupiter": {
                "api_url": "https://quote-api.jup.ag/v6",
                "api_key": os.getenv("JUPITER_API_KEY", "")
            },
            "orca": {
                "api_url": "https://api.orca.so",
            }
        },
        "ethereum": {
            "uniswap": {
                "api_url": "https://api.uniswap.org/v2",
            },
            "sushiswap": {
                "api_url": "https://api.sushi.com",
            }
        }
    }

    # GaiaNet configuration
    GAIANET_CONFIG = GaiaNetConfig(
        config_url="https://raw.gaianet.ai/llama-3-8b-instruct/config.json"
    )

    @staticmethod
    def get_chain_config(chain_id: str) -> Optional[ChainConfig]:
        return Config.CHAIN_CONFIGS.get(chain_id)

    @staticmethod
    def get_llm_config(provider: str) -> Optional[LLMConfig]:
        return Config.LLM_CONFIGS.get(provider)

    @staticmethod
    def get_dex_config(chain_id: str, dex_name: str) -> Optional[Dict]:
        chain_dexes = Config.DEX_CONFIGS.get(chain_id, {})
        return chain_dexes.get(dex_name)

    @staticmethod
    def get_trading_config() -> TradingConfig:
        return TradingConfig(
            max_position_size=float(os.getenv("MAX_POSITION_SIZE", "10000")),
            risk_tolerance=float(os.getenv("RISK_TOLERANCE", "0.7")),
            min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.7"))
        )

    @staticmethod
    def get_gaianet_config() -> GaiaNetConfig:
        return Config.GAIANET_CONFIG