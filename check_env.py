# check_env.py
import os
import sys
import ssl
import logging
import aiohttp
import asyncio
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def check_jupiter_connection():
    """Test connection to Jupiter API."""
    url = "https://quote-api.jup.ag/v6/price"
    params = {
        "inputMint": "So11111111111111111111111111111111111111112",  # SOL
        "outputMint": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"  # USDC
    }
    
    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    logger.info("✅ Jupiter API connection successful")
                    return True
                else:
                    logger.error(f"❌ Jupiter API error: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ Jupiter API connection failed: {e}")
            return False

async def check_gaianet_connection():
    """Test connection to GaiaNet API."""
    url = "https://raw.gaianet.ai/llama-3-8b-instruct/config.json"
    
    ssl_context = ssl.create_default_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    
    async with aiohttp.ClientSession(connector=connector) as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    logger.info("✅ GaiaNet API connection successful")
                    return True
                else:
                    logger.error(f"❌ GaiaNet API error: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"❌ GaiaNet API connection failed: {e}")
            return False

def check_environment():
    """Check if all required environment variables are set."""
    load_dotenv()
    
    required_vars = [
        "OPENAI_API_KEY",
        "HELIUS_API_KEY",
        "GAIANET_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ Environment variables check passed")
    return True

async def main():
    """Run all environment checks."""
    logger.info("Running environment checks...")
    
    checks = [
        ("Environment variables", check_environment()),
        ("Jupiter API connection", await check_jupiter_connection()),
        ("GaiaNet API connection", await check_gaianet_connection())
    ]
    
    all_passed = all(result for _, result in checks)
    
    if all_passed:
        logger.info("✅ All checks passed! You can run the agent now.")
        return 0
    else:
        logger.error("❌ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Check cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during checks: {e}")
        sys.exit(1)