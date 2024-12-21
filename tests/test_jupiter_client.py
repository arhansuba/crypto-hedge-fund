import pytest
from aioresponses import aioresponses
from src.executors.jupiter_client import JupiterClient, TOKEN_MINTS

# Mock API responses
MOCK_QUOTE_RESPONSE = {
    "inputMint": TOKEN_MINTS['SOL'],
    "outputMint": TOKEN_MINTS['USDC'],
    "inAmount": "1000000000",  # 1 SOL
    "outAmount": "20000000",   # 20 USDC
    "otherAmountThreshold": "19800000",
    "swapMode": "ExactIn",
    "slippageBps": 50,
    "priceImpactPct": "0.1",
    "routePlan": []
}

MOCK_SWAP_RESPONSE = {
    "swapTransaction": "base64_encoded_transaction",
    "lastValidBlockHeight": 123456789
}

MOCK_INSTRUCTIONS_RESPONSE = {
    "instructions": [
        {
            "programId": "string",
            "accounts": ["string"],
            "data": "base64_string"
        }
    ],
    "signers": [
        {
            "publicKey": "string",
            "isSigner": True
        }
    ],
    "address": {
        "associatedTokenAddress": "string"
    }
}

@pytest.fixture
async def jupiter_client():
    """Create a JupiterClient instance for testing."""
    client = JupiterClient()
    yield client
    await client.close()

@pytest.mark.asyncio
async def test_get_quote(jupiter_client):
    """Test getting a quote from Jupiter."""
    with aioresponses() as m:
        # Mock the quote API endpoint
        m.get(
            f"{jupiter_client.base_url}/quote?inputMint={TOKEN_MINTS['SOL']}&outputMint={TOKEN_MINTS['USDC']}&amount=1000000000&slippageBps=50&swapMode=ExactIn",
            payload=MOCK_QUOTE_RESPONSE
        )

        quote = await jupiter_client.get_quote(
            input_mint=TOKEN_MINTS['SOL'],
            output_mint=TOKEN_MINTS['USDC'],
            amount=1000000000,
            slippage_bps=50
        )

        assert quote is not None
        assert quote['inputMint'] == TOKEN_MINTS['SOL']
        assert quote['outputMint'] == TOKEN_MINTS['USDC']
        assert quote['inAmount'] == "1000000000"
        assert quote['outAmount'] == "20000000"

@pytest.mark.asyncio
async def test_get_swap_transaction(jupiter_client):
    """Test getting swap transaction."""
    with aioresponses() as m:
        # Mock the swap API endpoint
        m.post(
            f"{jupiter_client.base_url}/swap",
            payload=MOCK_SWAP_RESPONSE
        )

        user_public_key = "test_public_key"
        swap_tx = await jupiter_client.get_swap_transaction(
            quote_response=MOCK_QUOTE_RESPONSE,
            user_public_key=user_public_key
        )

        assert swap_tx is not None
        assert swap_tx['swapTransaction'] == "base64_encoded_transaction"
        assert swap_tx['lastValidBlockHeight'] == 123456789

@pytest.mark.asyncio
async def test_get_swap_instructions(jupiter_client):
    """Test getting swap instructions."""
    with aioresponses() as m:
        # Mock the swap-instructions API endpoint
        m.post(
            f"{jupiter_client.base_url}/swap-instructions",
            payload=MOCK_INSTRUCTIONS_RESPONSE
        )

        user_public_key = "test_public_key"
        instructions = await jupiter_client.get_swap_instructions(
            quote_response=MOCK_QUOTE_RESPONSE,
            user_public_key=user_public_key
        )

        assert instructions is not None
        assert 'instructions' in instructions
        assert 'signers' in instructions
        assert 'address' in instructions

@pytest.mark.asyncio
async def test_execute_swap(jupiter_client):
    """Test executing a complete swap."""
    with aioresponses() as m:
        # Mock all required endpoints
        m.get(
            f"{jupiter_client.base_url}/quote?inputMint={TOKEN_MINTS['SOL']}&outputMint={TOKEN_MINTS['USDC']}&amount=1000000000&slippageBps=50&swapMode=ExactIn",
            payload=MOCK_QUOTE_RESPONSE
        )
        m.post(
            f"{jupiter_client.base_url}/swap",
            payload=MOCK_SWAP_RESPONSE
        )

        result = await jupiter_client.execute_swap(
            input_token='SOL',
            output_token='USDC',
            amount="1000000000",
            user_public_key="test_public_key",
            slippage_bps=50
        )

        assert result['success'] is True
        assert result['input_token'] == 'SOL'
        assert result['output_token'] == 'USDC'
        assert result['amount_in'] == MOCK_QUOTE_RESPONSE['inAmount']
        assert result['amount_out'] == MOCK_QUOTE_RESPONSE['outAmount']
        assert result['tx_hash'] == MOCK_SWAP_RESPONSE['swapTransaction']

@pytest.mark.asyncio
async def test_error_handling(jupiter_client):
    """Test error handling in the Jupiter client."""
    with aioresponses() as m:
        # Mock failed quote request
        m.get(
            f"{jupiter_client.base_url}/quote?inputMint={TOKEN_MINTS['SOL']}&outputMint={TOKEN_MINTS['USDC']}&amount=1000000000&slippageBps=50&swapMode=ExactIn",
            status=400,
            body="Invalid request"
        )

        quote = await jupiter_client.get_quote(
            input_mint=TOKEN_MINTS['SOL'],
            output_mint=TOKEN_MINTS['USDC'],
            amount=1000000000,
            slippage_bps=50
        )

        assert quote is None

        # Test failed swap execution
        result = await jupiter_client.execute_swap(
            input_token='SOL',
            output_token='USDC',
            amount="1000000000",
            user_public_key="test_public_key",
            slippage_bps=50
        )

        assert result['success'] is False
        assert 'error' in result

@pytest.mark.asyncio
async def test_token_mint_resolution(jupiter_client):
    """Test token mint address resolution."""
    assert jupiter_client._get_token_mint('SOL') == TOKEN_MINTS['SOL']
    assert jupiter_client._get_token_mint('USDC') == TOKEN_MINTS['USDC']
    assert jupiter_client._get_token_mint('UNKNOWN') == 'UNKNOWN'

@pytest.mark.asyncio
async def test_session_management(jupiter_client):
    """Test session management."""
    assert jupiter_client.session is None
    
    # Test session creation
    await jupiter_client.ensure_session()
    assert jupiter_client.session is not None
    
    # Test session reuse
    original_session = jupiter_client.session
    await jupiter_client.ensure_session()
    assert jupiter_client.session is original_session
    
    # Test session closure
    await jupiter_client.close()
    assert jupiter_client.session is None