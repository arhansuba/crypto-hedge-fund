import unittest
from unittest.mock import AsyncMock, patch
import sys
import os

# Add the directory containing agents.py to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents import get_swap_instructions

class TestAgents(unittest.TestCase):
    
    @patch('agents.get_swap_instructions.ensure_session', new_callable=AsyncMock)
    @patch('agents.get_swap_instructions.session.post', new_callable=AsyncMock)
    async def test_get_swap_instructions_success(self, mock_post, mock_ensure_session):
        # Arrange
        mock_post.return_value.__aenter__.return_value.status = 200
        mock_post.return_value.__aenter__.return_value.json = AsyncMock(return_value={"data": "test"})
        
        user_public_key = "test_public_key"
        wrap_unwrap_sol = True
        use_shared_accounts = False
        quote_response = {"quote": "test_quote"}
        compute_unit_price_micro_lamports = 1000
        
        # Act
        result = await get_swap_instructions(user_public_key, wrap_unwrap_sol, use_shared_accounts, quote_response, compute_unit_price_micro_lamports)
        
        # Assert
        self.assertEqual(result, {"data": "test"})
        mock_ensure_session.assert_called_once()
        mock_post.assert_called_once_with(
            f"{get_swap_instructions.base_url}/swap-instructions",
            json={
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": wrap_unwrap_sol,
                "useSharedAccounts": use_shared_accounts,
                "quoteResponse": quote_response,
                "computeUnitPriceMicroLamports": compute_unit_price_micro_lamports
            }
        )
    
    @patch('agents.get_swap_instructions.ensure_session', new_callable=AsyncMock)
    @patch('agents.get_swap_instructions.session.post', new_callable=AsyncMock)
    async def test_get_swap_instructions_error(self, mock_post, mock_ensure_session):
        # Arrange
        mock_post.return_value.__aenter__.return_value.status = 400
        mock_post.return_value.__aenter__.return_value.text = AsyncMock(return_value="Error message")
        
        user_public_key = "test_public_key"
        wrap_unwrap_sol = True
        use_shared_accounts = False
        quote_response = {"quote": "test_quote"}
        compute_unit_price_micro_lamports = 1000
        
        # Act
        result = await get_swap_instructions(user_public_key, wrap_unwrap_sol, use_shared_accounts, quote_response, compute_unit_price_micro_lamports)
        
        # Assert
        self.assertIsNone(result)
        mock_ensure_session.assert_called_once()
        mock_post.assert_called_once_with(
            f"{get_swap_instructions.base_url}/swap-instructions",
            json={
                "userPublicKey": user_public_key,
                "wrapAndUnwrapSol": wrap_unwrap_sol,
                "useSharedAccounts": use_shared_accounts,
                "quoteResponse": quote_response,
                "computeUnitPriceMicroLamports": compute_unit_price_micro_lamports
            }
        )

if __name__ == '__main__':
    unittest.main()
