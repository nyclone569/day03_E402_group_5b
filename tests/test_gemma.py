import unittest
import sys
import os

# Add the project root directory to the Python path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch, MagicMock
from src.core.nvidia_provider import NVIDIAProvider
import httpx

class TestNVIDIAProvider(unittest.TestCase):
    def setUp(self):
        self.provider = NVIDIAProvider(api_key="fake-api-key")
        self.messages = [{"role": "user", "content": "How are you?"}]

    @patch('src.core.nvidia_provider.httpx.Client.post')
    def test_chat_completion_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "I am Gemma 4.",
                        "thought": "Generating response..."
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        result = self.provider.chat_completion(messages=self.messages)

        self.assertIn("choices", result)
        self.assertEqual(result["choices"][0]["message"]["content"], "I am Gemma 4.")
        self.assertEqual(result["choices"][0]["message"]["thought"], "Generating response...")
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(kwargs['json']['model'], "nvidia/gemma-4-31b-it-nvfp4")
        self.assertTrue(kwargs['json']['thinking']['enabled'])

    @patch('src.core.nvidia_provider.httpx.Client.post')
    def test_chat_completion_http_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.text = "Unauthorized"
        mock_post.side_effect = httpx.HTTPStatusError(
            message="401 Unauthorized", 
            request=MagicMock(), 
            response=mock_response
        )

        result = self.provider.chat_completion(messages=self.messages)

        self.assertIn("error", result)
        self.assertIn("NVIDIA API Error: Unauthorized", result["error"])

if __name__ == '__main__':
    unittest.main()
