import os
import json
import httpx
from typing import List, Dict, Any, Optional

class NVIDIAProvider:
    """
    Provider xử lý kết nối tới NVIDIA NIM API cho dòng model Gemma 4.
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = "https://integrate.api.nvidia.com/v1" or os.getenv("NVIDIA_BASE_URL")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "nvidia/gemma-4-31b-it-nvfp4",
        temperature: float = 0.6,
        max_tokens: int = 4096,
        enable_thinking: bool = True
    ) -> Dict[str, Any]:
        """
        Gửi yêu cầu đến Gemma 4 với hỗ trợ chế độ Thinking (Suy nghĩ).
        """
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": 0.7,
            "max_tokens": max_tokens,
            # Cấu hình đặc thù cho Gemma 4 Reasoning
            "thinking": {
                "enabled": enable_thinking,
                "include_thoughts": True
            },
            "stream": False
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPStatusError as e:
            return {"error": f"NVIDIA API Error: {e.response.text}"}
        except Exception as e:
            return {"error": f"Unexpected Error: {str(e)}"}

# --- Ví dụ sử dụng ---
if __name__ == "__main__":
    provider = NVIDIAProvider()
    chat_history = [
        {"role": "user", "content": "Tại sao Gemma 4 lại vượt trội trong việc suy luận logic?"}
    ]
    
    result = provider.chat_completion(messages=chat_history)
    
    if "error" in result:
        print(result["error"])
    else:
        # Tách biệt phần suy nghĩ và câu trả lời (nếu có)
        choice = result['choices'][0]['message']
        if 'thought' in choice:
            print(f"--- THOUGHT ---\n{choice['thought']}\n")
        print(f"--- ANSWER ---\n{choice['content']}")