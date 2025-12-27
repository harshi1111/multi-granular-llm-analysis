"""
REAL LLM Query Module - Groq API (Actually Free)
"""
import os
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime


class LLMQueryModule:
    """
    REAL LLM queries using Groq API - FREE credits.
    """
   
    def __init__(self, api_keys: Optional[Dict] = None):
        self.api_keys = api_keys or {}
        self.available_models = []
        self._initialize_clients()
   
    def _initialize_clients(self):
        """Initialize Groq client."""
        if "groq" in self.api_keys and self.api_keys["groq"]:
            # Groq's free models
            self.available_models = ["Groq-Llama3.3-70B"]
            print(f"✅ Groq API initialized. Models: {self.available_models}")
            print("🚀 FREE credits available - REAL responses")
        else:
            print("⚠️ Groq API key not found. Add GROQ_API_KEY to .env")
            self.available_models = ["Groq-Llama3.3-70B"]
   
    def query(self, prompt: str, model: str = "Groq-Llama3.3-70B",
              temperature: float = 0.3, max_tokens: int = 300) -> Dict:
        """
        Query REAL Groq API.
        """
        print(f"🚀 Querying {model} with: '{prompt[:50]}...'")
       
        # Map model names to Groq model IDs
        model_map = {
             "Groq-Llama3.3-70B": "llama-3.3-70b-versatile"
        }
       
        groq_model = model_map.get(model, "llama-3.1-70b-versatile")
       
        url = "https://api.groq.com/openai/v1/chat/completions"
       
        headers = {
            "Authorization": f"Bearer {self.api_keys.get('groq', '')}",
            "Content-Type": "application/json"
        }
       
        data = {
            "model": groq_model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
       
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
           
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
               
                print(f"✅ Got REAL response: {len(text)} characters")
               
                return {
                    "model": model,
                    "response": text,
                    "timestamp": datetime.now().isoformat(),
                    "provider": "Groq",
                    "is_real": True
                }
            else:
                error_msg = f"API error {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
               
                return {
                    "model": model,
                    "error": error_msg,
                    "response": "",
                    "timestamp": datetime.now().isoformat(),
                    "provider": "Groq"
                }
               
        except Exception as e:
            error_msg = f"Connection failed: {str(e)}"
            print(f"❌ {error_msg}")
           
            return {
                "model": model,
                "error": error_msg,
                "response": "",
                "timestamp": datetime.now().isoformat(),
                "provider": "Groq"
            }


class PromptDataset:
    """Not needed."""
    def __init__(self):
        self.prompts = []
   
    def get_prompts(self, n: Optional[int] = None) -> List[str]:
        return [] 