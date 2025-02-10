import json
import os
import sys
from typing import List, Optional
import httpx
from httpx._types import HeaderTypes, RequestData
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

class CryptoRate(BaseModel):
    id: str = Field(..., description="The unique identifier of the currency, e.g., 'bitcoin'")
    symbol: str = Field(..., description="The symbol of the currency, e.g., 'BTC'")
    currencySymbol: Optional[str] = Field(..., description="The symbol of the currency, e.g., '₿'")
    type: str = Field(..., description="The type of the currency, e.g., 'crypto'")
    rateUsd: float = Field(..., description="The current USD exchange rate of the currency")

class CryptoRateResponse(BaseModel):
    data: CryptoRate
    timestamp: int = Field(..., description="The timestamp of the response in Unix time")

class CryptoRateTool(BaseModel):
    currency: str = Field(..., description="The currency id (e.g., 'bitcoin')")

def get_crypto_rate(currency: str) -> CryptoRate:
    """Get the current exchange rate of a cryptocurrency."""
    url = f"https://api.coincap.io/v2/rates/{currency}"
    response = httpx.get(url)
    response.raise_for_status()
    data = response.json()["data"]
    return CryptoRate(
        id=data.get("id"),
        symbol=data.get("symbol"),
        currencySymbol=data.get("currencySymbol"),
        type=data.get("type"),
        rateUsd=data.get("rateUsd"),
    )

api_key = os.environ.get("GROQ_API_KEY")
model = "llama-3.3-70b-versatile"
url = f"https://api.groq.com/openai/v1/chat/completions"

headers: HeaderTypes = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

def run_conversation(user_input: str) -> str:
    messages = [
        {
            "role": "system", "content": "You are an exchange rate assistant. Use the get_crypto_rate tool to get the current exchange rate of a currency."
        },
        {
            "role": "user", "content": user_input
        }
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_crypto_rate",
                "description": "Get the current exchange rate of a currency.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "currency": {
                            "type": "string",
                            "description": "The currency id (e.g., 'bitcoin')",
                        }
                    },
                    "required": ["currency"],
                },
            },
        }
    ]

    data: RequestData = {
        "model": f"{model}",
        "messages": messages,
        "temperature": 0.6,
        "tools": tools,
        "tool_choice": "auto",
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)
        response.raise_for_status()

        response_message = response.json()["choices"][0]["message"]
        tool_calls = response_message["tool_calls"]

        available_tools = {
            get_crypto_rate.__name__: get_crypto_rate
        }

        if tool_calls:
            for tool_call in tool_calls:
                print(f"Calling tool: {tool_call['function']['name']}")
                print(f"Arguments: {tool_call['function']['arguments']}")
                function_args = tool_call["function"]["arguments"]
                function_args_dict = json.loads(function_args)
                tool_name = tool_call["function"]["name"]
                tool = available_tools[tool_name]
                tool_response = tool(**function_args_dict)

                messages.append(
                    {
                        "tool_call_id": tool_call["id"],
                        "role": "tool",
                        "name": tool_name,
                        "content": tool_response.model_dump_json(),
                    }
                )
                print(f"Tool response: {tool_response}")
                response_message = client.post(
                    url,
                    headers=headers,
                    json={
                        "model": f"{model}",
                        "messages": messages,
                        "temperature": 0.6,
                    },
                )
                response_message.raise_for_status()
                response_message = response_message.json()["choices"][0]["message"]


        return response_message["content"]

# response_message = run_conversation("What is the current exchange rate of Bitcoin?")


# print(response_message)

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python tools.py <currency>")
        print("Running with default currency: bitcoin")
        currency = "bitcoin"
    else:
        currency = args[0]

    print(f"Getting exchange rate for {currency}")
    response_message = run_conversation(f"What is the current exchange rate of {currency}?")
    print(response_message)