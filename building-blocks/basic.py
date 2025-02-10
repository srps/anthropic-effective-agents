import os
import httpx
from httpx._types import HeaderTypes, RequestData
from dotenv import load_dotenv

load_dotenv()

# Using Groq as it provides a mostly OpenAI compatible API on a free tier.
# We're also crafting the HTTP request instead of using the OpenAI library here for demonstration purposes.
# We could also use OpenAI's library, passing Groq's URL and API key. Maybe I'll add that later.
api_key = os.environ.get("GROQ_API_KEY")
url = f"https://api.groq.com/openai/v1/chat/completions"

headers: HeaderTypes = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}

data: RequestData = {
    "model": "deepseek-r1-distill-llama-70b", # These Deepseek R1 distilled models are interesting
    "messages": [{"role": "user", "content": "What are the basics of building effective AI agents?"}],
    "temperature": 0.6,
}

with httpx.Client() as client:
    response = client.post(url, headers=headers, json=data)

response.raise_for_status()

response_message = response.json()["choices"][0]["message"]["content"]

print(response_message)