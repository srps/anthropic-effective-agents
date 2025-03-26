import os
from enum import Enum
from typing import Optional
import httpx
from httpx._types import HeaderTypes, RequestData
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass

load_dotenv()


@dataclass
class AgentType(str, Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    RECOMMENDATIONS = "recommendations"
    UNKNOWN = "unknown"
    IRRELEVANT = "irrelevant"


class RouterResponse(BaseModel):
    agent_type: AgentType
    confidence: float = Field(ge=0.0, le=1.0)
    needs_clarification: bool
    clarification_question: Optional[str] = None
    response_to_user: str = Field(
        description="Response to send to the user, especially important for irrelevant queries or when clarification is needed"
    )


class CallRouter:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.router_model = "deepseek-r1-distill-llama-70b"
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers: HeaderTypes = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def route_query(self, query: str) -> RouterResponse:
        """Route the query to the appropriate agent"""
        data: RequestData = {
            "model": self.router_model,
            "messages": [
                {
                    "role": "system",
                    "content": f"""You are a query router that classifies user queries and provides appropriate responses.\n

                        Rules for classification:
                        1. Technical issues (code problems, errors, system issues) → TECHNICAL agent
                        2. Billing/payment issues (charges, invoices, subscriptions) → BILLING agent
                        3. Product suggestions/recommendations → RECOMMENDATIONS agent
                        4. If the query is completely unrelated to technical support, billing, or recommendations, mark as IRRELEVANT
                        5. If the query is related but unclear, mark as UNKNOWN and request clarification
                        
                        For each query, you must:
                        1. Determine the appropriate agent type
                        2. Set a confidence score (0.0 to 1.0)
                        3. Determine if clarification is needed
                        4. Provide a clear response message to the user
                        
                        For IRRELEVANT queries:
                        - Politely explain that you can only help with technical support, billing issues, or product recommendations
                        - Provide examples of questions you can help with
                        
                        For UNKNOWN queries:
                        - Ask specific clarifying questions to determine the correct agent
                        - Explain what information would help route their query
                        
                        Output must be valid JSON matching this schema:
                        {RouterResponse.model_json_schema()}""",
                },
                {"role": "user", "content": f"Route this query: {query}"},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"},
        }

        with httpx.Client() as client:
            response = client.post(self.url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"]
            return RouterResponse.model_validate_json(result)


class TechnicalSupportAgent:
    def __init__(self):
        self.model = "qwen-2.5-coder-32b"

    def handle_query(self, query: str) -> str:
        # Implement technical support logic
        pass


class BillingSupportAgent:
    def __init__(self):
        self.model = "qwen-2.5-coder-32b"

    def handle_query(self, query: str) -> str:
        # Implement billing support logic
        pass


class RecommendationsAgent:
    def __init__(self):
        self.model = "qwen-2.5-coder-32b"

    def handle_query(self, query: str) -> str:
        # Implement recommendations logic
        pass


def main():
    router = CallRouter()

    # Example queries to test various scenarios
    queries = [
        "My application keeps crashing when I try to save the game state",
        "Why was I charged twice last month for my subscription?",
        "Can you suggest a good database for my high-traffic web project?",
        "Tell me a joke about the weather",
        "I'm having trouble but I'm not sure what's wrong",
        "The log shows: ERROR: game_state.save() failed - weather API timeout",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        result = router.route_query(query)
        print("Routing Result:")
        print(f"Agent Type: {result.agent_type}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Needs Clarification: {result.needs_clarification}")
        if result.clarification_question:
            print(f"Clarification Question: {result.clarification_question}")
        print(f"Response to User: {result.response_to_user}")
        print("-" * 80)


if __name__ == "__main__":
    main()
