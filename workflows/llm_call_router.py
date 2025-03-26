from abc import ABC, abstractmethod
import json
import os
from enum import Enum
from typing import Dict, Optional
import httpx
from httpx._types import HeaderTypes, RequestData
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


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


class BaseAgent(ABC):
    """Base class for agents interacting with the Groq API."""

    def __init__(self, model: str, system_message: str):
        self.model = model
        self.system_message = system_message
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.headers: HeaderTypes = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _make_api_call(
        self,
        messages: list[Dict[str, str]],
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """Handles the common logic for making API calls."""
        data: RequestData = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_mode:
            data["response_format"] = {"type": "json_object"}

        with httpx.Client(timeout=30.0) as client:
            try:
                response = client.post(self.url, headers=self.headers, json=data)
                response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
                response_data = response.json()

                # Safer access to nested data
                choices = response_data.get("choices")
                if not choices or not isinstance(choices, list):
                    raise ValueError(
                        "Invalid response format: 'choices' missing or not a list."
                    )

                message = choices[0].get("message")
                if not message or not isinstance(message, dict):
                    raise ValueError(
                        "Invalid response format: 'message' missing or not a dict in first choice."
                    )

                content = message.get("content")
                if content is None:  # Allow empty string, but not None
                    raise ValueError(
                        "Invalid response format: 'content' missing in message."
                    )

                return str(content)  # Ensure it's a string

            except httpx.HTTPStatusError as e:
                # Log or handle specific HTTP errors
                print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
                raise  # Re-raise after logging
            except (
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                # Handle parsing errors or unexpected structure
                print(f"Error processing API response: {e}")
                print(
                    f"Raw response text: {response.text if 'response' in locals() else 'No response object'}"
                )
                raise RuntimeError(f"Failed to parse API response: {e}") from e
            except httpx.RequestError as e:
                # Handle network errors (timeout, connection error, etc.)
                print(f"Network error during API call: {e}")
                raise RuntimeError(f"Network error connecting to API: {e}") from e

    @abstractmethod
    def handle_query(self, query: str) -> str:
        """Each specific agent must implement how it handles a query."""
        pass


class CallRouter:
    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
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
            try:
                result = response.json()["choices"][0]["message"]["content"]
                if not result:
                    # Handle cases where content is missing or empty
                    raise ValueError("Failed to extract content from API response.")
                # Handle potential JSON within content (for router)
                if isinstance(
                    result, str
                ):  # Groq might return content as a string even with json_object
                    return RouterResponse.model_validate_json(result)
                else:  # Or it might already be a dict
                    return RouterResponse.model_validate(result)
            except (
                IndexError,
                KeyError,
                AttributeError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                print(f"Error parsing API response: {e}")
                print(f"Raw response: {response.text}")
                return RouterResponse(
                    agent_type=AgentType.UNKNOWN,
                    confidence=0.0,
                    needs_clarification=True,
                    clarification_question="I'm not sure how to help. Can you provide more details?",
                    response_to_user="I'm not sure how to help. Can you provide more details?",
                )


class TechnicalSupportAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model="llama-3.3-70b-versatile",
            system_message="You are a helpful and concise technical support agent specializing in software development, debugging, and system issues. Focus on providing actionable solutions and code examples where relevant.",
        )

    def handle_query(self, query: str) -> str:
        """Handles a technical query using the LLM."""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": query},
        ]
        return self._make_api_call(messages, temperature=0.2)


class BillingSupportAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model="llama-3.3-70b-versatile",
            system_message="You are a polite and professional billing support agent. Address questions about charges, invoices, subscriptions, and payment methods clearly and accurately. Do not ask for or process sensitive payment details.",
        )

    def handle_query(self, query: str) -> str:
        """Handles a billing query using the LLM."""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": query},
        ]
        return self._make_api_call(messages, temperature=0.1)


class RecommendationsAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            model="llama-3.3-70b-versatile",  # Maybe needs larger model for good recommendations
            system_message="You are an insightful product recommendations agent. Based on user needs and context, suggest relevant software, tools, or services. Explain *why* you are recommending something.",
        )

    def handle_query(self, query: str) -> str:
        """Handles a recommendation query using the LLM."""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": query},
        ]
        return self._make_api_call(messages, temperature=0.5)


def main():
    try:  # Added top-level try/except for initialization errors
        router = CallRouter()
        agents = {
            AgentType.TECHNICAL: TechnicalSupportAgent(),
            AgentType.BILLING: BillingSupportAgent(),
            AgentType.RECOMMENDATIONS: RecommendationsAgent(),
        }
    except ValueError as e:
        print(f"Initialization Error: {e}")
        return  # Exit if agents can't be created (e.g., missing API key)

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
        try:
            result = router.route_query(query)
            print("Routing Result:")
            print(f"Agent Type: {result.agent_type}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Needs Clarification: {result.needs_clarification}")
            if result.clarification_question:
                print(f"Clarification Question: {result.clarification_question}")
            print(f"Response to User: {result.response_to_user}")

            agent = agents.get(result.agent_type)
            if agent:
                print(f"--> Routing to {result.agent_type.value.capitalize()} Agent...")
                agent_response = agent.handle_query(query)
                print(f"Agent Response: {agent_response}")
            elif result.agent_type in (AgentType.IRRELEVANT, AgentType.UNKNOWN):
                print(
                    f"--> No specific agent required ({result.agent_type.value}). User response provided by router."
                )
            else:
                # Should not happen with Enum validation
                print(f"Warning: Unhandled agent type: {result.agent_type}")

        except (
            httpx.HTTPStatusError,
            httpx.RequestError,
            RuntimeError,
            ValidationError,
        ) as e:
            print(f"!! Error processing query: {e}")

        print("-" * 80)


if __name__ == "__main__":
    main()
