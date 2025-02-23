import os
import sys
import httpx
from httpx._types import HeaderTypes, RequestData
from dotenv import load_dotenv
from pydantic import BaseModel
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)


# Define data models
class Document(BaseModel):
    title: str
    content: str


class DocumentSection(BaseModel):
    title: str
    content: str


class DocumentOutline(BaseModel):
    title: str
    sections: list[DocumentSection]


class Query(BaseModel):
    original_query: str
    enhanced_query: str
    is_document: bool


# Set API key and model details
api_key = os.environ.get("GROQ_API_KEY")
model = "deepseek-r1-distill-llama-70b"
url = "https://api.groq.com/openai/v1/chat/completions"

# Set headers for API requests
headers: HeaderTypes = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}


def rewrite_user_prompt(user_prompt: str) -> Query:
    """
    Check if the user prompts for a document, and if so, rewrite it to make it clearer to the model.
    If the user prompt is not for writing a document, then fill in is_document with False.
    """
    logging.info("Rewriting user prompt")
    data: RequestData = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a query rewriter that outputs queries in JSON format.\n"
                    "Rules:\n"
                    "<rules>\n"
                    "If the user prompt is not for writing a document, do NOT fill in enhanced_query.\n"
                    "If the user prompt is for writing a document, then fill in enhanced_query with a clearer version of the user prompt.\n"
                    "If the user prompt is for writing a document, then fill in is_document with True.\n"
                    "Otherwise, fill in is_document with False.\n"
                    "</rules>\n"
                    "Examples:\n"
                    "Positive example:\n"
                    "User prompt: 'Write a detailed report on climate change.'\n"
                    "Enhanced query: 'Create a comprehensive report on the effects of climate change, including data on temperature changes, sea level rise, and impact on ecosystems.'\n"
                    "is_document: True\n"
                    "Negative example:\n"
                    "User prompt: 'Tell me a joke.'\n"
                    "Enhanced query: ''\n"
                    "is_document: False\n"
                    f"The JSON must adhere to the schema {Query.model_json_schema()}"
                ),
            },
            {
                "role": "user",
                "content": f"Rewrite the user prompt {user_prompt} to make it clearer to the model.",
            },
        ],
        "temperature": 0.6,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)

    response.raise_for_status()

    response_message = response.json()["choices"][0]["message"]["content"]
    return Query.model_validate_json(response_message)


def plan_document(query: str) -> DocumentOutline:
    """
    Plan a document based on the given query.
    """
    logging.info("Planning document")
    data: RequestData = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a document planner that outputs document outlines in JSON format.\n"
                    f"The JSON must adhere to the schema {DocumentOutline.model_json_schema()}"
                ),
            },
            {
                "role": "user",
                "content": f"Plan a document based on the following query: {query}",
            },
        ],
        "temperature": 0.6,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)

    response.raise_for_status()

    response_message = response.json()["choices"][0]["message"]["content"]
    return DocumentOutline.model_validate_json(response_message)


def write_document(outline: DocumentOutline) -> Document:
    """
    Call a model to write the document in markdown format based on the outline given.
    Parse the response into a Document object.
    """
    logging.info("Writing document")
    tries = 0
    while tries < 3:
        tries += 1
        data: RequestData = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a document writer that outputs documents in Markdown format.\n"
                        f"The JSON must adhere to the schema {Document.model_json_schema()}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Write the document outlined in the following markdown format:\n\n{outline.title}\n\n{outline.sections}",
                },
            ],
            "temperature": 0.6,
            "response_format": {"type": "json_object"},
        }

        try:
            with httpx.Client() as client:
                response = client.post(url, headers=headers, json=data)

                response.raise_for_status()

                response_message = response.json()["choices"][0]["message"]["content"]
                return Document.model_validate_json(response_message)
        except Exception as e:
            logging.error(f"Error writing document: {e}")
            logging.info("Retrying...")
            continue


def display_markdown(markdown: str):
    """
    Display the given markdown content using rich library.
    """
    from rich.markdown import Markdown
    from rich.console import Console

    console = Console()
    console.print(Markdown(markdown))


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 1:
        print("Usage: python prompt-chaining.py <user_prompt>")
        print(
            "Running with default user prompt: Write a blog post about the benefits of effective agents."
        )
        user_prompt = "Write a blog post about the benefits of effective agents."
    else:
        user_prompt = args[0]

    print(f"Rewriting user prompt: {user_prompt}")
    query = rewrite_user_prompt(user_prompt)
    print(f"Enhanced query: {query.enhanced_query}")
    print(f"Is document: {query.is_document}")
    if query.is_document:
        print("Planning document")
        outline = plan_document(query.enhanced_query)
        print("Writing document")
        document = write_document(outline)
        print("\n=== Generated Document ===\n")
        display_markdown(document.title)
        display_markdown(document.content)
        print("\n=======================\n")
    else:
        print("Not writing a document")
        sys.exit(0)
