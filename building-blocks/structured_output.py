import os
from typing import List, Optional
import httpx
from httpx._types import HeaderTypes, RequestData
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()


class Ingredient(BaseModel):
    name: str
    quantity: str
    quantity_unit: Optional[str]


class Recipe(BaseModel):
    recipe_name: str
    ingredients: List[Ingredient]
    directions: List[str]


api_key = os.environ.get("GROQ_API_KEY")
model = "deepseek-r1-distill-llama-70b"
url = "https://api.groq.com/openai/v1/chat/completions"

headers: HeaderTypes = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}


def get_recipe(recipe_name: str) -> Recipe:
    data: RequestData = {
        "model": f"{model}",
        "messages": [
            {
                "role": "system",
                "content": "You are a recipe generator that outputs recipes in JSON format.\n"
                f" The JSON must adhere to the schema {Recipe.model_json_schema()}",
            },
            {"role": "user", "content": f"Generate a recipe for {recipe_name}."},
        ],
        "temperature": 0.6,
        "response_format": {"type": "json_object"},
    }

    with httpx.Client() as client:
        response = client.post(url, headers=headers, json=data)

    response.raise_for_status()

    response_message = response.json()["choices"][0]["message"]["content"]
    return Recipe.model_validate_json(response_message)


def print_recipe(recipe: Recipe):
    print(f"Recipe for {recipe.recipe_name}:")
    print()
    print("Ingredients:")
    for ingredient in recipe.ingredients:
        print(f"{ingredient.name}: {ingredient.quantity} {ingredient.quantity_unit}")
    print()
    print("Directions:")
    for i, direction in enumerate(recipe.directions):
        print(f"{i + 1}. {direction}")


recipe = get_recipe("chicken parmesan")
print_recipe(recipe)
