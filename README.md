# AI Effective Agents

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Project Structure](#project-structure)
4. [Contributing](#contributing)
5. [License](#license)

## Introduction

Exploration of AI Effective Agents based on Anthropic's blog post

This project is a simple exploration of the ideas presented in <https://www.anthropic.com/research/building-effective-agents>.

We'll explore three main ideas:

1. **Building Blocks**: These are described as augmented LLMs, so essentially just a simple system around an LLM (typically an API call).

2. **Workflows**: Described as systems where LLMs and tools are orchestrated in predefined code paths.

3. **Agents**: Described as systems where LLMs dynamically direct their behaviour and tool usage, maintaining more control over how they accomplish their goals.

Note: To be fair, there is sometimes a blurry line separating the Workflows from Agents in these patterns by Anthropic's own definition, as some workflow patterns arguably have part of the control flow directed by an LLM. I like the classic agent definition from Russell and Norvig's book "Artificial Intelligence: A Modern Approach", where an agent is defined as a system that perceives its environment and acts upon it. Effectively you can build workflows with simple LLM calls that only "see" its own context and produce a single output, making it a stretch to call them agents.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/srps/anthropic-effective-agents.git
```

2. Sync dependencies using uv:

```bash
cd anthropic-effective-agents
uv sync
```

3. Create a `.env` file in the root directory with your Groq API key:
```bash
cp .env.example .env
```

Then, replace `your-groq-api-key` with your actual Groq API key.

4. Choose an example (e.g., the building blocks example), navigate into that directory, and run it:

```bash
cd building-blocks
uv run basic.py
```

## Project Structure

Here's a brief overview of the repository layout:

- `building-blocks/`: Contains examples demonstrating augmented LLMs as basic agent building blocks.
- `workflows/`: Hosts examples where LLMs and tools are orchestrated along predefined paths.
- `agents/`: Provides examples of dynamic agent control where the LLM directs behavior and tool usage.
- `README.md`: This file, providing an overview and setup instructions.
- `.env.example`: A template for environment variable configuration.

## Contributing

This project is just a very basic exploration of some ideas with basic examples. If you have any suggestions or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
