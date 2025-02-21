from dataclasses import dataclass
from typing import Any, Dict, List

from anthropic import Anthropic


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]


class AugmentedLLM:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
        self.tools: List[Tool] = []
        self.memory: List[Dict] = []

    def add_tool(self, tool: Tool):
        """Register a new tool with the LLM"""
        self.tools.append(tool)

    def get_tool_descriptions(self) -> str:
        """Format tool descriptions for the prompt"""
        descriptions = []
        for tool in self.tools:
            desc = f"Tool: {tool.name}\nDescription: {tool.description}\n"
            desc += f"Parameters: {tool.parameters}\n"
            descriptions.append(desc)
        return "\n".join(descriptions)

    def call(self, prompt: str) -> str:
        """Make an augmented call to the LLM"""
        # Add tools and memory context to the prompt
        context = f"""Available tools:\n{self.get_tool_descriptions()}\n\n"""
        if self.memory:
            context += "Previous context:\n"
            for item in self.memory[-5:]:  # Last 5 memory items
                context += f"{item}\n"

        full_prompt = context + prompt

        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            messages=[{"role": "user", "content": full_prompt}],
        )

        # Store response in memory
        self.memory.append({"prompt": prompt, "response": response.content[0].text})

        return response.content[0].text


# Example usage
if __name__ == "__main__":
    # Initialize augmented LLM
    llm = AugmentedLLM("your-api-key")

    # Add a search tool
    search_tool = Tool(
        name="search",
        description="Search for information on a given topic",
        parameters={"query": "str", "max_results": "int"},
    )
    llm.add_tool(search_tool)

    # Make a call
    response = llm.call(
        "What is the capital of Nepal and what's the weather like there?"
    )
    print(response)
