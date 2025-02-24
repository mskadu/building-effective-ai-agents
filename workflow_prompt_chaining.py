from dataclasses import dataclass
from typing import Callable, List, Optional

from augmented_llm import AugmentedLLM


@dataclass
class ChainStep:
    name: str
    prompt_template: str
    validation_func: Optional[Callable[[str], bool]] = None


class PromptChain:
    def __init__(self, llm: AugmentedLLM):
        self.llm = llm
        self.steps: List[ChainStep] = []

    def add_step(self, step: ChainStep):
        """Add a step to the chain"""
        self.steps.append(step)

    def execute(self, initial_input: str) -> List[dict]:
        """Execute the full chain"""
        results = []
        current_input = initial_input

        for step in self.steps:
            # Format prompt with previous input
            prompt = step.prompt_template.format(input=current_input)

            # Execute LLM call
            response = self.llm.call(prompt)

            # Validate if needed
            if step.validation_func and not step.validation_func(response):
                raise ValueError(f"Validation failed for step: {step.name}")

            # Store results
            results.append(
                {"step": step.name, "input": current_input, "output": response}
            )

            # Update input for next step
            current_input = response

        return results


# Example usage: Marketing copy generation and translation
def validate_marketing_copy(text: str) -> bool:
    """Validate marketing copy meets requirements"""
    # Add your validation logic here
    return len(text) >= 100 and not any(
        banned_word in text.lower() for banned_word in ["spam", "guarantee"]
    )


# Create chain steps
marketing_step = ChainStep(
    name="generate_marketing",
    prompt_template="Create marketing copy for the following product: {input}",
    validation_func=validate_marketing_copy,
)

translation_step = ChainStep(
    name="translate",
    prompt_template="Translate the following marketing copy to Spanish: {input}",
)

# Example usage
if __name__ == "__main__":
    # Initialize augmented LLM
    llm = AugmentedLLM("your-api-key")

    # Execute chain
    chain = PromptChain(llm)
    chain.add_step(marketing_step)
    chain.add_step(translation_step)
    # and any other steps you want to add...

    results = chain.execute(
        "Premium noise-cancelling headphones with 24-hour battery life"
    )
