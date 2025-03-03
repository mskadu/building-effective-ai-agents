import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from augmented_llm import AugmentedLLM


class ParallelizationType(Enum):
    SECTIONING = "sectioning"
    VOTING = "voting"


@dataclass
class ParallelTask:
    name: str
    prompt: str
    aggregation_func: Optional[Callable] = None


class ParallelProcessor:
    def __init__(self, llm: AugmentedLLM, parallel_type: ParallelizationType):
        self.llm = llm
        self.parallel_type = parallel_type
        self.executor = ThreadPoolExecutor(max_workers=5)

    async def process_section(self, task: ParallelTask) -> Dict[str, Any]:
        """Process a single section"""
        response = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.llm.call, task.prompt
        )
        return {"task": task.name, "response": response}

    async def process_vote(
        self, task: ParallelTask, num_votes: int = 3
    ) -> Dict[str, Any]:
        """Process multiple votes for the same task"""
        tasks = []
        for i in range(num_votes):
            tasks.append(self.process_section(task))

        responses = await asyncio.gather(*tasks)
        return {"task": task.name, "votes": [r["response"] for r in responses]}

    async def process_tasks(self, tasks: List[ParallelTask]) -> Dict[str, Any]:
        """Process multiple tasks in parallel"""
        if self.parallel_type == ParallelizationType.SECTIONING:
            # Run all sections in parallel
            section_tasks = [self.process_section(task) for task in tasks]
            results = await asyncio.gather(*section_tasks)

            # Aggregate results if specified
            if any(task.aggregation_func for task in tasks):
                aggregated_results = {}
                for task, result in zip(tasks, results):
                    if task.aggregation_func:
                        aggregated_results[task.name] = task.aggregation_func(
                            result["response"]
                        )
                    else:
                        aggregated_results[task.name] = result["response"]
                return aggregated_results

            return {r["task"]: r["response"] for r in results}

        else:  # VOTING
            # Run voting for each task
            vote_tasks = [self.process_vote(task) for task in tasks]
            results = await asyncio.gather(*vote_tasks)

            # Aggregate votes if specified
            aggregated_results = {}
            for task, result in zip(tasks, results):
                if task.aggregation_func:
                    aggregated_results[task.name] = task.aggregation_func(
                        result["votes"]
                    )
                else:
                    # Default aggregation: majority vote
                    votes = result["votes"]
                    aggregated_results[task.name] = max(set(votes), key=votes.count)

            return aggregated_results


# Example usage: Content moderation with parallel checks
def check_inappropriate_content(votes: List[str]) -> bool:
    """Aggregate votes to determine if content is inappropriate"""
    yes_votes = sum(1 for vote in votes if "inappropriate" in vote.lower())
    return yes_votes >= 2  # At least 2 votes needed to flag content


async def run_content_moderation_example(llm: AugmentedLLM):
    # Initialize parallel processor for voting
    processor = ParallelProcessor(llm, ParallelizationType.VOTING)

    # Create moderation task
    moderation_task = ParallelTask(
        name="content_check",
        prompt="Is the following content inappropriate? Answer only 'appropriate' or 'inappropriate': {content}",
        aggregation_func=check_inappropriate_content,
    )

    # Run parallel votes
    results = await processor.process_tasks([moderation_task])
    return results


# Example usage: Parallel document analysis
def combine_sentiment_scores(responses: List[str]) -> Dict[str, float]:
    """Aggregate sentiment scores from multiple analyzers"""
    scores = [float(response) for response in responses]
    return {
        "average": sum(scores) / len(scores),
        "max": max(scores),
        "min": min(scores),
    }


async def run_document_analysis_example(llm: AugmentedLLM):
    # Initialize parallel processor for sectioning
    processor = ParallelProcessor(llm, ParallelizationType.SECTIONING)

    # Create analysis tasks
    tasks = [
        ParallelTask(
            name="sentiment",
            prompt="Analyze the sentiment of this text. Return a score from -1 to 1: {text}",
            aggregation_func=combine_sentiment_scores,
        ),
        ParallelTask(
            name="key_topics", prompt="Extract the main topics from this text: {text}"
        ),
        ParallelTask(
            name="summary", prompt="Provide a brief summary of this text: {text}"
        ),
    ]

    # Run parallel analysis
    results = await processor.process_tasks(tasks)
    return results


# Run examples
if __name__ == "__main__":
    # Initialize augmented LLM
    llm = AugmentedLLM("your-api-key")

    content_moderation_results = asyncio.run(run_content_moderation_example(llm=llm))
    print("Content moderation results:", content_moderation_results)

    document_analysis_results = asyncio.run(run_document_analysis_example(llm=llm))
    print("Document analysis results:", document_analysis_results)
