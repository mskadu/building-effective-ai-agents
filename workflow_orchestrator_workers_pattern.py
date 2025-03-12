import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from augmented_llm import AugmentedLLM


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


# Subtask (or worker)
@dataclass
class SubTask:
    id: str
    description: str
    dependencies: List[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None


# The Orchestrator
class OrchestratorAgent:
    def __init__(self, llm: AugmentedLLM):
        self.llm = llm
        self.tasks: Dict[str, SubTask] = {}

    def plan_subtasks(self, main_task: str) -> List[SubTask]:
        """Use LLM to break down the main task into subtasks"""
        planning_prompt = f"""
        Break down the following task into smaller subtasks:
        {main_task}
        
        Return the subtasks in JSON format with the following structure:
        {{
            "subtasks": [
                {{
                    "id": "unique_id",
                    "description": "subtask description",
                    "dependencies": ["dependent_task_ids"]
                }}
            ]
        }}
        
        Ensure tasks are properly ordered with dependencies.
        """

        response = self.llm.call(planning_prompt)
        try:
            plan = json.loads(response)
            return [SubTask(**task) for task in plan["subtasks"]]
        except json.JSONDecodeError:
            raise ValueError("Failed to parse LLM response as JSON")

    async def execute_subtask(self, task: SubTask) -> str:
        """Execute a single subtask using a worker LLM"""
        execution_prompt = f"""
        Execute the following task:
        {task.description}
        
        If this task depends on other tasks, here are their results:
        {self._get_dependency_results(task)}
        """

        try:
            result = self.llm.call(execution_prompt)
            return result
        except Exception as e:
            task.status = TaskStatus.FAILED
            raise Exception(f"Failed to execute task {task.id}: {str(e)}")

    def _get_dependency_results(self, task: SubTask) -> str:
        """Get results of dependency tasks"""
        dependency_results = []
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if dep_task and dep_task.result:
                dependency_results.append(f"{dep_id}: {dep_task.result}")
        return "\n".join(dependency_results)

    def synthesize_results(self, results: Dict[str, str]) -> str:
        """Use LLM to synthesize all results into final output"""
        synthesis_prompt = f"""
        Synthesize the results of all subtasks into a coherent final output:
        
        Results:
        {json.dumps(results, indent=2)}
        """

        return self.llm.call(synthesis_prompt)

    async def execute_task(self, main_task: str) -> str:
        """Execute the main task using orchestrator-workers pattern"""
        # Plan subtasks
        subtasks = self.plan_subtasks(main_task)
        for task in subtasks:
            self.tasks[task.id] = task

        # Execute tasks respecting dependencies
        results = {}
        while any(task.status != TaskStatus.COMPLETED for task in self.tasks.values()):
            # Find tasks ready to execute (all dependencies completed)
            ready_tasks = [
                task
                for task in self.tasks.values()
                if task.status == TaskStatus.PENDING
                and all(
                    self.tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
            ]

            if not ready_tasks:
                # Check for failed tasks
                if any(
                    task.status == TaskStatus.FAILED for task in self.tasks.values()
                ):
                    raise Exception("Task execution failed")
                # If no tasks are ready but some are still pending, we have a dependency cycle
                if any(
                    task.status == TaskStatus.PENDING for task in self.tasks.values()
                ):
                    raise Exception("Dependency cycle detected")
                break

            # Execute ready tasks in parallel
            tasks = []
            for task in ready_tasks:
                task.status = TaskStatus.IN_PROGRESS
                tasks.append(self.execute_subtask(task))

            task_results = await asyncio.gather(*tasks)

            # Store results
            for task, result in zip(ready_tasks, task_results):
                task.status = TaskStatus.COMPLETED
                task.result = result
                results[task.id] = result

        # Synthesize results
        return self.synthesize_results(results)


# Example usage: Complex code change
async def run_code_change_example(llm: AugmentedLLM):
    orchestrator = OrchestratorAgent(llm)

    main_task = """
    Update our user authentication system to:
    1. Add support for two-factor authentication
    2. Implement password complexity requirements
    3. Add rate limiting for login attempts
    """

    try:
        result = await orchestrator.execute_task(main_task)
        return result
    except Exception as e:
        print(f"Failed to execute task: {str(e)}")
        return None


# Example usage: Search and analysis task
async def run_search_analysis_example(llm: AugmentedLLM):
    orchestrator = OrchestratorAgent(llm)

    main_task = """
    Research and analyze the impact of artificial intelligence on healthcare:
    1. Gather information from multiple sources
    2. Analyze trends and patterns
    3. Identify key challenges and opportunities
    4. Create a comprehensive report
    """

    try:
        result = await orchestrator.execute_task(main_task)
        return result
    except Exception as e:
        print(f"Failed to execute task: {str(e)}")
        return None


# Run our examples
if __name__ == "__main__":
    # Initialize augmented LLM
    llm = AugmentedLLM("your-api-key")

    code_change_results = asyncio.run(run_code_change_example(llm=llm))
    print("Code change results:", code_change_results)

    search_analysis_results = asyncio.run(run_search_analysis_example(llm=llm))
    print("Search analysis results:", search_analysis_results)
