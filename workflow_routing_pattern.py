from enum import Enum
from typing import Callable, Dict

from augmented_llm import AugmentedLLM


class QueryType(Enum):
    GENERAL = "general"
    TECHNICAL = "technical"
    REFUND = "refund"
    UNKNOWN = "unknown"


class Router:
    def __init__(self, llm: AugmentedLLM):
        self.llm = llm
        self.handlers: Dict[QueryType, Callable] = {}

    def register_handler(self, query_type: QueryType, handler: Callable):
        """Register a handler for a specific query type"""
        self.handlers[query_type] = handler

    def classify_query(self, query: str) -> QueryType:
        """Use LLM to classify the query type"""
        classification_prompt = f"""
        Classify the following customer query into one of these categories:
        - GENERAL: General product questions
        - TECHNICAL: Technical support issues
        - REFUND: Refund requests
        - UNKNOWN: Cannot be classified
        
        Query: {query}
        
        Respond with only the category name.
        """

        response = self.llm.call(classification_prompt).strip().upper()
        try:
            return QueryType[response]
        except KeyError:
            return QueryType.UNKNOWN

    def route_and_handle(self, query: str) -> str:
        """Classify query and route to appropriate handler"""
        query_type = self.classify_query(query)

        handler = self.handlers.get(query_type)
        if not handler:
            return f"No handler registered for query type: {query_type.value}"

        return handler(query)


# Example handlers
def handle_general(query: str) -> str:
    return f"General inquiry handler: {query}"


def handle_technical(query: str) -> str:
    return f"Technical support handler: {query}"


def handle_refund(query: str) -> str:
    return f"Refund request handler: {query}"


# Example usage
if __name__ == "__main__":
    # Initialize augmented LLM
    llm = AugmentedLLM("your-api-key")

    # Usage example
    router = Router(llm)
    router.register_handler(QueryType.GENERAL, handle_general)
    router.register_handler(QueryType.TECHNICAL, handle_technical)
    router.register_handler(QueryType.REFUND, handle_refund)

    # Example queries
    queries = [
        "How do I use the product?",
        "My device won't turn on",
        "I want a refund",
    ]

    for query in queries:
        response = router.route_and_handle(query)
        print(f"Query: {query}\nResponse: {response}\n")
