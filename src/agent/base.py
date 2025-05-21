"""
Base classes for Alpha Modeling Agents
"""
from typing import Dict, Any, List, Optional, TypedDict, Union
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import BaseTool
from abc import ABC, abstractmethod
import logging

from .config import GOOGLE_API_KEY, DEFAULT_MODEL_NAME

# Configure logging
logging.basicConfig(level=logging.INFO)


class WorkerInput(TypedDict):
    """Input format for worker agents"""
    task_type: str
    task_id: str
    inputs: Dict[str, Any]
    context: Optional[Dict[str, Any]]


class WorkerOutput(TypedDict):
    """Output format for worker agents"""
    task_id: str
    status: str  # "success" or "error"
    outputs: Dict[str, Any]
    error: Optional[str]


class BaseWorkerAgent(ABC):
    """Base worker agent class that all specialized agents will inherit from"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: Optional[List[BaseTool]] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = 0.1,
        **kwargs
    ):
        self.name = name
        self.logger = logging.getLogger(f"AlphaModeling.{self.name}")
        self.tools = tools or []

        # Initialize LLM
        if not GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found in environment. Please set it before running the agent.")
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=GOOGLE_API_KEY,
            temperature=temperature,
            **kwargs
        )

        # Create prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

    async def execute(self, worker_input: WorkerInput) -> WorkerOutput:
        """
        Execute the worker agent with the given input

        Args:
            worker_input: Input for the worker agent

        Returns:
            WorkerOutput containing results or error
        """
        self.logger.info(
            f"Executing {self.name} with task_id: {worker_input['task_id']}")

        try:
            # Process the input and generate output based on task type
            result = await self._process_task(worker_input)

            # Return success response
            return WorkerOutput(
                task_id=worker_input["task_id"],
                status="success",
                outputs=result,
                error=None
            )

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            # Return error response
            return WorkerOutput(
                task_id=worker_input["task_id"],
                status="error",
                outputs={},
                error=str(e)
            )

    @abstractmethod
    async def _process_task(self, worker_input: WorkerInput) -> Dict[str, Any]:
        """
        Process the specific task for this worker agent

        Args:
            worker_input: Input for the worker agent

        Returns:
            Dictionary containing the results
        """
        pass

    async def _call_llm(self, input_str: str) -> str:
        """Helper to call the LLM with a string input"""
        chain = self.prompt | self.llm
        response = await chain.ainvoke({"input": input_str})
        return response.content
