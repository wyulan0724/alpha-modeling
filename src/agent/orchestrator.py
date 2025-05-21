"""
Orchestrator Agent for Alpha Modeling

Responsible for coordinating the overall workflow and delegating tasks to specialized worker agents.
"""
from typing import Dict, Any, List, Tuple, Union, Annotated, TypedDict, Optional, Literal
import uuid
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import logging

from .config import (
    GOOGLE_API_KEY,
    AGENT_CONFIG,
    DEFAULT_PARQUET_FILE_PATH,
    DEFAULT_TRAIN_START,
    DEFAULT_TRAIN_END,
    DEFAULT_VALID_START,
    DEFAULT_VALID_END,
    DEFAULT_TEST_START,
    DEFAULT_TEST_END,
    DEFAULT_LEARN_PROCESSORS,
    DEFAULT_INFER_PROCESSORS
)
from .base import WorkerInput, WorkerOutput
from .workers import get_worker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaModeling.Orchestrator")

# State definition


class OrchestratorState(TypedDict):
    """State maintained by the orchestrator graph"""
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]]
    user_request: str
    tasks: Dict[str, Dict[str, Any]]
    current_task_id: Optional[str]
    current_worker: Optional[str]
    worker_response: Optional[WorkerOutput]
    results: Dict[str, Any]
    status: Literal["in_progress", "completed", "failed"]
    next_action: Optional[Literal["DELEGATE",
                                  "RESPOND", "THINK", "ERROR", "FINISH"]]


ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator Agent for an Alpha Modeling system, responsible for coordinating the overall workflow for training, testing, and evaluating machine learning models for financial predictions.

Your job is to:
1. Understand the user's request for alpha modeling tasks
2. Break down the request into a sequence of specific tasks for specialized worker agents
3. Determine which worker agent should handle each task
4. Pass the appropriate inputs to each worker agent
5. Collect outputs from worker agents to use as inputs for subsequent tasks
6. Track the progress and respond to the user with final results

You can delegate tasks to these specialized worker agents:
- DataPreparationWorker: Creates dataset configurations for model training
- ModelTrainingWorker: Trains ML models using the dataset configurations
- PredictionGenerationWorker: Generates predictions using trained models
- BacktestingWorker: Performs portfolio backtesting with predictions
- AnalysisReportingWorker: Analyzes results and generates reports

You must decide what step to take next based on the current state of the workflow. Valid next steps:
1. DELEGATE: Send a task to a worker agent
2. RESPOND: Provide a final answer to the user
3. THINK: Reason about what to do next

When you decide to DELEGATE, you must specify:
- Which worker agent to use
- The task ID
- The task type
- All required inputs for the task

Additional Instructions:
- If you cannot determine which worker agent or task type to use based on the user's request, or if the request is ambiguous or incomplete, you must RESPOND with a clear message explaining what information is missing or why the request cannot be processed. Do not enter a THINK loop.
- Only use "THINK" if you are actively reasoning about the next step and expect to resolve the ambiguity in the next turn. If you still cannot decide after one THINK, you must RESPOND.
- If the user does not specify certain parameters (e.g., file paths, dates), you can omit them from the 'inputs' dictionary, and the system will attempt to use sensible defaults. Or, clearly state what information is missing if it's critical and cannot be defaulted.
- If the user's request implies a single step (e.g., 'Create a dataset config') and that step's corresponding worker (e.g., DataPreparationWorker) has successfully completed, your next action should be RESPOND.

Example:
```json
{
  "action": "DELEGATE",
  "worker": "DataPreparationWorker",
  "task_id": "<unique_id>",
  "task_type": "prepare_dataset_config",
  "inputs": {
    "parquet_file_path": "/path/to/data.parquet",
    "train_start_date": "2015-01-01",
    "train_end_date": "2021-12-31",
    "valid_start_date": "2022-01-01",
    "valid_end_date": "2022-12-31", 
    "test_start_date": "2023-01-01",
    "test_end_date": "2023-12-31"
  }
}
```

When you decide to RESPOND, provide a clear summary of the completed workflow and results:
```json
{
  "action": "RESPOND",
  "response": "I've completed the alpha modeling workflow. Here are the results: <results>"
}
```

When you decide to THINK, explain your reasoning about the current state and what should be done next:
```json
{
  "action": "THINK",
  "reasoning": "I need to determine what task to perform next. The user has requested model training and evaluation for TW50 stocks. First, I should delegate dataset preparation to the DataPreparationWorker..."
}
```

Remember to always respond with valid JSON in one of these formats.
"""


def create_task_id() -> str:
    """Create a unique task ID"""
    return str(uuid.uuid4())


def create_orchestrator_tool() -> BaseTool:
    """Create a tool for the orchestrator to execute tasks"""
    pass  # To be implemented when we add more worker agents


def create_orchestrator_llm():
    """Create the LLM for the orchestrator"""
    config = AGENT_CONFIG["orchestrator"]
    return ChatGoogleGenerativeAI(
        model=config.get("model_name", "gemini-2.0-flash"),
        google_api_key=GOOGLE_API_KEY,
        temperature=config.get("temperature", 0),
        max_output_tokens=config.get("max_tokens", 2048)
    )


def create_orchestrator_prompt():
    """Create the prompt for the orchestrator"""
    system_prompt = ORCHESTRATOR_SYSTEM_PROMPT.replace(
        "{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])


def build_orchestrator_chain():
    """Build the orchestrator chain with LLM and prompt"""
    llm = create_orchestrator_llm()
    prompt = create_orchestrator_prompt()
    return prompt | llm


def parse_orchestrator_output(output: str) -> Dict[str, Any]:
    """Parse the output from the orchestrator"""
    try:
        # Extract JSON from potential markdown code blocks
        if "```json" in output:
            output = output.split("```json")[1].split("```")[0]
        elif "```" in output:
            output = output.split("```")[1].split("```")[0]

        return json.loads(output.strip())
    except Exception as e:
        logger.error(f"Error parsing orchestrator output: {str(e)}")
        logger.debug(f"Raw output: {output}")
        raise ValueError(f"Failed to parse orchestrator output: {str(e)}")


def process_user_input(state: OrchestratorState) -> Dict[str, Any]:
    """Process the initial user input"""
    logger.info(f"---PROCESSING USER INPUT: {state['user_request']}---")
    user_request = state["user_request"]

    messages = state.get("messages", [])
    messages.append(HumanMessage(content=user_request))

    # 僅返回已更改的部分
    return {
        "messages": messages,
        "status": "in_progress",
        "tasks": state.get("tasks", {}),
        "results": state.get("results", {})
    }


async def run_orchestrator(state: OrchestratorState) -> Dict[str, Any]:
    """Run the orchestrator to decide the next action"""
    logger.info("---ORCHESTRATOR: Deciding next action---")

    # Build the input for the orchestrator
    input_str = f"Current user request: {state['user_request']}\n\n"

    # Add information about completed tasks and results
    if state["tasks"]:
        input_str += "Completed tasks:\n"
        for task_id, task in state["tasks"].items():
            if task.get("status") == "completed":
                input_str += f"- {task.get('worker', 'Unknown')}: {task.get('task_type', 'Unknown')} (ID: {task_id})\n"

    # Add the latest worker response if available
    if state.get("worker_response"):
        worker_response = state["worker_response"]
        input_str += f"\nLatest worker response (from {worker_response.get('worker', 'Unknown')}):\n"
        input_str += json.dumps(worker_response, indent=2)

    # Get the orchestrator's decision
    chain = build_orchestrator_chain()
    response = await chain.ainvoke({"input": input_str})

    # Parse the response
    try:
        decision = parse_orchestrator_output(response.content)
    except ValueError as e:
        logger.error(f"Orchestrator output parsing error: {e}")
        state["messages"].append(AIMessage(
            content=f"Error parsing my own output: {response.content}. Will retry."))
        return {"messages": state["messages"], "next_action": "THINK"}

    action = decision.get("action", "").upper()

    logger.info(f"Orchestrator LLM decided action: {action}")
    logger.debug(f"Orchestrator LLM full decision: {decision}")

    # Add the AI's reasoning to the conversation
    messages = state.get("messages", [])
    messages.append(AIMessage(content=json.dumps(decision)))

    updated_state_delta: Dict[str, Any] = {"messages": state["messages"]}

    # Determine the next node based on the action
    if action == "DELEGATE":
        # Prepare to delegate to a worker
        worker = decision.get("worker")
        task_id = decision.get("task_id", create_task_id())
        task_type = decision.get("task_type")
        inputs = decision.get("inputs", {})

        if not all([worker, task_type, inputs is not None]):
            logger.error(
                f"Orchestrator DELEGATE action missing required fields: {decision}")
            updated_state_delta["next_action"] = "THINK"
            updated_state_delta["messages"].append(SystemMessage(
                content="Internal Error: DELEGATE action was invalid."))
            return updated_state_delta

        final_inputs = {}
        if worker == "DataPreparationWorker" and task_type == "prepare_dataset_config":
            logger.info(
                "Orchestrator applying default values for DataPreparationWorker if needed.")
            final_inputs = {
                "parquet_file_path": inputs.get('parquet_file_path', DEFAULT_PARQUET_FILE_PATH),
                "train_start_date": inputs.get('train_start_date', DEFAULT_TRAIN_START),
                "train_end_date": inputs.get('train_end_date', DEFAULT_TRAIN_END),
                "valid_start_date": inputs.get('valid_start_date', DEFAULT_VALID_START),
                "valid_end_date": inputs.get('valid_end_date', DEFAULT_VALID_END),
                "test_start_date": inputs.get('test_start_date', DEFAULT_TEST_START),
                "test_end_date": inputs.get('test_end_date', DEFAULT_TEST_END),
                "learn_processors_config": inputs.get('learn_processors_config', DEFAULT_LEARN_PROCESSORS),
                "infer_processors_config": inputs.get('infer_processors_config', DEFAULT_INFER_PROCESSORS),
            }

            for key, value in inputs.items():
                if key not in final_inputs:
                    final_inputs[key] = value
            logger.debug(
                f"Final inputs for DataPreparationWorker after defaults: {final_inputs}")
        else:
            final_inputs = inputs

        if not final_inputs and worker == "DataPreparationWorker":
            logger.error(
                f"DELEGATE for DataPreparationWorker has empty final_inputs. LLM decision: {decision}")
            return updated_state_delta

        # Store the task in the state
        current_tasks = state.get("tasks", {})
        current_tasks[task_id] = {
            "worker": worker,
            "task_type": task_type,
            "inputs": final_inputs,
            "status": "pending"
        }

        # Update current task information
        updated_state_delta["tasks"] = current_tasks
        updated_state_delta["current_task_id"] = task_id
        updated_state_delta["current_worker"] = worker
        updated_state_delta["next_action"] = "DELEGATE"

    elif action == "RESPOND":
        # Prepare the final response to the user
        response_content = decision.get(
            "response", "Task completed successfully.")

        # Update state
        updated_state_delta["messages"].append(
            AIMessage(content=response_content))
        updated_state_delta["status"] = "completed"
        updated_state_delta["next_action"] = "FINISH"

    elif action == "THINK":
        # The orchestrator is thinking about what to do next
        # Just continue back to the orchestrator
        updated_state_delta["next_action"] = "THINK"

    else:
        # Unknown action, log warning and continue
        logger.warning(
            f"Unknown action from Orchestrator LLM: {action}. Decision: {decision}")
        updated_state_delta["next_action"] = "THINK"

    return updated_state_delta


async def delegate_to_worker(state: OrchestratorState) -> Dict[str, Any]:
    """Delegate the current task to the appropriate worker and get the result."""
    task_id = state["current_task_id"]
    worker_name = state["current_worker"]
    task_info = state.get("tasks", {}).get(task_id)

    logger.info(
        f"---DELEGATING to worker: {worker_name} for task_id: {task_id}---")

    if not task_info:
        logger.error(f"No task info found for task_id: {task_id}")
        return {"worker_response": WorkerOutput(task_id=task_id, status="error", outputs={}, error="Task info not found"),
                "status": "failed"}

    worker_input_data = WorkerInput(
        task_type=task_info["task_type"],
        task_id=task_id,
        inputs=task_info["inputs"],
        context=state.get("results")
    )

    try:
        worker_instance = get_worker(worker_name)
        actual_worker_output: WorkerOutput = await worker_instance.execute(worker_input_data)

        logger.info(
            f"Worker {worker_name} completed task_id: {task_id} with status: {actual_worker_output['status']}")

        updated_tasks = state.get("tasks", {})
        if actual_worker_output["status"] == "success":
            updated_tasks[task_id]["status"] = "completed"
            updated_tasks[task_id]["output"] = actual_worker_output["outputs"]

            current_results = state.get("results", {})
            current_results[task_id] = actual_worker_output["outputs"]

            return {
                "tasks": updated_tasks,
                "worker_response": actual_worker_output,
                "results": current_results,
                "current_task_id": None,
                "current_worker": None
            }
        else:
            updated_tasks[task_id]["status"] = "error"
            updated_tasks[task_id]["error"] = actual_worker_output["error"]
            logger.error(
                f"Worker {worker_name} failed task {task_id}: {actual_worker_output['error']}")
            return {
                "tasks": updated_tasks,
                "worker_response": actual_worker_output,
                "status": "failed",
                "current_task_id": None,
                "current_worker": None
            }

    except Exception as e:
        logger.error(
            f"Exception during delegation to {worker_name} for task {task_id}: {str(e)}", exc_info=True)
        error_output = WorkerOutput(task_id=task_id, status="error", outputs={
        }, error=f"Orchestrator-level error: {str(e)}")
        updated_tasks = state.get("tasks", {})
        updated_tasks[task_id]["status"] = "error"
        updated_tasks[task_id]["error"] = str(e)
        return {
            "tasks": updated_tasks,
            "worker_response": error_output,
            "status": "failed",
            "current_task_id": None,
            "current_worker": None
        }


def create_orchestrator_graph():
    """Create the orchestrator workflow graph"""
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("process_user_input", process_user_input)
    workflow.add_node("run_orchestrator_node", run_orchestrator)
    workflow.add_node("delegate_to_worker_node", delegate_to_worker)

    # Set the entry point
    workflow.set_entry_point("process_user_input")

    # Add edges
    workflow.add_edge("process_user_input", "run_orchestrator_node")
    # Back to Orchestrator after worker completes
    workflow.add_edge("delegate_to_worker_node", "run_orchestrator_node")

    # Conditional routing from orchestrator
    def decide_next_step(state: OrchestratorState) -> str:
        logger.info(
            f"Orchestrator decision router: next_action is '{state.get('next_action')}'")
        if state.get("status") == "failed":
            logger.error("Workflow status is 'failed'. Ending.")
            return END

        next_action = state.get("next_action")
        if next_action == "DELEGATE":
            return "delegate_to_worker_node"
        elif next_action == "FINISH":
            return END
        elif next_action == "THINK":
            return "run_orchestrator_node"
        else:
            logger.warning(
                f"Unexpected next_action '{next_action}', defaulting to run_orchestrator_node.")
            return "run_orchestrator_node"

    workflow.add_conditional_edges(
        "run_orchestrator_node",  # Source node
        decide_next_step,       # Function to decide the route
        {                       # Mapping from route decision to next node
            "delegate_to_worker_node": "delegate_to_worker_node",
            "run_orchestrator_node": "run_orchestrator_node",
            END: END
        }
    )

    return workflow.compile()


async def run_alpha_modeling_workflow(user_request: str) -> Dict[str, Any]:
    """
    Run the alpha modeling workflow with the given user request

    Args:
        user_request: User's request for alpha modeling

    Returns:
        Dictionary containing the workflow results
    """
    logger.info(
        f"Starting alpha modeling workflow for request: {user_request}")

    # Initialize the state
    initial_state = OrchestratorState(
        messages=[],
        user_request=user_request,
        tasks={},
        current_task_id=None,
        current_worker=None,
        worker_response=None,
        results={},
        status="in_progress",
        next_action=None
    )

    # Create and run the graph
    graph = create_orchestrator_graph()
    final_state = await graph.ainvoke(initial_state)

    logger.info(f"Workflow completed with status: {final_state['status']}")

    # Return the results
    return {
        "status": final_state["status"],
        "messages": final_state.get("messages", []),
        "results": final_state["results"],
        "tasks": final_state["tasks"]
    }
