"""
Main application file for Alpha Modeling with langgraph-supervisor
This file implements the new langgraph-supervisor workflow.
"""

import asyncio
import uuid
import logging
import argparse
import os
from typing import Dict, Any

import qlib
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor  # Main import for supervisor
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import convert_to_messages

# Import local modules
from .config import (
    GOOGLE_API_KEY,
    AGENT_CONFIG,
    OUTPUT_DIRS,
    QLIB_PROVIDER_URI,
    QLIB_REGION,
    MLFLOW_STORAGE_URL,
    QLIB_DEFAULT_EXP_NAME,
)
from .tools import (
    SUPERVISOR_NODE_NAME,
    DATA_PREP_AGENT_NODE_NAME,
    MODEL_TRAINING_AGENT_NODE_NAME,
    PREDICTION_AGENT_NODE_NAME,
    #    BACKTESTING_AGENT_NODE_NAME,
    # ANALYSIS_AGENT_NODE_NAME,
    supervisor_tools,  # Use the supervisor_tools list for the supervisor
)

# Import tools for each worker agent
from .data_preparation_agent import data_preparation_tools
from .model_training_agent import model_training_tools
from .prediction_generation_agent import prediction_generation_tools

# from .backtesting_agent import backtesting_tools
# from .analysis_reporting_agent import analysis_reporting_tools

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AlphaModelingApp")


def pretty_print_message(message, indent=False):
    """Pretty print a single message with optional indentation"""
    try:
        pretty_message_str = message.pretty_repr(
            html=False
        )  # Use html=False for console
    except Exception:
        pretty_message_str = str(message)  # Fallback

    if not indent:
        print(pretty_message_str)
        return

    indented = "\n".join("\t" + line for line in pretty_message_str.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    """Pretty print messages from workflow updates"""
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        # skip parent graph updates in the printouts
        if len(ns) == 0:
            return

        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label

        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def initialize_qlib():
    """Initialize Qlib with the configured provider URI and region"""
    try:
        # Ensure the Qlib data directory exists
        os.makedirs(QLIB_PROVIDER_URI, exist_ok=True)
        logger.info(f"Ensured Qlib data directory exists: {QLIB_PROVIDER_URI}")

        logger.info(
            f"Initializing Qlib with provider URI: {QLIB_PROVIDER_URI} and region: {QLIB_REGION}"
        )
        qlib.init(
            provider_uri=QLIB_PROVIDER_URI,
            region=QLIB_REGION,
            exp_manager={
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": MLFLOW_STORAGE_URL,
                    "default_exp_name": QLIB_DEFAULT_EXP_NAME,
                },
            },
        )
        logger.info("Qlib initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Qlib: {str(e)}")
        raise


# Define the supervisor system prompt - this guides the supervisor LLM's decision-making
# It should instruct the supervisor on when to delegate to which agent.
SUPERVISOR_SYSTEM_PROMPT = """You are the Supervisor Agent for an Alpha Modeling system.
Your primary role is to understand a user's request for financial model training, prediction, backtesting, and analysis,
and then to orchestrate a team of specialized worker agents to fulfill this request.

You must break down the user's request into a sequence of tasks and delegate each task to the most appropriate worker agent using your available tools.
Always provide clear context to the worker agents. Use the output from one worker as input for the next when necessary.

Available Worker Agents and when to use them:
- Use '{data_prep_node}' to handle data preparation and dataset configuration tasks. This is often the first step.
- Use '{model_train_node}' for model configuration and training tasks. This usually follows data preparation.
- Use '{prediction_node}' to generate predictions from a trained model and calculate initial metrics like IC. This follows model training.

CURRENT WORKFLOW LIMITATION:
Currently, backtesting agent and analysis reporting agent are not implemented due to issues that need to be resolved. 
The workflow supports: data preparation → model training → prediction generation.

WORKFLOW COMPLETION GUIDELINES:
1. For simple requests (e.g., "Create a dataset config"), execute: {data_prep_node} only.
2. For training requests (e.g., "Train a model"), execute: {data_prep_node} → {model_train_node}.
3. For full prediction pipelines (e.g., "Train a model and predict"), execute: {data_prep_node} → {model_train_node} → {prediction_node}.
4. After all agents complete their tasks, YOU (the supervisor) must provide a comprehensive summary of the workflow execution and results.

IMPORTANT DELEGATION PRINCIPLES:
- Each agent should complete their FULL task before moving to the next agent
- Do NOT go back to a previous agent unless there's a critical error that needs fixing
- Pass ALL relevant information and context when delegating to each agent
- The {data_prep_node} should provide the complete dataset configuration dictionary
- The {model_train_node} should provide the trained model details (experiment_name, recorder_id, model_config)
- The {prediction_node} should provide prediction results and IC metrics

TASK DELEGATION INSTRUCTIONS:
- When delegating to {data_prep_node}: Provide data file path, date ranges, and any processor configurations
- When delegating to {model_train_node}: Provide the dataset configuration AND model requirements (type, hyperparameters)
- When delegating to {prediction_node}: Provide experiment_name, recorder_id, dataset_config, and prediction segment

FINAL SUMMARY REQUIREMENTS:
After all worker agents have completed their tasks, YOU must provide a final summary IN TRADITIONAL CHINESE (繁體中文) that includes:
1. **已完成任務**: List of tasks completed by each agent
2. **執行狀態**: Which agents were successfully executed and any issues encountered
3. **主要結果**: 
   - Dataset configuration details (if data preparation was performed)
   - Model training results and metrics (if model training was performed)
   - Prediction results and IC metrics (if prediction generation was performed)
4. **建議**: Next steps or suggestions for the user

LANGUAGE REQUIREMENTS:
- Internal agent communication and task delegation should be in English for clarity and precision
- ALL final summaries and user-facing outputs MUST be in Traditional Chinese (繁體中文)
- Technical terms can be kept in English when appropriate, but explanations should be in Chinese

EFFICIENCY REQUIREMENTS:
- Minimize the number of agent handoffs
- Each agent should do substantial work before handing back control
- Only return control to supervisor when the agent has completed their full task or encountered an error
- Provide clear, concise, and actionable final summary in Traditional Chinese

ERROR HANDLING:
- If a worker agent reports an error, analyze the error and decide whether to:
  1. Retry with corrected parameters
  2. Try an alternative approach
  3. Report the error to the user with clear explanation
- Always provide meaningful responses, never send empty messages
- If model training fails due to parameter issues, suggest parameter corrections

User request: {user_request}
Current conversation:
{messages}
"""

# Function to create the complete alpha modeling workflow


async def create_alpha_modeling_workflow():
    """Create the alpha modeling workflow using langgraph-supervisor"""
    # Initialize Qlib first
    initialize_qlib()

    logger.info("Initializing Alpha Modeling workflow with langgraph-supervisor...")

    # 1. Initialize LLMs
    # LLM for Worker Agents (can be the same or different for each)
    worker_llm = ChatGoogleGenerativeAI(
        # Using data_prep config as a general worker LLM config
        model=AGENT_CONFIG["data_preparation"]["model_name"],
        google_api_key=GOOGLE_API_KEY,
        temperature=0,  # Lower temperature for more deterministic tool usage
    )

    # LLM for the Supervisor Agent
    supervisor_llm = ChatGoogleGenerativeAI(
        # Orchestrator config for supervisor LLM
        model=AGENT_CONFIG["orchestrator"]["model_name"],
        google_api_key=GOOGLE_API_KEY,
        temperature=0,  # Zero temperature for deterministic orchestration
    )

    # 2. Create Worker Agents using create_react_agent

    # Data Preparation Agent
    data_prep_agent = create_react_agent(
        model=worker_llm,
        tools=data_preparation_tools,
        name=DATA_PREP_AGENT_NODE_NAME,
        prompt="You are a Data Preparation Agent specializing in Qlib dataset configuration. Use the available tools to generate valid dataset configurations based on user requirements.",
    )
    logger.info(f"Created {DATA_PREP_AGENT_NODE_NAME}")

    # Model Training Agent
    model_training_agent = create_react_agent(
        model=worker_llm,
        tools=model_training_tools,
        name=MODEL_TRAINING_AGENT_NODE_NAME,
        prompt=(
            "You are a Model Training Agent specializing in Qlib model training. "
            "Follow the workflow: access templates, generate configuration, then train the model using available tools. "
            "After successfully training the model using the 'train_model' tool, you MUST summarize the key results from the tool's output (like experiment name, recorder ID, and status) in your final response before finishing your task."
        ),
    )
    logger.info(f"Created {MODEL_TRAINING_AGENT_NODE_NAME}")

    # Prediction Generation Agent (using placeholder tools)
    prediction_agent = create_react_agent(
        model=worker_llm,
        tools=prediction_generation_tools,
        name=PREDICTION_AGENT_NODE_NAME,
        prompt=(
            "You are a Prediction Generation Agent specializing in model inference and signal quality evaluation. "
            "Your primary task is a TWO-STEP process: "
            "1. Generate predictions from trained models using 'generate_model_predictions_tool'. "
            "2. THEN, using the 'new_experiment' and 'new_recorder_id' from the output of the first tool, calculate signal quality metrics (IC, Rank IC) using 'calculate_signal_metrics_tool'. "
            "Always use the available tools to complete this two-step prediction workflow. "
            "Summarize the results from both steps in your final response. "
            "If any step fails, provide clear error messages and suggest solutions."
        ),
    )
    logger.info(f"Created {PREDICTION_AGENT_NODE_NAME}")

    # # Backtesting Agent (commented out for initial testing)
    # backtesting_agent = create_react_agent(
    #     model=worker_llm,
    #     tools=backtesting_tools,
    #     name=BACKTESTING_AGENT_NODE_NAME,
    #     prompt=(
    #         "You are a Backtesting Agent. "
    #         "Your tasks are to create a backtest configuration and then execute a portfolio backtest. "
    #         "Use 'create_backtest_config_tool' followed by 'execute_backtest_simulation_tool'. "
    #         "You will need predictions and portfolio parameters."
    #     )
    # )
    # logger.info(f"Created {BACKTESTING_AGENT_NODE_NAME}")

    # Analysis & Reporting Agent (using placeholder tools)
    # analysis_agent = create_react_agent(
    #     model=worker_llm,  # Could use a more powerful/creative LLM for summarization
    #     tools=analysis_reporting_tools,
    #     name=ANALYSIS_AGENT_NODE_NAME,
    #     prompt=(
    #         "You are an Analysis and Reporting Agent specialized in financial modeling workflows. "
    #         "Your primary responsibility is to create comprehensive reports from modeling pipeline results.\n\n"
    #         "WORKFLOW:\n"
    #         "1. First, use 'aggregate_results_tool' to collect and organize all task outputs from previous agents\n"
    #         "2. Then, use 'summarize_findings_tool' to generate a professional markdown report\n\n"
    #         "INPUT HANDLING:\n"
    #         "- You will receive outputs from various agents (data preparation, model training, prediction generation)\n"
    #         "- Extract key information like dataset configurations, model details, training results, and metrics\n"
    #         "- Handle cases where some steps may have been skipped or failed\n\n"
    #         "REPORT REQUIREMENTS:\n"
    #         "- Generate clear, professional markdown reports\n"
    #         "- Include executive summary, methodology, results, and recommendations\n"
    #         "- Highlight key metrics, performance indicators, and any warnings\n"
    #         "- Save the report to the outputs/reports directory\n"
    #         "- Provide actionable insights based on the results\n\n"
    #         "ERROR HANDLING:\n"
    #         "- If report generation fails, provide a fallback summary\n"
    #         "- Always attempt to extract and present available information\n"
    #         "- Be robust to incomplete or missing data from previous steps"
    #     ),
    # )
    # logger.info(f"Created {ANALYSIS_AGENT_NODE_NAME}")

    # 3. Create the supervisor workflow
    # The supervisor_tools are the handoff tools.
    # The prompt guides the supervisor's decision-making.
    # output_mode="last_message" can be useful to only pass the final response of a worker to the supervisor,
    # or "full_history" to pass everything. Default is "full_history".
    # add_handoff_back_messages=True ensures the supervisor sees a message when a worker hands back control.
    workflow = create_supervisor(
        agents=[
            data_prep_agent,
            model_training_agent,
            prediction_agent,
            # analysis_agent,
        ],
        model=supervisor_llm,
        prompt=SUPERVISOR_SYSTEM_PROMPT.format(
            data_prep_node=DATA_PREP_AGENT_NODE_NAME,
            model_train_node=MODEL_TRAINING_AGENT_NODE_NAME,
            prediction_node=PREDICTION_AGENT_NODE_NAME,
            user_request="{{user_request}}",
            messages="{{messages}}",
        ),
        tools=supervisor_tools,  # Handoff tools for the supervisor
        # Good for the supervisor to know when a worker is done
        add_handoff_back_messages=True,
        output_mode="full_history",  # Supervisor sees all messages from workers
        supervisor_name=SUPERVISOR_NODE_NAME,
    )
    logger.info("Supervisor workflow created.")
    return workflow


async def run_workflow(user_request: str):
    """Run the alpha modeling workflow with the given user request"""
    logger.info(f"Running workflow with user request: '{user_request}'")

    # Validate user request
    if not user_request or not user_request.strip():
        raise ValueError("User request cannot be empty")

    # Ensure output directories exist
    for directory in OUTPUT_DIRS.values():
        os.makedirs(directory, exist_ok=True)

    # Create the workflow graph
    graph = await create_alpha_modeling_workflow()
    checkpointer = InMemorySaver()
    app = graph.compile(checkpointer=checkpointer)
    logger.info("Workflow compiled.")

    thread_id = str(uuid.uuid4())
    logger.info(f"Generated Thread ID for this run: {thread_id}")

    initial_input = {
        "messages": [{"role": "user", "content": user_request.strip(), "name": "user"}],
        "user_request": user_request.strip(),
    }

    logger.info("Streaming workflow execution with 'updates' mode...")
    print("\n=== Alpha Modeling Workflow Execution - Real-time Log ===")
    print(f"User Request: {user_request}\n")

    accumulated_state_for_summary = {}

    async for event in app.astream_events(
        initial_input, config={"configurable": {"thread_id": thread_id}}, version="v1"
    ):
        kind = event["event"]
        if kind == "on_chain_end":
            # The 'final_state_values' will capture the output of the graph invocation.
            # For the supervisor, this is typically the final set of messages.
            # We need to ensure 'accumulated_state_for_summary' is built correctly.
            # The supervisor's state should contain the final messages.
            # Let's retrieve the full state from the checkpointer at the end.
            pass  # Continue accumulating events

    # Retrieve the final state from the checkpointer
    final_checkpoint = checkpointer.get(
        config={"configurable": {"thread_id": thread_id}}
    )
    if final_checkpoint and final_checkpoint.channel_values:
        # The supervisor's final state, including all messages, should be in channel_values
        # We need to reconstruct a structure similar to what accumulated_state_for_summary expected
        # or directly use the checkpoint for the summary.
        # For simplicity, let's try to populate accumulated_state_for_summary from the checkpoint.
        # The supervisor node's state is typically what we need for the summary.
        if SUPERVISOR_NODE_NAME in final_checkpoint.channel_values:
            accumulated_state_for_summary[SUPERVISOR_NODE_NAME] = {
                "messages": final_checkpoint.channel_values[SUPERVISOR_NODE_NAME]
            }
        # If other nodes' final states are needed for some reason, they could be extracted too.
        # For now, the supervisor's final messages are the primary target for the summary.

    # --- Summary Printing Logic at the End  ---
    print("\n" + "=" * 60)
    print("=== Workflow Execution Summary ===")
    print("=" * 60)

    supervisor_final_messages = []
    # Try to get messages from the accumulated state of the supervisor
    supervisor_state = accumulated_state_for_summary.get(SUPERVISOR_NODE_NAME, {})
    if isinstance(supervisor_state, dict):
        supervisor_raw_messages = supervisor_state.get("messages")
        if supervisor_raw_messages:
            supervisor_final_messages = convert_to_messages(supervisor_raw_messages)

    if supervisor_final_messages:
        final_supervisor_summary_content = None
        for msg in reversed(supervisor_final_messages):
            if (
                hasattr(msg, "type")
                and msg.type == "ai"
                and hasattr(msg, "name")
                and msg.name == SUPERVISOR_NODE_NAME
                and hasattr(msg, "content")
            ):
                content_str = str(msg.content)
                if any(
                    keyword in content_str
                    for keyword in ["已完成任務", "執行狀態", "主要結果", "建議"]
                ):
                    final_supervisor_summary_content = content_str
                    break

        if final_supervisor_summary_content:
            print("\n--- Supervisor's Final Summary ---")
            print(final_supervisor_summary_content)
        else:
            print(
                "\n--- No specific Supervisor summary found. Last few messages from Supervisor: ---"
            )
            for msg_to_print in supervisor_final_messages[-3:]:
                pretty_print_message(msg_to_print)
    else:
        print(
            "Supervisor messages not found in accumulated state or final state was empty."
        )

    print(f"\n{'='*60}")
    print("=== Workflow execution complete ===")
    print("=" * 60)
    logger.info("Workflow execution complete.")
    return accumulated_state_for_summary  # Return the accumulated state


def print_incremental_messages(
    event, shown_message_ids
):  # Renamed from node_message_counts
    """Print only new messages from each node that haven't been shown globally."""
    is_subgraph = False
    if isinstance(event, tuple):
        ns, event_data = event  # Avoid reassigning event, use event_data
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True
    else:
        event_data = event  # If not a tuple, event_data is just event

    for node_name, node_update in event_data.items():
        if not isinstance(node_update, dict) or "messages" not in node_update:
            continue

        current_messages_from_node = convert_to_messages(node_update["messages"])
        new_messages_to_print = []

        # Iterate through all messages provided by the current node in this event
        # The list node_update["messages"] can contain historical messages too
        for msg in current_messages_from_node:
            # Create a unique identifier for each message based on its content and key attributes
            # Using id() is not reliable as messages can be recreated.
            msg_identifier_parts = [
                str(getattr(msg, "content", "")),
                str(getattr(msg, "type", "")),
                str(getattr(msg, "name", "")),
            ]
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                # Add tool call details to the identifier if they exist
                # Sort tool calls by id to ensure consistent hash for same set of calls
                sorted_tool_calls = sorted(
                    msg.tool_calls,
                    key=lambda tc: tc.get("id", "") if isinstance(tc, dict) else "",
                )
                msg_identifier_parts.append(str(sorted_tool_calls))
            else:
                msg_identifier_parts.append("NoToolCalls")

            # Consider additional_kwargs like tool_call_id for ToolMessages
            if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                if "tool_call_id" in msg.additional_kwargs:
                    msg_identifier_parts.append(
                        str(msg.additional_kwargs["tool_call_id"])
                    )

            msg_hash = hash(tuple(msg_identifier_parts))

            if msg_hash not in shown_message_ids:
                shown_message_ids.add(msg_hash)
                new_messages_to_print.append(msg)

        # Print new messages if any
        if new_messages_to_print:
            update_label = f"Update from node {node_name}:"
            if is_subgraph:
                update_label = "\t" + update_label
            print(update_label)
            print("\n")

            for msg_to_print in new_messages_to_print:
                pretty_print_message(msg_to_print, indent=is_subgraph)
            print("\n")


def pretty_print_incremental_chunk(chunk, shown_message_ids):
    """Pretty print an incremental chunk from stream_mode='updates', avoiding duplicates globally."""
    is_subgraph_update = isinstance(chunk, tuple)
    namespace_str = ""
    actual_update_dict = {}

    if is_subgraph_update:
        ns, update_content = chunk
        if not ns:
            return
        namespace_str = " -> ".join(ns) + " "
        actual_update_dict = update_content
    elif isinstance(chunk, dict):
        actual_update_dict = chunk
    else:
        # print(f"Unknown chunk format: {chunk}") # Optional: log unknown formats
        return

    for node_name, node_content in actual_update_dict.items():
        if isinstance(node_content, dict) and "messages" in node_content:
            messages_in_chunk = convert_to_messages(node_content["messages"])
            new_messages_to_print = []

            for msg in messages_in_chunk:
                msg_identifier_parts = [
                    str(getattr(msg, "content", "")),
                    str(getattr(msg, "type", "")),
                    str(getattr(msg, "name", "")),
                ]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    try:
                        sorted_tool_calls = sorted(
                            msg.tool_calls,
                            key=lambda tc: (
                                tc.get("id", "") if isinstance(tc, dict) else str(tc)
                            ),
                        )
                    except TypeError:
                        sorted_tool_calls = sorted(msg.tool_calls, key=str)
                    msg_identifier_parts.append(str(sorted_tool_calls))
                else:
                    msg_identifier_parts.append("NoToolCalls")

                if hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
                    if "tool_call_id" in msg.additional_kwargs:
                        msg_identifier_parts.append(
                            str(msg.additional_kwargs["tool_call_id"])
                        )

                msg_hash = hash(tuple(msg_identifier_parts))

                if msg_hash not in shown_message_ids:
                    shown_message_ids.add(msg_hash)
                    new_messages_to_print.append(msg)

            if new_messages_to_print:
                print(f"Update from {namespace_str}node '{node_name}':")
                for msg_to_print in new_messages_to_print:
                    pretty_print_message(msg_to_print, indent=is_subgraph_update)
                print("\n")


async def print_results(results_state: Dict[str, Any]):
    """
    Simplified results printer - now mostly handled by run_workflow's real-time logging
    This function is kept for CLI compatibility but does minimal work
    """
    print("\n=== print_results called (most output already shown above) ===")

    # Only show a brief summary since detailed output was already shown
    if isinstance(results_state, dict) and "messages" in results_state:
        total_messages = len(results_state.get("messages", []))
        print(
            f"Final state contains {total_messages} total messages in conversation history."
        )
    else:
        print("Final state format not as expected for message counting.")

    print("=== print_results complete ===")


async def main_cli():
    """Main entry point for Alpha Modeling Agents - Command Line Interface"""
    parser = argparse.ArgumentParser(
        description="Alpha Modeling Agents with langgraph-supervisor"
    )
    parser.add_argument(
        "--request",
        "-r",
        type=str,
        required=True,
        help="User request for alpha modeling (e.g., 'Train an XGBoost model for 0050 stocks from 2015-2021, then predict for 2022 and calculate IC metrics.')",
    )
    args = parser.parse_args()

    try:
        logger.info(f"Starting Alpha Modeling CLI with request: {args.request}")
        # Run the workflow with the user's request
        final_state = await run_workflow(args.request)
        await print_results(final_state)
    except Exception as e:
        logger.error(
            f"Error running alpha modeling workflow via CLI: {str(e)}", exc_info=True
        )


async def main_example_run():
    """Example entry point with a hardcoded request for direct testing"""
    logger.info("Starting Alpha Modeling example run...")

    user_req = (
        "訓練一個XGBoost模型（使用預設參數），"
        "訓練期間為2015-2021年，驗證期間為2022年，測試期間為2023年。"
        "然後生成預測結果並計算IC/Rank IC指標。"
    )

    try:
        final_state = await run_workflow(user_req)
        logger.info("Alpha Modeling example run completed.")
        return final_state
    except Exception as e:
        logger.error(f"Error in main_example_run: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # If arguments were provided (e.g., --request "..." ), use the CLI mode
        asyncio.run(main_cli())
    else:
        # No arguments, run the hardcoded example
        asyncio.run(main_example_run())
