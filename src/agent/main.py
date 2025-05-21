"""
Main module for Alpha Modeling Agents
"""
import asyncio
import argparse
import logging
from typing import Dict, Any
import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from .orchestrator import run_alpha_modeling_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AlphaModeling")


async def main():
    """Main entry point for Alpha Modeling Agents"""
    parser = argparse.ArgumentParser(description="Alpha Modeling Agents")
    parser.add_argument("--request", "-r", type=str, required=True,
                        help="User request for alpha modeling")
    args = parser.parse_args()

    try:
        # Run the workflow with the user's request
        results = await run_alpha_modeling_workflow(args.request)

        # Print the results
        print("\n=== Alpha Modeling Workflow Results ===")
        print(f"Status: {results['status']}")
        print("\nConversation Log:")
        for msg_obj in results.get('messages', []):
            role = "Unknown"
            content_to_print = ""

            if isinstance(msg_obj, HumanMessage):
                role = "User"
                content_to_print = msg_obj.content
            elif isinstance(msg_obj, AIMessage):
                role = "AI"
                try:
                    parsed_content = json.loads(msg_obj.content)
                    content_to_print = json.dumps(parsed_content, indent=2)
                except json.JSONDecodeError:
                    content_to_print = msg_obj.content
            elif isinstance(msg_obj, SystemMessage):
                role = "System"
                content_to_print = msg_obj.content
            else:
                role = msg_obj.type if hasattr(msg_obj, 'type') else 'Message'
                content_to_print = msg_obj.content if hasattr(
                    msg_obj, 'content') else str(msg_obj)

            print(f"\n{role}:")
            print(content_to_print)

        print("\n=== Completed Tasks Details ===")
        for task_id, task_details in results.get('tasks', {}).items():
            print(f"\n--- Task ID: {task_id} ---")
            print(f"  Worker: {task_details.get('worker', 'Unknown')}")
            print(f"  Task Type: {task_details.get('task_type', 'Unknown')}")
            print(f"  Status: {task_details.get('status', 'Unknown')}")
            if task_details.get('status') == 'completed' and 'output' in task_details:
                print("  Output:")
                print(json.dumps(task_details['output'], indent=4))
            elif 'error' in task_details:
                print(f"  Error: {task_details['error']}")

    except Exception as e:
        logger.error(
            f"Error running alpha modeling workflow: {str(e)}", exc_info=True)
        # raise

if __name__ == "__main__":
    asyncio.run(main())
