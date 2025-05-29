"""
Tools for Alpha Modeling Agents

Includes handoff tools for the Orchestrator (Supervisor) to delegate tasks to Worker agents.
"""

from langgraph_supervisor import create_handoff_tool

# Define worker agent node names (will be used in the supervisor graph)
DATA_PREP_AGENT_NODE_NAME = "DataPreparationAgent"
MODEL_TRAINING_AGENT_NODE_NAME = "ModelTrainingAgent"
PREDICTION_AGENT_NODE_NAME = "PredictionGenerationAgent"
# BACKTESTING_AGENT_NODE_NAME = "BacktestingAgent"
ANALYSIS_AGENT_NODE_NAME = "AnalysisReportingAgent"

# Create handoff tools
# These descriptions help the supervisor LLM decide which agent to delegate to.
handoff_to_data_prep = create_handoff_tool(
    agent_name=DATA_PREP_AGENT_NODE_NAME,
    description="Delegates tasks related to data preparation and Qlib DatasetH configuration. This is usually the first step in the pipeline if data is not already prepared.",
)

handoff_to_model_training = create_handoff_tool(
    agent_name=MODEL_TRAINING_AGENT_NODE_NAME,
    description="Delegates tasks for model configuration and training. This step requires a dataset configuration, typically from the DataPreparationAgent.",
)

handoff_to_prediction = create_handoff_tool(
    agent_name=PREDICTION_AGENT_NODE_NAME,
    description="Delegates tasks for generating model predictions and calculating signal-level metrics (e.g., IC, Rank IC). This step requires a trained model and a dataset configuration.",
)

# handoff_to_backtesting = create_handoff_tool(
#     agent_name=BACKTESTING_AGENT_NODE_NAME,
#     description="Delegates tasks for performing portfolio backtesting using the generated predictions. This step requires predictions and a dataset configuration."
# )

handoff_to_analysis = create_handoff_tool(
    agent_name=ANALYSIS_AGENT_NODE_NAME,
    description="Delegates tasks to aggregate all results from previous steps (data prep, training, prediction, backtesting) and generate a final comprehensive report. This is typically the last step.",
)

# List of handoff tools for the supervisor.
# The supervisor LLM will be given these tools to orchestrate the workflow.
supervisor_tools = [
    handoff_to_data_prep,
    handoff_to_model_training,
    handoff_to_prediction,
    # handoff_to_backtesting,
    handoff_to_analysis,
]

# The default name for the supervisor node within the graph created by `create_supervisor` is "supervisor".
SUPERVISOR_NODE_NAME = "OrchestratorAgent"
