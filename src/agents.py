from langchain.agents import AgentExecutor  # Used for executing agent-based workflows
from langchain.agents.format_scratchpad import format_to_openai_function_messages  # Helper function for formatting messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser  # Parser for processing the output of OpenAI functions

# Define a function to create an agent executor
def create_agent_executor(prompt, llm_with_tools, tools):
    # Define an agent using a pipeline of functions
    agent = (
        {
            # The first part of the agent pipeline: Processes input
            "input": lambda x: x["input"],

            # The second part: Formats the intermediate steps to be compatible with OpenAI function messages
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt  # Integrates the given prompt into the pipeline
        | llm_with_tools  # Adds the language model (with tools) to the pipeline
        | OpenAIFunctionsAgentOutputParser()  # Adds an output parser that handles OpenAI function responses
    )
    # Create an AgentExecutor instance with the defined agent, tools, and set verbose mode on
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Return the created agent executor
    return agent_executor
