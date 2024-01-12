# Importing necessary libraries and modules
import csv
import json
import os
import box
import pandas as pd
import yaml
from langchain.callbacks import get_openai_callback
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.render import format_tool_to_openai_function
from tenacity import retry, stop_after_attempt, wait_fixed
from src.agents import create_agent_executor
from src.llm import llm
from src.tools import wikipedia_tool
from src.prompts import system_prompt, generate_input_prompt
from src.utils import default_values

# Load configuration settings from a YAML file
with open("config.yaml", "r", encoding="utf8") as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

# Define tools to be used
tools = [wikipedia_tool]
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# Define the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),  # System's part of the conversation
        ("user", "{input}"),        # User's input placeholder
        MessagesPlaceholder(variable_name="agent_scratchpad"),  # Placeholder for intermediate steps or scratchpad
    ]
)

# Create an agent executor with the defined prompt and tools
agent_executor = create_agent_executor(
    prompt=prompt, llm_with_tools=llm_with_tools, tools=tools
)

# Define a retry mechanism for the web scraping function
@retry(
    stop=stop_after_attempt(2),  # Stop retrying after 2 attempts
    wait=wait_fixed(10),         # Wait for 10 seconds before retrying
    retry_error_callback=default_values  # Default values to return in case of error
)
def execute_web_scraping(
    input_file_path: str = cfg.INPUT_FILE, output_file_path: str = cfg.OUTPUT_FILE
):
    df = pd.read_csv(input_file_path)  # Read input CSV file into a DataFrame
    for _, row in df.iterrows():  # Iterate through each row in the DataFrame
        song, artist = row["song"], row["artist"]  # Extract song and artist from the row

        # Check if output file exists, and read it if it does
        if os.path.exists(output_file_path):
            df_song_info = pd.read_csv(output_file_path, encoding="utf-8")
        else:
            # If output file doesn't exist, create a new DataFrame for song information
            df_song_info = pd.DataFrame(
                columns=[
                    "artist",
                    "song",
                    "genre",
                    "label",
                    "language",
                    "llm_cost",
                    "llm_tokens",
                    "producers",
                    "songwriters",
                ]
            )
            df_song_info.to_csv(output_file_path, index=False)  # Save the DataFrame to CSV

        # Process the song if it's not already in the output file
        if song not in df_song_info["song"].tolist():
            print(f"***** Processing: {song} by {artist} *****")
            input_prompt = generate_input_prompt(song, artist)  # Generate the input prompt for the song

            # Use OpenAI callback for the agent execution
            with get_openai_callback() as cb:
                response = agent_executor.invoke({"input": input_prompt})
                cost = cb.total_cost
                tokens = cb.total_tokens
                output = response["output"]
                print(output)
                output_dict = json.loads(output)  # Parse the JSON response

                # Prepare a new row with the extracted information
                new_row = {
                    "artist": artist,
                    "song": song,
                    "genre": output_dict.get("genre"),
                    "label": output_dict.get("label"),
                    "language": output_dict.get("language"),
                    "llm_cost": cost,
                    "llm_tokens": tokens,
                    "producers": output_dict.get("producers"),
                    "songwriters": output_dict.get("songwriters"),
                }

                # Append the new row to the output CSV file
                with open(output_file_path, "a", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(new_row.values())

# Main entry point of the script
if __name__ == "__main__":
    execute_web_scraping()  # Execute the web scraping function
