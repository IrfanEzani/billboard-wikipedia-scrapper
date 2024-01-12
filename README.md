# Web Scraper using OpenAI API and LangChain Utilization

Python script designed to scrape detailed information about songs and artists using the Langchain framework, Wikipedia API and OpenAI's language models. The script processes a list of songs and their respective artists, queries for additional information like genre, label, language, producers, and songwriters, and then compiles the data into a structured format.

## Key Features
Data Scraping: Extracts detailed information about songs from various sources using OpenAI's language models.
Efficient Processing: Implements retry mechanisms to handle potential failures in data scraping.
Dynamic Input and Output: Reads songs and artists from an input CSV file and writes detailed information to an output CSV file.
Langchain Integration: Utilizes the Langchain library for creating and managing AI agents and callbacks.

## Requirements
Python 3.x
Langchain Library
Pandas
Box
PyYAML
Tenacity
OpenAI API Access

## Configuration
The script requires a configuration file (config.yaml) which should contain model names, temperature settings, seeds, and paths to .env files for environment variables.

### Components
Langchain Agents: Implements agents for querying information using OpenAI's language models.
Prompt Templates: Uses the ChatPromptTemplate for structured interaction with the language models.
Retry Mechanism: The tenacity library is used for implementing retry logic in case of failures.
Data Processing: Reads from and writes to CSV files, handling data in an efficient and structured manner.README: Langchain Music Information Scraper


### Running
Run python main.py to execute the web scraping loop for the input songs dataset