#LangChain comes with a host of third-party integrations which we can utilize as tools. 
# While we can define multiple tools, we will focus on just setting up the Wikipedia search tool.
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from langchain.agents import Tool

#We do so by passing the WikipediaAPIWrapper into the func parameter of LangChain’sTool() class
wikipedia_tool = Tool(
    name="wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Useful for when you need to look up the songwriters, genre, \
                and producers for a song on wikipedia",
)
#tool descriptions must be clear and specific since the LLM agent reads them to decide whether to invoke the tool to answer the input queries.

duckduckgo_tool = Tool(
    name="DuckDuckGo_Search",
    func=DuckDuckGoSearchRun().run,
    description="Useful for when you need to do a search on the internet to find \
                information that the other tools can't find.",
)

# After defining the tools, we place them in a list and bind them (equivalent to passing arguments) to the LLM object to run
# We use format_tool_to_openai_function to render our list of tools specifically for OpenAI function calling. 
# Under the hood, it formats the tools into a dictionary structure based on their name, description, and any other arguments we pass.