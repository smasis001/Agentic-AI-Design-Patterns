import os
from dotenv import load_dotenv
from crewai import Task, Crew, Agent
from crewai.tools import tool
from crewai_tools import SerperDevTool
from exa_py import Exa

# Load Environment Variables
load_dotenv(os.path.join(os.getcwd(), './.env'))

# Create a Custom Tool
exa_api_key = os.getenv("EXA_API_KEY")

@tool("Exa search and get contents tool")
def exa_search_tool(search_query: str) -> str:
    """Tool using Exa's Python SDK to run semantic search and return result highlights."""

    exa = Exa(exa_api_key)

    response = exa.search_and_contents(
        search_query,
        type="neural",
        use_autoprompt=True,
        num_results=5,
        highlights=True
    )

    parsedResult = ''.join([
      f'<Title id={idx}>{eachResult.title}</Title>'\
      f'<URL id={idx}>{eachResult.url}</URL>'\
      f'<Highlight id={idx}>{"".join(eachResult.highlights)}</Highlight>'\
      for (idx, eachResult) in enumerate(response.results)
    ])

    return parsedResult

# Initialize Tools
search_tool = exa_search_tool
# search_tool = SerperDevTool(
#   search_url = "https://google.serper.dev/scholar",
#   n_results = 5
# )

# Invoke Tool
if __name__ == "__main__":
  print(search_tool.run(search_query="Agentic AI Design Patterns"))
