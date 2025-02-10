
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
from crewai.tools import tool
from crewai_tools import SerperDevTool
from exa_py import Exa

# Load Environment Variables
load_dotenv(os.path.join(os.getcwd(), './.env'))

# Choose LLM with https://docs.litellm.ai/docs/providers/
llm = "claude-3-haiku-20240307"
fn_llm = "gpt-4o-mini"

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

    parsed_result = ''.join([
      f'<Title id={idx}>{eachResult.title}</Title>'\
      f'<URL id={idx}>{eachResult.url}</URL>'\
      f'<Highlight id={idx}>{"".join(eachResult.highlights)}</Highlight>'\
      for (idx, eachResult) in enumerate(response.results)
    ])

    return parsed_result

# Initialize Tools
search_tool = exa_search_tool
# search_tool = SerperDevTool(
#   search_url = "https://google.serper.dev/scholar",
#   n_results = 5
# )

# Create a Researcher Agent
researcher = Agent(
  role="Researcher",
  goal="Get the latest research on {topic}",
  verbose=True,
  memory=True,
  backstory=(
    "Driven by curiosity, you're at the forefront of"
    "innovation, eager to explore and share knowledge that could change"
    "the world."
  ),
  tools=[search_tool],
  allow_delegation=False,
  llm=llm,
  function_calling_llm=fn_llm
)

# Create a Researching Task
research_task = Task(
  description=(
    "Identify the latest research in {topic} keeping in mind that it's currently February 2025. "
    "Your final report should clearly articulate the key points as bullets"
  ),
  expected_output='A succint report on the {topic}.',
  tools=[search_tool],
  agent=researcher,
)

# Create a Writer Agent
article_writer = Agent(
  role="Article Writer",
  goal="Write a great article on {topic}",
  verbose=True,
  memory=True,
  backstory=(
    "Driven by a love of writing and passion for"
    "innovation, you are eager to share knowledge with"
    "the world."
  ),
  allow_delegation=False,
  llm=llm,
)

# Create a Writer Task
write_article = Task(
  description=(
    "Write an article on {topic}."
    "Your article should be engaging, informative, and accurate."
  ),
  expected_output = (
    "A comprehensive article no more than 4 paragraphs long on the {topic}."
    "Should cover at least 4 categories of {topic}."
  ),
  agent=article_writer
)

# Create a Crew
crew = Crew(
  agents=[researcher, article_writer],
  tasks=[research_task, write_article],
  process=Process.sequential,
  memory=True,
  cache=True
)

# Kickoff that Crew with an Input
if __name__ == "__main__":
  inputs = {
      'topic': 'Agentic AI Design Patterns'
  }
  response = crew.kickoff(inputs=inputs)

