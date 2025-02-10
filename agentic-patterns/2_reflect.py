import os
from typing import Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew
from crewai.tools import tool
from crewai.flow.flow import Flow, or_, listen, router, start
from pydantic import BaseModel
from exa_py import Exa

# Load Environment Variables
load_dotenv(os.path.join(os.getcwd(), './.env'))

# Choose LLM with https://docs.litellm.ai/docs/providers/
llm = "gpt-4"
fn_llm = "gpt-4o-mini"
editor_llm = "claude-3-haiku-20240307"

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

# Create a Research Crew
research_crew = Crew(
  agents=[researcher],
  tasks=[research_task],
  verbose=True
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
  llm=llm
)

# Create a Writer Task
write_article = Task(
  description=(
    "Write an article on {topic}."
    "Your article should be engaging, informative, and accurate based on"
    "the following research: "
    "{report}"
    "Please incorporate the following feedback if present: "
    "{feedback}"
  ),
  expected_output = (
    "A comprehensive article no more than 4 paragraphs long on the {topic}."
    "Should cover at least 4 categories or concepts of {topic}."
  ),
  agent=article_writer
)

# Create a Writer Crew
writer_crew = Crew(
  agents=[article_writer],
  tasks=[write_article],
  verbose=True
)

# Create a Editor Agent
article_editor = Agent(
  role="Editor",
  goal="Proofread an article on {topic} to meet strict guidelines.",
  verbose=True,
  memory=True,
  backstory=(
    "You are a careful reviewer, skilled at understanding the core message of an article. "
    "Your job is to maintain the clarity and brevity of an article ensuring it contains "
    "no emojis, unnecessary commentary, or excessive verbosity"
  ),
  allow_delegation=False,
  llm=llm
)

# Model for Verification
class ArticleVerification(BaseModel):
    valid: bool
    feedback: Optional[str]

# Create a Proofreadering Task
proofread_article = Task(
  description=(
    "Verify that the article on {topic} meets the following criteria:"
    "- No more than four paragraphs long."
    "- Covers at least 4 categories or concepts of {topic}."
    "- It contains no emojis."
    "- It contains only the article itself, without additional commentary."
    "- Includes one bold statement to hook the reader."
    "If you believe there are any issues with the article or ways it could be "
    "improved, such as the structure of the article, rhythm, word choice, please "
    "provide feedback. If any of the criteria are not met, the article is considered "
    "invalid. Provide actionable changes about what is wrong and what actions need "
    "to be taken to fix it. Your final response must include:"
    "- Valid: True/False."
    "- Feedback: Provide commentary if the article fails any of the criteria."
    "Article to Verify: "
    "{article}"
  ),
  expected_output = (
    "Pass: True/False. "
    "Feedback: Commentary here if failed."
  ),
  output_pydantic=ArticleVerification,
  agent=article_editor
)

# Create a Editor Crew
editor_crew = Crew(
  agents=[article_editor],
  tasks=[proofread_article],
  verbose=True
)

# Create the Flow State Object
class ArticleFlowState(BaseModel):
    report: str = ""
    article: str = ""
    valid: bool = False
    retry_count: int = 0
    feedback: Optional[str] = ""

# Create the Flow
class VerifiedArticleFlow(Flow[ArticleFlowState]):
    topic = "Latest AI Developments"

    @start()
    def research(self):
        result = (
            research_crew
            .kickoff(inputs={"topic": self.topic})
        )
        self.state.report = result.raw

    @listen(or_(research, "retry"))
    def write_article(self):
        result = (
            writer_crew
            .kickoff(inputs={"topic": self.topic, "report":self.state.report, "feedback": self.state.feedback})
        )
        self.state.article = result.raw

    @router(write_article)
    def validate_article(self):
        if self.state.retry_count > 3:
            return "max_retry_exceeded"

        result = (
            editor_crew
            .kickoff(inputs={"topic":self.topic, "article": self.state.article})
        )
        self.state.valid = result["valid"]
        self.state.feedback = result["feedback"]

        print("valid: ", self.state.valid)
        print("feedback: ", self.state.feedback)
        self.state.retry_count += 1

        if self.state.valid:
            return "completed"

        print("RETRY")
        return "retry"

    @listen("completed")
    def save_result(self):
        print("article is valid")
        print("article:", self.state.article)

        # Save the valid article to a file
        with open("article.txt", "w") as file:
            file.write(self.state.article)

    @listen("max_retry_exceeded")
    def max_retry_exceeded_exit(self):
        print("Max retry count exceeded")
        print("article:", self.state.article)
        print("feedback:", self.state.feedback)


# Kickoff the Flow
if __name__ == "__main__":
    verified_article = VerifiedArticleFlow()
    #verified_article.plot()
    verified_article.topic = "Agentic AI Design Patterns"
    verified_article.kickoff()
