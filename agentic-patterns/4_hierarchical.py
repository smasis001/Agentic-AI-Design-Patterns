import os
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool

# Load Environment Variables
load_dotenv(os.path.join(os.getcwd(), './.env'))

# Choose LLM with https://docs.litellm.ai/docs/providers/
llm = "claude-3-haiku-20240307"
fn_llm = "gpt-4o-mini"
planning_llm = "gpt-4o"

# Initialize Tools
search_tool = SerperDevTool()
web_rag_tool = WebsiteSearchTool()

# Create a Researcher Agent
researcher = Agent(
    role="Researcher",
    goal="Expert in technology research",
    backstory=(
        "You're an expert researcher, specialized in technology, software engineering, "
        "AI, and startups. You work as a freelancer and are currently researching for a new client."
    ),
    tools=[search_tool, web_rag_tool],
    allow_delegation=False,
    llm=llm,
    function_calling_llm=fn_llm
)

# Create a Writer Agent
writer = Agent(
    role="Senior Writer",
    goal="Create compelling content about AI and AI agents",
    backstory=(
        "You're a senior writer, specialized in technology, software engineering, "
        "AI, and startups. You work as a freelancer and are currently writing content for a new client."
    ),
    allow_delegation=False,
    llm=llm,
)

# Create a Manager Agent
manager = Agent(
    role="Project Manager",
    goal="Efficiently manage the crew and ensure high-quality task completion",
    backstory=(
        "You're an experienced project manager, skilled in overseeing complex projects and "
        "guiding teams to success. Your role is to coordinate the efforts of the crew members, "
        "ensuring that each task is completed on time and to the highest standard."
    ),
    allow_delegation=True,
    llm=planning_llm
)

# Define your task
task = Task(
    description=(
        "Generate a list of 5 interesting ideas for an article about {topic}, then write one captivating "
        "paragraph for each idea that showcases the potential of a full article on this topic. "
        "Return the list of ideas with their paragraphs and your notes."
    ),
    expected_output="5 bullet points, each with a paragraph and accompanying notes.",
)

# Create a Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task],
    manager_agent=manager,
    process=Process.hierarchical,
    respect_context_window=True,
    memory=True,
    planning=True,
    planning_llm=planning_llm,
    verbose=True
)

# Kickoff the Crew with an Input
if __name__ == "__main__":
    inputs = {
        'topic': 'Agentic AI'
    }
    response = crew.kickoff(inputs=inputs)
    print(response)
