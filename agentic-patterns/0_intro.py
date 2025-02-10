import os
from dotenv import load_dotenv
from crewai import Agent, Task, Process, Crew

# Load Environment Variables
load_dotenv(os.path.join(os.getcwd(), './.env'))

# Choose LLM with https://docs.litellm.ai/docs/providers/
llm = "gpt-4o"

# Create an Agent
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
    debug=True
)

# Create a Task
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
    agents=[article_writer],
    tasks=[write_article],
    memory=True,
    cache=True
)

# Kickoff that Crew with an Input
if __name__ == "__main__":
    inputs = {
        'topic': 'Agentic AI Design Patterns'
    }
    response = crew.kickoff(inputs=inputs)
