# Agentic AI Design Patterns

## Description

There are at least four main agentic AI design patterns: reflection, tool-use, planning and multi-agent. At a basic level they are execution workflows where there's one or more large language model processing inputs and this might change the flow dynamically. 

Andrew Ng discusses patterns in more detail [here](https://www.deeplearning.ai/the-batch/issue-242/) and [here](https://www.youtube.com/watch?v=sal78ACtGTc).

This repo is quick overview of some agentic patterns using `crewai` code and library. This library was chosen because of it's didactic nature.

Similar patterns can be represented in [autogen](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/reflection.html), [Langgraph](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/tutorials/reflection/reflection.ipynb) and [smolagents](https://github.com/huggingface/smolagents) with a bit more effort but, given how customizable they are, depending on the use case, might be better than CrewAI. Other popular agentic frameworks such as [Phidata](https://docs.phidata.com/introduction), [MetaGPT](https://github.com/geekan/MetaGPT) and [swarm](https://github.com/openai/swarm) are not as malleable.

## Project Structure

```
.
└── Music-GenAI-App/
    ├── agentic-patterns/
    │   ├── __init__.py
    │   ├── 0_intro.py
    │   ├── 1a_tool_test.py
    │   ├── 1b_tool_use.py
    │   ├── 2_reflect.py
    │   ├── 3_planning.py
    │   └── 4_hierarchical.py
    ├── pyproject.toml
    ├── requirements.txt
    └── README.md
```

## Getting Started

### Install Basic Requirements

You'll need Python 3.11 and above, and GIT. On Windows, the best way is via the [Python website](https://python.org/downloads/windows/), and [GIT website](https://git-scm.com/downloads/win). On other operating systems command line options might be the easiest:

**MacOs**:

```sh
brew install python git
```
(assuming you have brew installed)

**Linux**:

```sh
sudo apt install python3 python-is-python3 git
```

It is recommended that you open the folder where this repository will be cloned with an IDE like VSCode or PyCharm.

### Create Accounts and Get API keys

- [OpenAI](https://platform.openai.com/)
- [Anthropic](https://docs.anthropic.com/en/api/getting-started)
- [Exa](https://exa.ai/)
- [Serper](https://serper.dev/)

Then, open `.env.example` and store values for each one of the keys and rename file as `.env`.

### Create Environment

First you need to clone this repository, and then create the environment. You can use `pyenv`, `conda` or whichever one you want but for this example we are using `venv` which is included in `python`.

```sh
cd /path/where/you/want/to/clone/repo
git clone https://github.com/smasis001/Agentic-AI-Design-Patterns.git
python -m venv .venv
```

Then for managing the packages, this repository comes with `pyproject.toml` for `poetry`, but you can use `requirements.txt` and `pip`. For poetry, this is how it goes:

```sh
pip install poetry
python -m poetry env use .venv/bin/python
python -m poetry install
source .venv/bin/activate
```

For installing the packages with `pip`, you must activate the virtual enviroment first.

```sh
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Scripts locally

To run in the command line any of the scripts do as follows: 


```sh
python agentic-patterns/0_intro.py
```
:
```sh
python agentic-patterns/4_hierarchical.py
```


