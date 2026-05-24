# React Agent

A simple Python ReAct-style agent that uses Gemini via the `google-generativeai` SDK to solve tasks by interleaving thoughts, tool actions, and observations.

## Project structure

- `agent.py` - main agent implementation and tool dispatch logic.
- `.gitignore` - configured to ignore `env/` and `venv/` virtual environment directories.

## Requirements

- Python 3.10+
- `pydantic`
- `python-dotenv`
- `google-generativeai`

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv env
env\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install pydantic python-dotenv google-generativeai
```

3. Set your Gemini API key:

```powershell
set GEMINI_API_KEY=your_key_here
```

## Run

```powershell
python agent.py
```

The script runs a small example task by default. Update the `run_agent(...)` call in `agent.py` to change the task.

## Notes

- The current tool implementations are mocked for `web_search`, `calculator`, and `get_weather`.
- The agent expects responses in a strict `Thought:` / `Action:` format.
- If you want to persist environment variables, create a `.env` file and load it with `python-dotenv`.
