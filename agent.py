from pydantic import BaseModel
from typing import List, Literal, Optional

from dotenv import load_dotenv
load_dotenv()   

class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str

class ToolResult(BaseModel):
    tool_name: str
    result: str
    error: Optional[str] = None

class AgentState(BaseModel):
    task: str
    messages: List[Message] = []
    tool_results: List[ToolResult] = []
    scratchpad: str = ""
    iteration: int = 0
    max_iterations: int = 10
    error_count: int = 0
    status: Literal["running", "done", "failed"] = "running"
    final_answer: Optional[str] = None


import math

def web_search(query: str) -> dict:
    # Mock — in production this would call SerpAPI, Tavily, etc.
    mock_results = {
        "python creator": "Python was created by Guido van Rossum in 1991.",
        "eiffel tower height": "The Eiffel Tower is 330 meters tall.",
        "current president india": "The current President of India is Droupadi Murmu.",
        "rajgir location": "Rajgir is a city in Nalanda district, Bihar, India.",
    }
    query_lower = query.lower()
    for key, value in mock_results.items():
        if any(word in query_lower for word in key.split()):
            return {"result": value, "error": None}
    return {"result": f"No results found for: {query}", "error": None}


def calculator(expression: str) -> dict:
    try:
        # Safe eval — only allows math operations
        allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return {"result": str(result), "error": None}
    except Exception as e:
        return {"result": None, "error": f"Calculation failed: {str(e)}"}


def get_weather(city: str) -> dict:
    # Mock — in production this would call OpenWeatherMap, etc.
    mock_weather = {
        "delhi": "32°C, sunny, humidity 45%",
        "rajgir": "28°C, partly cloudy, humidity 60%",
        "mumbai": "30°C, humid, humidity 80%",
        "london": "15°C, overcast, humidity 70%",
    }
    result = mock_weather.get(city.lower())
    if result:
        return {"result": f"Weather in {city}: {result}", "error": None}
    return {"result": f"Weather data not available for {city}", "error": None}


TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "get_weather": get_weather,
}


def dispatch_tool(name: str, args: dict) -> ToolResult:
    if name not in TOOLS:
        return ToolResult(
            tool_name=name,
            result="",
            error=f"Unknown tool '{name}'. Available tools: {list(TOOLS.keys())}"
        )
    result = TOOLS[name](**args)
    return ToolResult(
        tool_name=name,
        result=result.get("result") or "",
        error=result.get("error")
    )

SYSTEM_PROMPT = """You are a ReAct agent. You solve tasks by interleaving Thought, Action, and Observation steps.

For every turn, you MUST respond in exactly this format:

Thought: [your reasoning about what to do next]
Action: [one of the following]

Available actions:
- web_search({"query": "your search query"})
- calculator({"expression": "math expression"})
- get_weather({"city": "city name"})
- final_answer({"answer": "your final answer to the task"})

Rules:
- Always start with Thought:
- Always follow Thought with exactly one Action:
- Only use the actions listed above
- When you have enough information to answer, use final_answer
- Never make up information — if you don't know, use web_search
- Do not explain yourself outside the Thought/Action format

Example:
Thought: I need to find the height of the Eiffel Tower.
Action: web_search({"query": "Eiffel Tower height"})

Thought: I now have the height. The user asked for it in feet so I need to convert from meters.
Action: calculator({"expression": "330 * 3.28084"})

Thought: I have the answer in both meters and feet.
Action: final_answer({"answer": "The Eiffel Tower is 330 meters or approximately 1082 feet tall."})
"""

import os
import re
import json
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    system_instruction=SYSTEM_PROMPT
)


def parse_action(text: str) -> tuple[str, dict]:
    """
    Parses the Action line from LLM output.
    Returns (tool_name, args_dict)
    """
    # Match pattern: tool_name({"key": "value"})
    match = re.search(r'Action:\s*(\w+)\((\{.*?\})\)', text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse action from response:\n{text}")
    
    tool_name = match.group(1)
    args_raw = match.group(2)
    
    try:
        args = json.loads(args_raw)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in action args: {args_raw}")
    
    return tool_name, args


def extract_thought(text: str) -> str:
    """Extracts the Thought line from LLM output."""
    match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def build_messages(state: AgentState) -> list:
    """Converts agent state messages to Gemini format."""
    gemini_messages = []
    for msg in state.messages:
        if msg.role == "user":
            gemini_messages.append({
                "role": "user",
                "parts": [msg.content]
            })
        elif msg.role in ("assistant", "tool"):
            gemini_messages.append({
                "role": "model",
                "parts": [msg.content]
            })
    return gemini_messages


def run_agent(task: str) -> AgentState:
    """Main agent loop."""
    state = AgentState(task=task)

    # Inject the task as the first user message
    state.messages.append(Message(role="user", content=task))

    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")

    while state.status == "running":

        # Guard: max iterations
        if state.iteration >= state.max_iterations:
            state.status = "failed"
            state.final_answer = "Max iterations reached without completing the task."
            print("\n[AGENT] Max iterations hit — stopping.")
            break

        # Guard: too many consecutive errors
        if state.error_count >= 3:
            state.status = "failed"
            state.final_answer = "Too many consecutive errors — stopping."
            print("\n[AGENT] Too many errors — stopping.")
            break

        print(f"\n--- Iteration {state.iteration + 1} ---")

        # Call Gemini
        try:
            history = build_messages(state)
            # Last message is the current user turn
            current = history[-1]["parts"][0]
            past = history[:-1]

            chat = model.start_chat(history=past)
            response = chat.send_message(current)
            llm_output = response.text

        except Exception as e:
            print(f"[LLM ERROR] {e}")
            state.error_count += 1
            state.iteration += 1
            continue

        print(f"[LLM OUTPUT]\n{llm_output}")

        # Parse thought
        thought = extract_thought(llm_output)
        if thought:
            state.scratchpad += f"\nIteration {state.iteration + 1} — {thought}"

        # Parse and dispatch action
        try:
            tool_name, args = parse_action(llm_output)
        except ValueError as e:
            print(f"[PARSE ERROR] {e}")
            state.error_count += 1
            state.iteration += 1
            # Tell the LLM it made a format mistake
            state.messages.append(Message(
                role="tool",
                content="Error: Your response did not follow the required format. Always respond with Thought: followed by Action:"
            ))
            continue

        # Check for final answer
        if tool_name == "final_answer":
            state.final_answer = args.get("answer", "No answer provided.")
            state.status = "done"
            print(f"\n[FINAL ANSWER] {state.final_answer}")
            break

        # Dispatch tool
        tool_result = dispatch_tool(tool_name, args)
        state.tool_results.append(tool_result)

        # Format observation and inject back into messages
        if tool_result.error:
            observation = f"Error from {tool_name}: {tool_result.error}"
            state.error_count += 1
        else:
            observation = f"Observation: {tool_result.result}"
            state.error_count = 0  # Reset on success

        print(f"[TOOL] {tool_name}({args}) → {observation}")

        # Append LLM output and observation to message history
        state.messages.append(Message(role="assistant", content=llm_output))
        state.messages.append(Message(role="tool", content=observation))

        state.iteration += 1

    print(f"\n{'='*60}")
    print(f"STATUS: {state.status} | Iterations: {state.iteration}")
    print(f"{'='*60}\n")

    return state

if __name__ == "__main__":
    # Set your API key as environment variable before running:
    # Windows: set GEMINI_API_KEY=your_key_here
    # Mac/Linux: export GEMINI_API_KEY=your_key_here
    
    
    result = run_agent("What is the weather in Delhi and what is 15% of 3200?")
    print(f"Answer: {result.final_answer}")

