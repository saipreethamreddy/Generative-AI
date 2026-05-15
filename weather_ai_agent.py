"""
=============================================================
   Basic AI Agent using Claude API + Tools
   Tools: DuckDuckGo Search  |  Weather (Open-Meteo, free)
=============================================================

WHAT IS AN AI AGENT?
---------------------
A regular LLM just answers from memory.
An AGENT can decide to USE TOOLS to get real information,
then use that information to form a final answer.

HOW TOOL CALLING WORKS (the agent loop):
  You ──► Claude: "What is the capital of Telangana and today's weather?"
  Claude thinks: "I need to search for the capital, then get weather."
          ↓
  Claude ──► calls tool: duckduckgo_search("capital of Telangana")
  You run the tool ──► return result to Claude
          ↓
  Claude ──► calls tool: get_weather(latitude, longitude)
  You run the tool ──► return result to Claude
          ↓
  Claude reads both results ──► writes final answer ──► You

INSTALL DEPENDENCIES:
  pip install anthropic duckduckgo-search requests

FREE APIS USED (no key needed):
  - DuckDuckGo Search  : web search
  - Open-Meteo         : real-time weather (https://open-meteo.com)
  - Geocoding API      : city name → lat/lon (https://geocoding-api.open-meteo.com)
"""

import os
import creds
import json
import requests
from duckduckgo_search import DDGS
from anthropic import Anthropic

# =============================================================================
# STEP 1 – CONFIGURATION
# =============================================================================

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", creds.anthropic_api_key)
MODEL             = "claude-sonnet-4-5"
MAX_AGENT_LOOPS   = 10   # safety: stop after this many tool-call rounds

client = Anthropic(api_key=ANTHROPIC_API_KEY)

# =============================================================================
# STEP 2 – TOOL IMPLEMENTATIONS (real Python functions)
# =============================================================================

def duckduckgo_search(query: str, max_results: int = 5) -> str:
    """
    Search the web using DuckDuckGo and return the top results as a
    formatted string that Claude can read.
    """
    print(f"   🔍 [Tool] DuckDuckGo search: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            return "No results found."

        output = []
        for i, r in enumerate(results, 1):
            output.append(
                f"Result {i}:\n"
                f"  Title : {r.get('title', 'N/A')}\n"
                f"  Link  : {r.get('href',  'N/A')}\n"
                f"  Snippet: {r.get('body', 'N/A')}"
            )
        return "\n\n".join(output)

    except Exception as e:
        return f"Search error: {e}"


def get_weather(city: str) -> str:
    """
    1. Convert city name → latitude/longitude using the free Open-Meteo
       geocoding API (no key required).
    2. Fetch today's weather from the free Open-Meteo weather API.
    Returns a human-readable weather summary.
    """
    print(f"   🌤️  [Tool] Getting weather for: '{city}'")
    try:
        # ── Geocoding: city name → lat/lon ────────────────────────────────────
        geo_url    = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": city, "count": 1, "language": "en", "format": "json"}
        geo_resp   = requests.get(geo_url, params=geo_params, timeout=10)
        geo_data   = geo_resp.json()

        if "results" not in geo_data or not geo_data["results"]:
            return f"Could not find coordinates for '{city}'."

        location = geo_data["results"][0]
        lat      = location["latitude"]
        lon      = location["longitude"]
        name     = location.get("name", city)
        country  = location.get("country", "")

        # ── Weather: lat/lon → current conditions ─────────────────────────────
        weather_url    = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude"       : lat,
            "longitude"      : lon,
            "current"        : [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "weather_code",
                "wind_speed_10m",
                "precipitation",
            ],
            "timezone"       : "auto",
            "forecast_days"  : 1,
        }
        w_resp = requests.get(weather_url, params=weather_params, timeout=10)
        w_data = w_resp.json()

        current = w_data.get("current", {})

        # Weather code → plain-English description
        code_map = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog",
            51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm with hail",
        }
        code        = current.get("weather_code", 0)
        description = code_map.get(code, f"Weather code {code}")

        return (
            f"Weather in {name}, {country}:\n"
            f"  Condition       : {description}\n"
            f"  Temperature     : {current.get('temperature_2m')}°C\n"
            f"  Feels like      : {current.get('apparent_temperature')}°C\n"
            f"  Humidity        : {current.get('relative_humidity_2m')}%\n"
            f"  Wind speed      : {current.get('wind_speed_10m')} km/h\n"
            f"  Precipitation   : {current.get('precipitation')} mm"
        )

    except Exception as e:
        return f"Weather error: {e}"

# =============================================================================
# STEP 3 – TOOL SCHEMAS (tell Claude what tools exist and how to call them)
# =============================================================================
# Claude reads these JSON schemas to understand:
#   - the tool's name and purpose
#   - what arguments to pass
#   - which arguments are required

TOOLS = [
    {
        "name": "duckduckgo_search",
        "description": (
            "Search the web using DuckDuckGo. Use this to find factual information, "
            "current events, locations, definitions, or anything you need to look up."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query string.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Get the current weather for a city. "
            "Provide the city name (e.g. 'Hyderabad') to get temperature, "
            "humidity, wind speed, and conditions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to get weather for.",
                },
            },
            "required": ["city"],
        },
    },
]

# =============================================================================
# STEP 4 – TOOL DISPATCHER (routes Claude's tool calls to real functions)
# =============================================================================

def run_tool(tool_name: str, tool_input: dict) -> str:
    """
    Claude returns a tool_name + tool_input.
    This function calls the matching Python function and returns the result.
    """
    if tool_name == "duckduckgo_search":
        return duckduckgo_search(**tool_input)
    elif tool_name == "get_weather":
        return get_weather(**tool_input)
    else:
        return f"Unknown tool: {tool_name}"

# =============================================================================
# STEP 5 – THE AGENT LOOP
# =============================================================================

def run_agent(user_query: str) -> str:
    """
    The core agent loop:

    1. Send the user's message to Claude along with the tool schemas.
    2. If Claude decides to use a tool:
         a. Extract tool name + arguments from the response.
         b. Run the actual Python function.
         c. Send the result back to Claude as a "tool_result".
         d. Repeat until Claude stops calling tools.
    3. When Claude's stop_reason is "end_turn", return the final text answer.
    """
    print(f"\n{'='*60}")
    print(f"🧠 Agent received: {user_query}")
    print(f"{'='*60}")

    messages = [{"role": "user", "content": user_query}]

    for loop_num in range(1, MAX_AGENT_LOOPS + 1):
        print(f"\n── Agent loop #{loop_num} ──────────────────────────────────")

        # ── Ask Claude (with tools available) ────────────────────────────────
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 4096,
            tools      = TOOLS,
            messages   = messages,
        )

        print(f"   Stop reason: {response.stop_reason}")

        # ── If Claude is done → return the final answer ───────────────────────
        if response.stop_reason == "end_turn":
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            return final_text

        # ── If Claude wants to use tools ──────────────────────────────────────
        if response.stop_reason == "tool_use":

            # Add Claude's response (with tool_use blocks) to history
            messages.append({"role": "assistant", "content": response.content})

            # Process every tool call Claude made in this turn
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"   🔧 Claude calls tool: '{block.name}' with {block.input}")

                    # Run the real tool
                    result = run_tool(block.name, block.input)
                    print(f"   ✅ Tool result preview: {result[:120]}…")

                    tool_results.append({
                        "type"       : "tool_result",
                        "tool_use_id": block.id,     # must match the tool_use id
                        "content"    : result,
                    })

            # Send tool results back to Claude so it can continue
            messages.append({"role": "user", "content": tool_results})

        else:
            # Unexpected stop reason
            break

    return "Agent reached maximum loops without a final answer."

# =============================================================================
# STEP 6 – MAIN: INTERACTIVE LOOP
# =============================================================================

def main():
    print("=" * 60)
    print("   AI Agent  |  DuckDuckGo Search + Weather")
    print("   Powered by Claude claude-sonnet-4-5")
    print("=" * 60)
    print("\nExample queries:")
    print('  • "What is the capital of Telangana and what is the weather today?"')
    print('  • "Who is the current PM of India and what is the weather in Delhi?"')
    print('  • "What is the largest ocean and what is the weather in Mumbai?"')
    print('\nType "quit" to exit.\n')

    while True:
        query = input("❓ Your query: ").strip()
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            print("\n👋 Goodbye!")
            break

        answer = run_agent(query)
        print(f"\n💬 Final Answer:\n{'-'*60}\n{answer}\n{'='*60}")


if __name__ == "__main__":
    main()
