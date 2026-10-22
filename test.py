# import asyncio
# from mcp_client import get_all_tools,tavily_mcp_search  



# if __name__ == "__main__":
#     asyncio.run(get_all_tools())
import os
import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()   
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()  
from typing import TypedDict, List,Annotated
import operator 
import uuid
import asyncio  
import psycopg
from psycopg.rows import dict_row
from mcp_client import tavily_mcp_search,aviation_mcp_call,destination_extractor,forecast_mcp_search,weather_mcp_search
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)

from flight_tool_updated import get_raw_flights, extract_airports_and_airlines

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Raw live flight snapshot (for grounding, not to be dumped verbatim):
{flight_data}

Generate:

1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def _is_mcp_restricted_or_failed(result) -> bool:
    """
    True if an MCP call result indicates the endpoint is unavailable
    (access restricted, 403, timeout, exception string, None, etc.).
    """
    if result is None:
        return True

    text = str(result).lower()

    restricted_markers = [
        "function_access_restricted",
        "403",
        "forbidden",
        "unauthorized",
        "access_restricted",
        "error",
        "exception",
    ]

    return any(marker in text for marker in restricted_markers)


def _get_airports_and_airlines_from_mcp(query: str):
    """
    Attempts the MCP server lookups. Returns (airports, airlines, ok)
    where ok is False if the MCP call failed, was restricted, or raised.
    Never raises - any problem is reported back via ok=False.
    """
    try:
        mcp_airports = asyncio.run(aviation_mcp_call("list_airports"))
        mcp_airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        print("\nAIRPORTS (MCP):", mcp_airports)
        print("\nAIRLINES (MCP):", mcp_airlines)

        if _is_mcp_restricted_or_failed(mcp_airports) or _is_mcp_restricted_or_failed(mcp_airlines):
            print("\nMCP airport/airline lookup unavailable - will fall back to AviationStack extraction.\n")
            return None, None, False

        return mcp_airports, mcp_airlines, True

    except NameError:
        # aviation_mcp_call isn't defined/imported in this environment at all.
        print("\naviation_mcp_call is not available - will fall back to AviationStack extraction.\n")
        return None, None, False

    except Exception as e:
        print(f"\nMCP call raised an exception, falling back: {e}\n")
        return None, None, False


def _get_airports_and_airlines_from_aviationstack(query: str):
    """
    Fallback: derive airports/airlines from live flight data for this
    query via search_flights.get_raw_flights() + extract_airports_and_airlines().

    Returns (airports, airlines, raw_flights, error_message).
    error_message is None on success; if set, the caller should return it
    directly as flight_results without calling the LLM.
    """
    raw_flights = get_raw_flights(query)

    if isinstance(raw_flights, str):
        return [], [], [], raw_flights

    airports, airlines = extract_airports_and_airlines(raw_flights)

    if not airports and not airlines:
        return (
            [],
            [],
            raw_flights,
            "No live flight data was found for this route, so airport "
            "and airline details could not be derived.",
        )

    return airports, airlines, raw_flights, None


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")

    query = state["user_query"]
    airports = None
    airlines = None
    raw_flights = []
    source = None  # "mcp" or "aviationstack", for logging/debugging

    try:
        # 1. Try the MCP server first.
        mcp_airports, mcp_airlines, mcp_ok = _get_airports_and_airlines_from_mcp(query)

        if mcp_ok:
            airports = mcp_airports
            airlines = mcp_airlines
            source = "mcp"

        # 2. Fallback: derive from live AviationStack flight data.
        if source is None:
            fb_airports, fb_airlines, fb_raw_flights, fb_error = (
                _get_airports_and_airlines_from_aviationstack(query)
            )

            if fb_error:
                return {
                    "flight_results": fb_error,
                    "messages": [
                        AIMessage(content="Flight data unavailable.")
                    ],
                    "llm_calls": state.get("llm_calls", 0),
                }

            airports = fb_airports
            airlines = fb_airlines
            raw_flights = fb_raw_flights
            source = "aviationstack"

        # 3. Build the prompt from whichever source succeeded.
        print(f"\nUsing airport/airline data from: {source}\n")

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
            flight_data=str(raw_flights[:5])[:3000] if raw_flights else "N/A (sourced from MCP, not live flight records)",
        )

        response = llm.invoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e:
        flight_data = f"Flight information unavailable: {str(e)}"
        return {
            "flight_results": flight_data,
            "messages": [
                AIMessage(content="Flight information unavailable.")
            ],
            "llm_calls": state.get("llm_calls", 0),
        }

    return {
        "flight_results": flight_data,
        "messages": [
            AIMessage(content="Flight recommendations generated")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }