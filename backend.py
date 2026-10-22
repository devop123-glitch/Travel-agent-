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
from tools.flight_tool import search_flights        
from tools.tavily_tool import tavily_search
from tools.flight_tool_updated import  get_raw_flights, extract_airports_and_airlines

def get_database_url():
    database_url=os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set in the environment variables.")
    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url=f"{database_url}{separator}sslmode=require"

    return database_url    

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")


llm=ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    max_tokens=3000,
    groq_api_key=GROQ_API_KEY
)

class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flight_results:str
    hotel_results:str
    itinerary:str
    llm_calls:int
    weather_results:str

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

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

# def flight_agent(state: TravelState):
#     print("\nINSIDE FLIGHT AGENT\n")

#     query = state["user_query"]

#     try:

#         airports = asyncio.run(
#             aviation_mcp_call(
#                 "list_airports"
#             )
#         )

#         airlines = asyncio.run(
#             aviation_mcp_call(
#                 "list_airlines"
#             )
#         )


#         print("\nAIRPORTS:", airports)
#         print("\nAIRLINES:", airlines)

#         if (
#             "function_access_restricted" in str(airports)
#             or "function_access_restricted" in str(airlines)
#         ):
#             flight_data = (
#                 "Flight reference data is unavailable because the current "
#                 "AviationStack subscription does not allow the airport and "
#                 "airline lookup endpoints. Upgrade the plan or use another "
#                 "flight-data provider to enable these lookups."
#             )
#             return {
#                 "flight_results": flight_data,
#                 "messages": [
#                     AIMessage(content="Flight reference data is unavailable.")
#                 ],
#                 "llm_calls": state.get("llm_calls", 0),
#             }

#         prompt = FLIGHT_AGENT_PROMPT.format(
#             query=query,
#             airport_data=str(airports)[:3000],
#             airline_data=str(airlines)[:3000]
#         )

#         response = llm.invoke([
#             SystemMessage(
#                 content="You are an expert travel flight planner."
#             ),
#             HumanMessage(content=prompt)
#         ])

#         flight_data = response.content

#     except Exception as e:

#         flight_data = f"Flight information unavailable: {str(e)}"

#     return {
#         "flight_results": flight_data,
#         "messages": [
#             AIMessage(
#                 content="Flight recommendations generated"
#             )
#         ],
#         "llm_calls": state.get("llm_calls", 0) + 1
#     }
     
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

def hotel_agent(state:TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = asyncio.run(tavily_mcp_search(query))

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(content="Hotel information fetched.")
        ],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def weather_agent(state: TravelState):
    city=destination_extractor(state["user_query"])
    weather_data=asyncio.run(weather_mcp_search(city))
    forecast_data=asyncio.run(forecast_mcp_search(city))

    return{
        "weather_results":f"""
            Current_weather:
            {weather_data}
            Forecast:
            {forecast_data}
            """,
            "messages":[
                AIMessage(
                    content="Weather Information fetched"
                )
            ]
    }


def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Flight Results:
{state['flight_results']}

Hotel Results:
{state['hotel_results']}

Weather Results:
{state['weather_results']}

Make the itinerary practical, budget-aware, and easy to follow.
"""

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
    ])

    return {
        "itinerary": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def final_agent(state: TravelState):
    final_prompt = f"""
Generate the final travel response for the user.

User Request:
{state['user_query']}

Flights:
{state['flight_results']}

Hotels:
{state['hotel_results']}

Weather:
{state['weather_results']}

Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Informaton
4. Day-by-Day Itinerary
5. Estimated Budget
6. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight API may not provide ticket prices if pricing is unavailable.
- Include weather_based travel advice
- Keep the response useful for real travel planning.
"""

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
    ])

    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent",weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "final_agent")
graph.add_edge("final_agent", END)


DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

def run_travel_agent(user_input: str, thread_id: str | None = None):
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [
                HumanMessage(content=user_input)
            ],
            "user_query": user_input,
            "flight_results": "",
            "hotel_results": "",
            "weather_results":"",
            "itinerary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results":result.get("weather_results"),
        "itinerary": result.get("itinerary", ""),
        "llm_calls": result.get("llm_calls", 0),
    }