import os
import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()   
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()  
from typing import TypedDict, List,Annotated,Any
import operator 
import uuid
import asyncio  
import psycopg
import json
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
from langgraph.types import Command, interrupt
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
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")


groq_llm=ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    max_tokens=800,
    groq_api_key=GROQ_API_KEY
)

gemini_llm=None
if GEMINI_API_KEY:
    gemini_llm=ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        google_api_key=GEMINI_API_KEY,
        max_output_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "8192")),
    )
else:
    print("GEMINI_API_KEY not set - Gemini fallback is disabled.")


def _to_text_message(response):
    """Gemini 3.x can return content as a list of blocks; flatten it to a plain string."""
    content=response.content
    if isinstance(content, list):
        content="".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return AIMessage(content=str(content))


class GroqWithGeminiFallback:
    """Tries Groq first; switches to Gemini if Groq errors or its output is truncated."""

    def __init__(self, primary, fallback=None):
        self.primary=primary
        self.fallback=fallback

    def invoke(self, messages, **kwargs):
        try:
            response=self.primary.invoke(messages, **kwargs)
        except Exception as exc:
            if self.fallback is None:
                raise
            print(f"\nGroq failed ({exc}) - switching to Gemini 3.8 Flash.\n")
            return _to_text_message(self.fallback.invoke(messages, **kwargs))

        finish_reason=(response.response_metadata or {}).get("finish_reason")
        if finish_reason == "length" and self.fallback is not None:
            print("\nGroq output hit the token limit - regenerating with Gemini 3.8 Flash.\n")
            try:
                return _to_text_message(self.fallback.invoke(messages, **kwargs))
            except Exception as exc:
                print(f"\nGemini fallback failed ({exc}) - using the truncated Groq output.\n")

        return response


llm=GroqWithGeminiFallback(groq_llm, gemini_llm)

class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str

    guardrail_allowed:bool
    guardrail_reason:str
    selected_agents:list[str]
    trip_constraints:dict[str,Any]
    supervisor_reasoning:str

    flight_results:str
    hotel_results:str
    weather_results:str
    itinerary:str

    budget_results:str
    approval_request:str
    approved:bool
    human_feedback:str
    final_response:str

    llm_calls:int

KNOWN_AGENTS={
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
}   

AGENT_ORDER=[
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent"
]

def _llm_text(system_prompt:str,user_prompt:str)->str:
    response=llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
    )
    return str(response.content)

def json_from_llm(text:str)->dict[str,any]:
    start=text.find("{")
    end=text.rfind("}")

    if start==-1 or end==-1 or end<start:
        raise ValueError("The modeldid not returnajson object")
    return json.loads(text[start:end+1])

def  empty_constriants()->dict[str,Any]:
    return{
        "destinaton":"",
        "origin":"",
        "duration":"",
        "budget":"",
        "travel_stylr":"",
        "special_preferences":[]
    }


def supervisor_agent(state:TravelState):
    query=state["user_query"]
    llm_calls=state.get("llm_calls",0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""
    try:
        guardrail_raw=_llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = json_from_llm(guardrail_raw)
        allowed = guardrail_result.get("allowed", True)
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()

    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints":empty_constriants(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""
    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = empty_constriants()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints =empty_constriants()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }

def guardrail_blocked_agent(state:TravelState):
    reason=state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }

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

def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
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

Budget Results:
{state.get('budget_results', '')}


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

def human_approval_agent(state:TravelState):
    review=interrupt(
        {
          "question":"Do you appeove this itinerary?",
          "draft_itinerary":state.get("itinerary",""),
          "approval_request":state.get("approval_request",""),
          "selected_agent":state.get("selected_agents",""),
          "supervisor_response":state.get("supervisor_reasoning",""),
          "expected_response":{
              "approved":True,
              "feedback":"Optimal revision feedback",
          }, 
        }
    )
    approved=bool(review.get("approved",False))
    human_feedback=str(review.get("feedback","")).strip()

    return {
        "approved":approved,
        "human_feedback":human_feedback,
        "messages":[AIMessage(content="Human approval step completed.")]
    }
def final_agent(state: TravelState):
    if state.get("approved",False):
        review_instruction=(
            "The user approved the draft .Preserve its decision while polishing it."
        )

    else:
        review_instruction=f"""
The user requested a revison.Apply this feedback carefully:
{state.get('human_feedback','') or 'Improve the draft before finalizing it.'}"""    
    final_prompt = f"""
Generate the final travel response for the user.

Human_Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints',{})}
Flights:
{state.get('flight_results','')}

Hotels:
{state.get('hotel_results','')}

Weather:
{state.get('weather_results','')}

Budget Analysis:
{state.get('budget_results','')}

Draft Itinerary:
{state['itinerary']}

Format the final answer beautifully using these sections:

1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Informaton
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

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
        "final_response": response.content,
        "messages":[response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

DATABASE_URL = get_database_url()

_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row
)

checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)

def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}

def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints":empty_constriants(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)

def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)