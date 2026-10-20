from tools.tavily_tool import tavily_search
from tools.flight_tool import search_flights
from backend import run_travel_agent
user_input=input("Enter your travel query: ")
response = run_travel_agent(
    user_input=user_input,
    thread_id="test_user"
    )
print(response["answer"])