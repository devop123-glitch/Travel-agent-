import os
import asyncio
import sys
import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_groq import ChatGroq

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()  

load_dotenv()
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY=os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY=os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

llm=ChatGroq(
    model=os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b"),
    max_tokens=900,
    groq_api_key=GROQ_API_KEY
)

client=MultiServerMCPClient(
    {
        "tavily":{
            "transport":"streamable_http",
            "url":f"https://mcp.tavily.com/mcp/?tavilyApiKey={TAVILY_API_KEY}"


        },

        "aviationstack": {
            "transport": "stdio",
            "command": "uvx",
            "args": [
                "aviationstack-mcp"
            ],
            "env": {
               "AVIATION_STACK_API_KEY": AVIATION_STACK_API_KEY
            }
            },
        "weather":{
            "transport":"stdio",
            "command":sys.executable,
            "args":[
                r"C:\Users\Arghadip Dutta\Travel-agent-\custom_mcp_server.py"
            ],
            "env":{
                "OPENWEATHER_API_KEY":OPENWEATHER_API_KEY
            }
        }    
    }
)

async def get_all_tools():
    tools = await client.get_tools()
    for tool in tools:
        print(f"Tool Name: {tool.name}, Tool Description: {tool.description}")

search_tools=None
async def initialize_mcp(): 
    global search_tools
    if search_tools is not None:
        return
    tools=await client.get_tools()
    search_tools=next(
        (tool for tool in tools if tool.name=="tavily_search"),
        None,
    )
    if search_tools is None:
        available_tools = ", ".join(tool.name for tool in tools)
        raise RuntimeError(
            "Could not find the Tavily search tool. "
            f"Available tools: {available_tools}"
        )

async def tavily_mcp_search(query:str):
    await initialize_mcp()
    result=await search_tools.ainvoke(
        {
            "query":query
        }
    )    
    return result

async def aviation_mcp_call(
        tool_name:str,
        tool_args:dict=None
):
    tools=await client.get_tools()
    tool=next(
        t for t in tools 
        if t.name==tool_name
    )
    result=await tool.ainvoke(
        tool_args or {}
    )
    return result


weather_tool=None
forecast_tool=None
async def initialize_weather_tool():
    global weather_tool,forecast_tool

    if weather_tool is not None:
        return
    tools=await client.get_tools()
    weather_tool=next(
        t for t in tools
        if t.name=="get_current_weather"
    )
    forecast_tool=next(
            t for t in tools
            if t.name=="get_forecast"
    )

async def weather_mcp_search(city:str):
    await initialize_weather_tool()
    return await weather_tool.ainvoke(
        {
            "city":city
        }
    )    

async def forecast_mcp_search(city:str):
    await initialize_weather_tool()
    return await forecast_tool.ainvoke(
        {
            "city":city
        }
    )    

def destination_extractor(query:str):
    prompt=f"""
    Extract only the destination city or country.
    
    Query:
    {query}
    Return only destination name.
    """
    response =llm.invoke(prompt)
    return response.content.strip()
