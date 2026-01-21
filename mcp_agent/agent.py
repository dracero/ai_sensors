"""
MCP Agent - Connects to external MCP servers (specifically the Weather MCP).
"""

import sys
import os
import asyncio
from google.adk.agents import Agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Path to the external weather MCP server script
WEATHER_SERVER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external_mcp_server", "MCP-main", "weather.py"
)

async def get_real_weather_forecast(latitude: float = 33.7360, longitude: float = -118.2938) -> dict:
    """
    Get real weather forecast using the external MCP Weather Server.
    Default coordinates are for San Pedro, CA (since NWS is US-only).
    
    Args:
        latitude: Latitude (default: San Pedro, CA)
        longitude: Longitude (default: San Pedro, CA)
        
    Returns:
        dict: The forecast result or error information.
    """
    print(f"[MCP Agent] Connecting to Weather Server at {WEATHER_SERVER_PATH}...")
    
    # Define server parameters
    server_params = StdioServerParameters(
        command=sys.executable,  # Use current python executable
        args=[WEATHER_SERVER_PATH],
        env=os.environ.copy()
    )
    
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Call the 'get_forecast' tool provided by the server
                print(f"[MCP Agent] Requesting forecast for {latitude}, {longitude}...")
                result = await session.call_tool(
                    "get_forecast", 
                    arguments={"latitude": latitude, "longitude": longitude}
                )
                
                if result.isError:
                    return {"error": True, "message": str(result.content)}
                
                # content is a list of TextContent or ImageContent. taking the first text.
                text_content = ""
                if hasattr(result, 'content') and result.content:
                    for item in result.content:
                        if hasattr(item, 'text'):
                            text_content += item.text
                
                return {
                    "location": "San Pedro (US)",
                    "forecast_text": text_content,
                    "raw_result": str(result)
                }
                
    except Exception as e:
        print(f"[MCP Agent] Error calling Weather Server: {e}")
        return {"error": True, "message": str(e)}

# Wrapper for the synchronous tool call if needed (ADK might need sync tools depending on runner, 
# but ADK supports async tools too. We'll provide a sync wrapper just in case or usage via asyncio.run)
def get_weather_forecast_tool(location: str = "San Pedro") -> dict:
    """
    Tool exposed to other agents to get weather.
    
    Args:
        location: Name of the location (ignored, uses San Pedro coords for demo)
    """
    import nest_asyncio
    
    result = None
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            nest_asyncio.apply()
            result = loop.run_until_complete(get_real_weather_forecast())
        else:
            result = loop.run_until_complete(get_real_weather_forecast())

    except Exception as e:
        print(f"[MCP Agent] Error in tool wrapper: {e}")
        return {
            "error": True, 
            "message": str(e),
            "is_raining_soon": False, # Fail-safe
            "location": location,
            "condition": "Error"
        }
        
    if not result:
        return {
            "error": True, 
            "message": "No result returned",
            "is_raining_soon": False,
            "location": location,
            "condition": "Unknown"
        }
        
    # Map the text result to the schema expected by Orchestrator
    forecast_text = result.get("forecast_text", "").lower()
    is_raining = "rain" in forecast_text or "shower" in forecast_text
    
    # Enrich the result with fields expected by Orchestrator
    result["is_raining_soon"] = is_raining
    result["condition"] = "Rainy" if is_raining else "Clear"
    result["rain_probability"] = 80.0 if is_raining else 0.0
    
    return result

# Define tools
MCP_TOOLS = [get_weather_forecast_tool]

# Create the agent
mcp_agent = Agent(
    name="mcp_agent",
    model="gemini-2.0-flash",
    description="Agent that connects to external MCP servers to fetch specialized data (e.g. Weather).",
    instruction="You are the MCP Agent. Your job is to fetch real weather data from the external Weather MCP Server.",
    tools=MCP_TOOLS
)
