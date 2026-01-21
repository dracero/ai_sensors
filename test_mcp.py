
import asyncio
import nest_asyncio
from mcp_agent.agent import get_weather_forecast_tool

# Apply nest_asyncio to allow nested event loops if necessary
nest_asyncio.apply()

print("Testing MCP Agent Weather Connection...")
try:
    result = get_weather_forecast_tool()
    print("\nResult:")
    print(result)
except Exception as e:
    print(f"\nError: {e}")
