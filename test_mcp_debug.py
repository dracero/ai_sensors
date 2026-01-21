
import asyncio
from mcp_agent.agent import get_real_weather_forecast

async def main():
    print("Calling get_real_weather_forecast...")
    result = await get_real_weather_forecast()
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
