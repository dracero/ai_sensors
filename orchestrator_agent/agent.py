"""
Orchestrator Agent - ADK Agent that makes irrigation decisions.

This agent evaluates sensor data and decides whether to irrigate based on:
- Humidity < 50% AND Temperature > 25°C → Start irrigation ("regar")
- Otherwise → No irrigation needed ("no_regar")
"""

import os
from dotenv import load_dotenv
from google.adk.agents import Agent

# Import tools from other agents
from mcp_sensor_server.server import get_sensor_data, get_temperature, get_humidity
from mcp_agent.agent import get_weather_forecast_tool as get_weather_forecast
from mqtt_agent.agent import send_irrigation_command, get_mqtt_status

# Load environment variables
load_dotenv()

# Irrigation thresholds
HUMIDITY_THRESHOLD = 50.0  # Below this percentage, irrigation may be needed
TEMPERATURE_THRESHOLD = 25.0  # Above this temperature, irrigation may be needed


def evaluate_irrigation_need() -> dict:
    """
    Evaluate whether irrigation is needed based on current sensor readings.
    
    The irrigation logic is:
    - If humidity < 50% AND temperature > 25 C
    - AND No rain in forecast for San Pedro
    - THEN -> Irrigation needed
    - Otherwise -> No irrigation needed
    
    Returns:
        dict: Evaluation result containing:
            - needs_irrigation (bool): True if irrigation is recommended
            - temperature (float): Current temperature reading
            - humidity (float): Current humidity reading
            - forecast (dict): Weather forecast data
            - reason (str): Human-readable explanation of the decision
            - thresholds (dict): The threshold values used for evaluation
    """
    sensor_data = get_sensor_data()
    forecast = get_weather_forecast("San Pedro")
    
    temp = sensor_data.get("temperature")
    humidity = sensor_data.get("humidity")
    
    # Check if we have valid data
    if temp is None or humidity is None:
        return {
            "needs_irrigation": None,
            "temperature": temp,
            "humidity": humidity,
            "forecast": forecast,
            "reason": "[ERROR] No sensor data available. Cannot evaluate irrigation need.",
            "thresholds": {
                "humidity_max": HUMIDITY_THRESHOLD,
                "temperature_min": TEMPERATURE_THRESHOLD
            },
            "data_available": False
        }
    
    # Check if data is too old - allow up to 120s for slow networks
    if sensor_data.get("is_stale", True):
        age = sensor_data.get("age_seconds", 999)
        if age > 120: # Increased timeout for slow network
            return {
                "needs_irrigation": None,
                "temperature": temp,
                "humidity": humidity,
                "forecast": forecast,
                "reason": f"[WARN] Sensor data is too stale ({age}s old). Network may be slow.",
                "thresholds": {
                    "humidity_max": HUMIDITY_THRESHOLD,
                    "temperature_min": TEMPERATURE_THRESHOLD
                },
                "data_available": True,
                "data_stale": True
            }
    
    # Evaluate conditions: humidity < 50% AND temperature > 25 C AND No Rain
    env_conditions_met = humidity < HUMIDITY_THRESHOLD and temp > TEMPERATURE_THRESHOLD
    
    rain_expected = False
    if forecast and isinstance(forecast, dict):
        rain_expected = forecast.get("is_raining_soon", False)
    else:
        print(f"[WARN] Forecast unavailable or invalid: {forecast}")
    
    needs_irrigation = env_conditions_met and not rain_expected
    
    if needs_irrigation:
        reason = (
            f"[IRRIGATE] IRRIGATION NEEDED:\n"
            f"  - Humidity ({humidity}%) is below {HUMIDITY_THRESHOLD}%\n"
            f"  - Temperature ({temp}C) is above {TEMPERATURE_THRESHOLD}C\n"
            f"  - Forecast for {forecast['location']}: {forecast['condition']} (No rain)\n"
            f"  -> Plants need water!"
        )
    else:
        reasons = []
        if not env_conditions_met:
            if humidity >= HUMIDITY_THRESHOLD:
                reasons.append(f"Humidity ({humidity}%) is adequate (>= {HUMIDITY_THRESHOLD}%)")
            if temp <= TEMPERATURE_THRESHOLD:
                reasons.append(f"Temperature ({temp}C) is not high (<= {TEMPERATURE_THRESHOLD}C)")
        
        if rain_expected:
            reasons.append(f"RAIN EXPECTED in {forecast['location']} ({forecast['condition']}) - Saving water")
        
        reason = (
            f"[SKIP] NO IRRIGATION NEEDED:\n"
            f"  - {chr(10).join('- ' + r for r in reasons)}\n"
            f"  -> Conditions are acceptable or rain is coming."
        )
    
    return {
        "needs_irrigation": needs_irrigation,
        "temperature": temp,
        "humidity": humidity,
        "forecast": forecast,
        "reason": reason,
        "thresholds": {
            "humidity_max": HUMIDITY_THRESHOLD,
            "temperature_min": TEMPERATURE_THRESHOLD
        },
        "data_available": True,
        "data_stale": False
    }


def execute_irrigation_decision() -> dict:
    """
    Evaluate sensor data and automatically send the appropriate irrigation command.
    
    This function:
    1. Reads current sensor data
    2. Evaluates if irrigation is needed
    3. Sends the appropriate command to ESP32
    
    Returns:
        dict: Complete decision and execution result.
    """
    # First, evaluate the need
    evaluation = evaluate_irrigation_need()
    
    if evaluation.get("needs_irrigation") is None:
        return {
            "evaluation": evaluation,
            "command_sent": False,
            "command_result": None,
            "message": "Cannot make irrigation decision - no valid sensor data"
        }
    
    # Determine command based on evaluation
    command = "regar" if evaluation["needs_irrigation"] else "no_regar"
    
    # Send the command
    command_result = send_irrigation_command(command)
    
    return {
        "evaluation": evaluation,
        "command_sent": True,
        "command": command,
        "command_result": command_result,
        "message": f"Decision made and command '{command}' sent to ESP32"
    }


def set_irrigation_thresholds(humidity_threshold: float = None, temperature_threshold: float = None) -> dict:
    """
    Update the irrigation threshold values.
    
    Args:
        humidity_threshold: New humidity threshold (irrigation needed if below this %)
        temperature_threshold: New temperature threshold (irrigation needed if above this °C)
    
    Returns:
        dict: Updated threshold values.
    """
    global HUMIDITY_THRESHOLD, TEMPERATURE_THRESHOLD
    
    if humidity_threshold is not None:
        HUMIDITY_THRESHOLD = humidity_threshold
    if temperature_threshold is not None:
        TEMPERATURE_THRESHOLD = temperature_threshold
    
    return {
        "humidity_threshold": HUMIDITY_THRESHOLD,
        "temperature_threshold": TEMPERATURE_THRESHOLD,
        "message": f"Thresholds updated: Irrigate when humidity < {HUMIDITY_THRESHOLD}% AND temp > {TEMPERATURE_THRESHOLD}°C"
    }


def get_system_status() -> dict:
    """
    Get comprehensive status of the entire irrigation system.
    
    Returns:
        dict: Complete system status including MQTT, sensors, and thresholds.
    """
    mqtt_status = get_mqtt_status()
    sensor_data = get_sensor_data()
    evaluation = evaluate_irrigation_need()
    
    return {
        "mqtt": mqtt_status,
        "sensors": sensor_data,
        "irrigation_evaluation": evaluation,
        "thresholds": {
            "humidity_max": HUMIDITY_THRESHOLD,
            "temperature_min": TEMPERATURE_THRESHOLD
        }
    }


# Define Orchestrator Agent tools
ORCHESTRATOR_TOOLS = [
    evaluate_irrigation_need,
    execute_irrigation_decision,
    set_irrigation_thresholds,
    get_system_status,
    get_sensor_data,
    send_irrigation_command,
    get_weather_forecast,
]


# Create the Orchestrator Agent
orchestrator_agent = Agent(
    name="orchestrator_agent",
    model="gemini-2.0-flash",
    description="Intelligent agent that orchestrates irrigation decisions based on sensor data and weather forecast.",
    instruction="""You are the Orchestrator Agent for an IoT irrigation system (San Pedro location).

Your primary responsibility is to make intelligent irrigation decisions.

DECISION LOGIC:
- Check sensor data (Temp/Humidity)
- Check weather forecast for "San Pedro"
- IF (Humidity < 50% AND Temp > 25 C) AND (No Rain Forecast):
  -> Send "regar" (Start Irrigation)
- ELSE:
  -> Send "no_regar" (Stop/Skip)

SLOW NETWORK:
- The network is slow (Public MQTT).
- Be patient with data fetching.
- Prioritize reliability (QoS 2 is enabled).

AVAILABLE ACTIONS:
- evaluate_irrigation_need(): Checks sensors + forecast
- execute_irrigation_decision(): Auto-decide & Execute
- get_weather_forecast(location="San Pedro"): Check forecast
""",
    tools=ORCHESTRATOR_TOOLS
)


if __name__ == "__main__":
    # Test the orchestrator
    print("Testing Orchestrator Agent tools...")
    
    # Initialize MQTT connection
    from mqtt_agent.agent import initialize
    initialize()
    
    import time
    time.sleep(3)
    
    print("\n--- Testing get_system_status ---")
    status = get_system_status()
    print(f"System status: {status}")
    
    print("\n--- Testing evaluate_irrigation_need ---")
    evaluation = evaluate_irrigation_need()
    print(f"Evaluation: {evaluation}")
