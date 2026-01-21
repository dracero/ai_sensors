"""
MQTT Agent - ADK Agent that communicates with ESP32 via MQTT.

This agent handles sending irrigation commands to the ESP32 device.
"""

import os
from dotenv import load_dotenv
import paho.mqtt.client as mqtt
from google.adk.agents import Agent

# Import sensor tools from MCP server
from mcp_sensor_server.server import (
    start_mqtt_listener,
    get_sensor_data,
    get_temperature,
    get_humidity,
    is_mqtt_connected,
    SENSOR_TOOLS
)

# Load environment variables
load_dotenv()

# Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-dashboard.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_COMMAND = os.getenv("MQTT_TOPIC_COMMAND", "esp32-sub")


def send_irrigation_command(command: str) -> dict:
    """
    Send an irrigation command to the ESP32 device via MQTT.
    
    Args:
        command: The command to send. Must be one of:
            - "regar": Start irrigation
            - "no_regar": Stop/skip irrigation
            - "reset": Reset the ESP32 device
    
    Returns:
        dict: Result with status and message.
            - success (bool): True if command was sent successfully
            - command (str): The command that was sent
            - topic (str): The MQTT topic used
            - message (str): Human-readable result message
    """
    valid_commands = ["regar", "no_regar", "reset"]
    
    if command not in valid_commands:
        return {
            "success": False,
            "command": command,
            "topic": MQTT_TOPIC_COMMAND,
            "message": f"Invalid command. Must be one of: {valid_commands}"
        }
    
    try:
        # Create a temporary client for publishing
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        
        # Publish the command with QoS 2 for guaranteed delivery
        result = client.publish(MQTT_TOPIC_COMMAND, command, qos=2)
        result.wait_for_publish(timeout=10)
        
        client.disconnect()
        
        command_descriptions = {
            "regar": "[IRRIGATE] Irrigation command sent - ESP32 will start watering",
            "no_regar": "[SKIP] No irrigation needed - ESP32 will keep pump off",
            "reset": "[RESET] Reset command sent - ESP32 will restart"
        }
        
        return {
            "success": True,
            "command": command,
            "topic": MQTT_TOPIC_COMMAND,
            "message": command_descriptions[command]
        }
        
    except Exception as e:
        return {
            "success": False,
            "command": command,
            "topic": MQTT_TOPIC_COMMAND,
            "message": f"Failed to send command: {str(e)}"
        }


def get_mqtt_status() -> dict:
    """
    Get the current status of the MQTT connection and last sensor reading.
    
    Returns:
        dict: Status information including connection state and last sensor data.
    """
    sensor_data = get_sensor_data()
    
    return {
        "mqtt_connected": is_mqtt_connected(),
        "broker": MQTT_BROKER,
        "port": MQTT_PORT,
        "sensor_topic": os.getenv("MQTT_TOPIC_SENSOR", "fadena/test"),
        "command_topic": MQTT_TOPIC_COMMAND,
        "last_sensor_data": sensor_data
    }


# Define MQTT Agent tools
MQTT_TOOLS = [
    send_irrigation_command,
    get_mqtt_status,
    *SENSOR_TOOLS  # Include sensor tools
]


# Create the MQTT Agent
mqtt_agent = Agent(
    name="mqtt_agent",
    model="gemini-2.0-flash",
    description="Agent that communicates with ESP32 via MQTT for sensor data and irrigation commands.",
    instruction="""You are an MQTT communication agent specialized in IoT irrigation systems.

Your capabilities:
1. Read temperature and humidity data from ESP32 sensors
2. Send irrigation commands to ESP32 devices
3. Monitor MQTT connection status

When asked about sensor data, use get_sensor_data() or individual get_temperature()/get_humidity() functions.
When asked to send commands, use send_irrigation_command() with 'regar', 'no_regar', or 'reset'.

Always report the status of operations clearly.""",
    tools=MQTT_TOOLS
)


def initialize():
    """Initialize the MQTT agent by starting the sensor listener."""
    print("[MQTT Agent] Initializing...")
    start_mqtt_listener()
    print("[MQTT Agent] Ready")


if __name__ == "__main__":
    # Test the agent tools
    print("Testing MQTT Agent tools...")
    initialize()
    
    import time
    time.sleep(3)
    
    print("\n--- Testing get_mqtt_status ---")
    status = get_mqtt_status()
    print(f"Status: {status}")
    
    print("\n--- Testing get_sensor_data ---")
    data = get_sensor_data()
    print(f"Sensor data: {data}")
    
    print("\n--- Testing send_irrigation_command ---")
    # Uncomment to test sending commands
    # result = send_irrigation_command("no_regar")
    # print(f"Command result: {result}")
