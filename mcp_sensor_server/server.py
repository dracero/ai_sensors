"""
MCP Sensor Server - Exposes temperature and humidity data from ESP32 via MQTT.

This server connects to the MQTT broker to receive sensor data and exposes
it as MCP tools that can be used by ADK agents.
"""

import os
import json
import threading
import time
from typing import Optional
from dotenv import load_dotenv
import paho.mqtt.client as mqtt

# Load environment variables
load_dotenv()

# Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt-dashboard.com")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_TOPIC_SENSOR = os.getenv("MQTT_TOPIC_SENSOR", "fadena/test")

# Global storage for sensor data
_sensor_data = {
    "temperature": None,
    "humidity": None,
    "client_id": None,
    "last_update": None
}
_lock = threading.Lock()
_mqtt_client: Optional[mqtt.Client] = None
_is_connected = False


def _on_connect(client, userdata, flags, reason_code, properties):
    """Callback when connected to MQTT broker."""
    global _is_connected
    print(f"[MCP Server] Connected to MQTT broker with result code: {reason_code}")
    _is_connected = True
    client.subscribe(MQTT_TOPIC_SENSOR, qos=2)
    print(f"[MCP Server] Subscribed to topic: {MQTT_TOPIC_SENSOR} (QoS 2)")


def _on_disconnect(client, userdata, flags, reason_code, properties):
    """Callback when disconnected from MQTT broker."""
    global _is_connected
    print(f"[MCP Server] Disconnected from MQTT broker: {reason_code}")
    _is_connected = False


def _on_message(client, userdata, msg):
    """Callback when message received from MQTT."""
    global _sensor_data
    try:
        payload = json.loads(msg.payload.decode())
        with _lock:
            _sensor_data["temperature"] = payload.get("temp")
            _sensor_data["humidity"] = payload.get("humidity")
            _sensor_data["client_id"] = payload.get("client_id")
            _sensor_data["last_update"] = time.time()
        print(f"[MCP Server] Received sensor data: temp={_sensor_data['temperature']}°C, humidity={_sensor_data['humidity']}%")
    except json.JSONDecodeError as e:
        print(f"[MCP Server] Error decoding message: {e}")
    except Exception as e:
        print(f"[MCP Server] Error processing message: {e}")


def start_mqtt_listener():
    """Start the MQTT listener in a background thread."""
    global _mqtt_client
    
    if _mqtt_client is not None:
        print("[MCP Server] MQTT listener already running")
        return
    
    _mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    _mqtt_client.on_connect = _on_connect
    _mqtt_client.on_disconnect = _on_disconnect
    _mqtt_client.on_message = _on_message
    
    def mqtt_loop():
        try:
            print(f"[MCP Server] Connecting to MQTT broker: {MQTT_BROKER}:{MQTT_PORT} (Timeout: 60s)")
            _mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
            _mqtt_client.loop_forever(retry_first_connection=True)
        except Exception as e:
            print(f"[MCP Server] MQTT connection error: {e}")
            # Retry logic could be added here
            time.sleep(5)
    
    thread = threading.Thread(target=mqtt_loop, daemon=True)
    thread.start()
    print("[MCP Server] MQTT listener started in background")
    
    # Wait a bit for connection
    time.sleep(2)


def stop_mqtt_listener():
    """Stop the MQTT listener."""
    global _mqtt_client, _is_connected
    if _mqtt_client:
        _mqtt_client.disconnect()
        _mqtt_client = None
        _is_connected = False
        print("[MCP Server] MQTT listener stopped")


# ============================================================================
# MCP Tool Functions - These are exposed as tools for ADK agents
# ============================================================================

def get_sensor_data() -> dict:
    """
    Get the current temperature and humidity readings from the ESP32 sensor.
    
    Returns:
        dict: A dictionary containing:
            - temperature (float): Current temperature in Celsius
            - humidity (float): Current humidity percentage
            - client_id (str): ID of the ESP32 client
            - last_update (float): Timestamp of last update
            - is_stale (bool): True if data is older than 30 seconds
    """
    with _lock:
        data = _sensor_data.copy()
    
    # Check if data is stale (older than 30 seconds)
    if data["last_update"]:
        age = time.time() - data["last_update"]
        data["is_stale"] = age > 30
        data["age_seconds"] = round(age, 1)
    else:
        data["is_stale"] = True
        data["age_seconds"] = None
    
    return data


def get_temperature() -> float | None:
    """
    Get the current temperature reading from the ESP32 sensor.
    
    Returns:
        float | None: Current temperature in Celsius, or None if no data available.
    """
    with _lock:
        return _sensor_data["temperature"]


def get_humidity() -> float | None:
    """
    Get the current humidity reading from the ESP32 sensor.
    
    Returns:
        float | None: Current humidity percentage, or None if no data available.
    """
    with _lock:
        return _sensor_data["humidity"]


def get_weather_forecast(location: str = "San Pedro") -> dict:
    """
    Get the weather forecast for a specific location.
    
    Args:
        location (str): The city name to get forecast for. Defaults to "San Pedro".
        
    Returns:
        dict: Weather information containing:
            - location (str): The location name
            - condition (str): Text description (e.g., "Sunny", "Rainy")
            - rain_probability (float): Probability of rain (0-100)
            - is_raining_soon (bool): True if rain is expected
    """
    # In a real implementation, this would call a Weather API.
    # For this ADK demo, we will simulate the forecast for San Pedro.
    # We'll simulate a "No Rain" condition to allow irrigation if temp/humidity match.
    
    print(f"[MCP Server] Fetching forecast for {location}...")
    time.sleep(1) # Simulate network latency
    
    # Mock data - ideally this would act as a client for a real forecast service
    return {
        "location": location,
        "condition": "Despejado", # Clear
        "rain_probability": 10.0,
        "is_raining_soon": False,
        "temperature_forecast": 26.0
    }



def is_mqtt_connected() -> bool:
    """
    Check if the MCP server is connected to the MQTT broker.
    
    Returns:
        bool: True if connected, False otherwise.
    """
    return _is_connected


# Export tools list for ADK
SENSOR_TOOLS = [
    get_sensor_data,
    get_temperature,
    get_humidity,
    is_mqtt_connected,
    get_weather_forecast,
]


if __name__ == "__main__":
    # Test the server
    print("Starting MCP Sensor Server test...")
    start_mqtt_listener()
    
    try:
        while True:
            data = get_sensor_data()
            print(f"Current data: {data}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopping...")
        stop_mqtt_listener()
