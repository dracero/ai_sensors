"""
ESP32 + DHT22 - Código actualizado para el sistema de riego IoT.

Este código se ejecuta en un ESP32 con MicroPython y:
1. Lee temperatura y humedad del sensor DHT22
2. Publica datos al broker MQTT
3. Recibe comandos de riego ('regar', 'no_regar', 'reset')
"""

import network
import time
from machine import Pin
import dht
import ujson
from umqtt.simple import MQTTClient

# ============================================================================
# CONFIGURATION
# ============================================================================
WIFI_SSID = "Wokwi-GUEST"  # Replace with your WiFi SSID
WIFI_PASSWORD = ""         # Replace with your WiFi password

MQTT_BROKER = "mqtt-dashboard.com"
MQTT_PORT = 1883
CLIENT_ID = "fadena_id"
TOPIC_SUB = "esp32-sub"       # Topic for receiving commands
TOPIC_PUB = "fadena/test"     # Topic for publishing sensor data
DHT_PIN = 15

# Optional: Pin for relay/pump control
RELAY_PIN = 2  # Built-in LED on most ESP32 boards (for testing)

# ============================================================================
# SETUP
# ============================================================================
led = Pin(RELAY_PIN, Pin.OUT)
led.value(0)  # Start with relay/pump OFF

irrigation_active = False


def connect_wifi():
    """Connect to WiFi network."""
    print("Connecting to Wi-Fi...", end="")
    sta_if = network.WLAN(network.STA_IF)
    sta_if.active(True)
    sta_if.connect(WIFI_SSID, WIFI_PASSWORD)
    
    timeout = 100  # 10 seconds timeout
    while not sta_if.isconnected() and timeout > 0:
        print(".", end="")
        time.sleep(0.1)
        timeout -= 1
    
    if sta_if.isconnected():
        print(" Connected!")
        print("Network config:", sta_if.ifconfig())
        return True
    else:
        print(" Failed to connect!")
        return False


def sub_cb(topic, msg):
    """
    Callback for received MQTT messages.
    
    Handles commands:
    - 'regar': Start irrigation (activate relay/pump)
    - 'no_regar': Stop irrigation (deactivate relay/pump)
    - 'reset': Reset the ESP32
    """
    global irrigation_active
    
    print(f"Received: topic={topic}, msg={msg}")
    
    if topic == b'esp32-sub':
        if msg == b'reset':
            print('🔄 Resetting ESP32...')
            import machine
            machine.reset()
            
        elif msg == b'regar':
            print('🌱 ====================================')
            print('🌱 IRRIGATION STARTED!')
            print('🌱 Activating pump/relay...')
            print('🌱 ====================================')
            led.value(1)  # Turn ON relay/LED
            irrigation_active = True
            
            # Publish confirmation
            confirm_msg = ujson.dumps({
                "status": "irrigation_started",
                "client_id": CLIENT_ID
            })
            # Note: Can't publish here directly, will be done in main loop
            
        elif msg == b'no_regar':
            print('⏸️ ====================================')
            print('⏸️ NO IRRIGATION NEEDED')
            print('⏸️ Pump/relay remains OFF')
            print('⏸️ ====================================')
            led.value(0)  # Turn OFF relay/LED
            irrigation_active = False
            
        else:
            print(f'❓ Unknown command: {msg}')


def connect_mqtt():
    """Connect to MQTT broker and subscribe to command topic."""
    print("Connecting to MQTT Broker...", end="")
    client = MQTTClient(CLIENT_ID, MQTT_BROKER, MQTT_PORT)
    client.set_callback(sub_cb)
    
    try:
        client.connect()
        client.subscribe(TOPIC_SUB, qos=2)  # QoS 2 for reliable delivery
        print(" Connected!")
        print(f"Subscribed to: {TOPIC_SUB} (QoS 2)")
        return client
    except Exception as e:
        print(f" Failed: {e}")
        return None


def main():
    """Main application loop."""
    global irrigation_active
    
    # Connect to WiFi
    if not connect_wifi():
        print("WiFi connection failed. Retrying in 5 seconds...")
        time.sleep(5)
        return main()  # Retry
    
    # Connect to MQTT
    client = connect_mqtt()
    if client is None:
        print("MQTT connection failed. Retrying in 5 seconds...")
        time.sleep(5)
        return main()  # Retry
    
    # Initialize DHT sensor
    sensor = dht.DHT22(Pin(DHT_PIN))
    
    print("\n" + "="*50)
    print("🌡️  ESP32 IoT Irrigation System Ready!")
    print("="*50)
    print(f"Publishing sensor data to: {TOPIC_PUB}")
    print(f"Listening for commands on: {TOPIC_SUB}")
    print("Commands: 'regar', 'no_regar', 'reset'")
    print("="*50 + "\n")
    
    last_publish = 0
    publish_interval = 5  # seconds
    
    while True:
        try:
            # Check for incoming MQTT messages
            client.check_msg()
            
            current_time = time.time()
            
            # Publish sensor data at intervals
            if current_time - last_publish >= publish_interval:
                # Read sensor
                sensor.measure()
                temp = sensor.temperature()
                hum = sensor.humidity()
                
                # Create message with sensor data and irrigation status
                msg = ujson.dumps({
                    "temp": temp,
                    "humidity": hum,
                    "client_id": CLIENT_ID,
                    "irrigation_active": irrigation_active
                })
                
                # Visual indicators
                irrigation_status = "🌱 ACTIVE" if irrigation_active else "⏸️ OFF"
                print(f"📊 Temp: {temp}°C | Humidity: {hum}% | Irrigation: {irrigation_status}")
                print(f"   Publishing to {TOPIC_PUB} (QoS 2)")
                
                client.publish(TOPIC_PUB, msg, qos=2)
                last_publish = current_time
            
            time.sleep(0.1)  # Small delay to prevent busy loop
            
        except OSError as e:
            print(f"❌ Error: {e}")
            print("Attempting to reconnect...")
            
            try:
                client = connect_mqtt()
                if client is None:
                    time.sleep(5)
            except Exception as reconnect_error:
                print(f"Reconnection failed: {reconnect_error}")
                time.sleep(5)


if __name__ == "__main__":
    main()
