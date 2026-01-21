
import time
import json
import paho.mqtt.client as mqtt

# Configuration
BROKER = "mqtt-dashboard.com"
PORT = 1883
TOPIC = "fadena/test"

def publish_data():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    print(f"Connecting to {BROKER}:{PORT}...")
    try:
        client.connect(BROKER, PORT, 60)
        
        while True:
            # Simulate data: Temp 28°C (high), Humidity 45% (low) -> Should trigger irrigation
            payload = {
                "temp": 28.5,
                "humidity": 45.0,
                "client_id": "simulated_esp32",
                "irrigation_active": False
            }
            msg = json.dumps(payload)
            print(f"Publishing: {msg}")
            client.publish(TOPIC, msg, qos=2)
            time.sleep(5)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()

if __name__ == "__main__":
    publish_data()
