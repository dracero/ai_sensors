"""
IoT Irrigation Control System - Main Entry Point

This is the main entry point for the multi-agent ADK irrigation control system.
It initializes all agents and provides an interactive interface for testing.
"""

import os
import sys
import time
import asyncio
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Verify API key is set
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found in .env file")
    print("Please add your Gemini API key to the .env file")
    sys.exit(1)

# Import agents and tools
from mcp_sensor_server.server import start_mqtt_listener, stop_mqtt_listener
from mqtt_agent.agent import mqtt_agent, initialize as init_mqtt
from mcp_agent.agent import mcp_agent
from orchestrator_agent.agent import (
    orchestrator_agent,
    evaluate_irrigation_need,
    execute_irrigation_decision,
    get_system_status
)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


def print_banner():
    """Print the application banner."""
    print("""
+==================================================================+
|         IoT Irrigation Control System - ADK Multi-Agent         |
+==================================================================+
|  Agents:                                                         |
|  - MCP Agent: Connects to external Weather MCP Server            |
|  - MQTT Agent: Sends commands to ESP32                           |
|  - Orchestrator Agent: Makes irrigation decisions                |
|                                                                  |
|  Logic: If humidity < 50% AND temp > 25C => IRRIGATE             |
+==================================================================+
    """)


def run_automatic_mode():
    """Run in automatic mode - continuously monitor and control irrigation."""
    print("\n[AUTO] Running in AUTOMATIC mode...")
    print("   Monitoring sensor data and making irrigation decisions")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            print("-" * 50)
            print(f"[CHECK] Time: {time.strftime('%H:%M:%S')}")
            
            # Evaluate and potentially execute irrigation
            result = execute_irrigation_decision()
            
            evaluation = result.get("evaluation", {})
            if evaluation.get("data_available"):
                temp = evaluation.get("temperature", "N/A")
                humidity = evaluation.get("humidity", "N/A")
                print(f"   [DATA] Temperature: {temp}C | Humidity: {humidity}%")
                print(f"   {evaluation.get('reason', 'No reason available')}")
                
                if result.get("command_sent"):
                    cmd_result = result.get("command_result", {})
                    print(f"   [OK] Command sent: {result.get('command')}")
                    print(f"   {cmd_result.get('message', '')}")
            else:
                print(f"   [WARN] {evaluation.get('reason', 'Waiting for sensor data...')}")
            
            # Wait before next check
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n[STOP] Automatic mode stopped")


def run_manual_mode():
    """Run in manual mode - interactive testing."""
    print("\n[MANUAL] Running in MANUAL mode...")
    print("   Available commands:")
    print("   [1] Check system status")
    print("   [2] Evaluate irrigation need")
    print("   [3] Execute irrigation decision")
    print("   [4] Send 'regar' command")
    print("   [5] Send 'no_regar' command")
    print("   [6] Switch to automatic mode")
    print("   [q] Quit\n")
    
    from mqtt_agent.agent import send_irrigation_command
    
    while True:
        try:
            choice = input("\n> Enter command (1-6, q): ").strip().lower()
            
            if choice == 'q':
                break
            elif choice == '1':
                status = get_system_status()
                print("\n[STATUS] System Status:")
                print(f"   MQTT Connected: {status['mqtt']['mqtt_connected']}")
                print(f"   Broker: {status['mqtt']['broker']}:{status['mqtt']['port']}")
                sensors = status['sensors']
                print(f"   Temperature: {sensors.get('temperature', 'N/A')}C")
                print(f"   Humidity: {sensors.get('humidity', 'N/A')}%")
                print(f"   Data age: {sensors.get('age_seconds', 'N/A')}s")
            elif choice == '2':
                evaluation = evaluate_irrigation_need()
                print(f"\n{evaluation.get('reason', 'Error evaluating')}")
            elif choice == '3':
                result = execute_irrigation_decision()
                print(f"\n[OK] Decision executed: {result.get('command', 'none')}")
                print(f"   {result.get('message', '')}")
            elif choice == '4':
                result = send_irrigation_command("regar")
                print(f"\n{result.get('message', 'Command sent')}")
            elif choice == '5':
                result = send_irrigation_command("no_regar")
                print(f"\n{result.get('message', 'Command sent')}")
            elif choice == '6':
                run_automatic_mode()
            else:
                print("[?] Unknown command")
                
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")



class AdkMessage:
    def __init__(self, role, text):
        self.role = role
        self.parts = [{"text": text}]

async def run_with_adk_chat():
    """Run interactive chat with the Orchestrator Agent using ADK."""
    print("\n[CHAT] Running in CHAT mode with Orchestrator Agent...")
    print("   You can chat naturally with the agent")
    print("   Type 'quit' to exit\n")
    
    # Create session service and runner
    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator_agent,
        app_name="irrigation_system",
        session_service=session_service
    )
    
    # Create a session
    session = await session_service.create_session(
        app_name="irrigation_system",
        user_id="user1"
    )
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not user_input:
                continue
            
            print("Agent: ", end="", flush=True)
            
            # Run the agent
            async for event in runner.run_async(
                session_id=session.id,
                user_id="user1",
                new_message=AdkMessage(role="user", text=user_input)
            ):
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text'):
                                print(part.text, end="", flush=True)
            print()  # New line after response
            
        except KeyboardInterrupt:
            break
    
    print("\nChat ended!")


def main():
    """Main entry point."""
    print_banner()
    
    # Initialize MQTT listener
    print("[INIT] Initializing MQTT connection...")
    init_mqtt()
    
    # Wait for potential initial data
    print("[WAIT] Waiting for sensor data (5 seconds)...")
    time.sleep(5)
    
    # Check initial status
    status = get_system_status()
    if status['mqtt']['mqtt_connected']:
        print("[OK] MQTT connected successfully!")
    else:
        print("[WARN] MQTT not connected yet - will keep trying in background")
    
    # Mode selection
    print("\n[MENU] Select mode:")
    print("   [1] Manual mode - Interactive testing")
    print("   [2] Automatic mode - Continuous monitoring")
    print("   [3] Chat mode - Talk with AI agent")
    print("   [q] Quit")
    
    try:
        choice = input("\n> Enter choice (1-3, q): ").strip().lower()
        
        if choice == '1':
            run_manual_mode()
        elif choice == '2':
            run_automatic_mode()
        elif choice == '3':
            asyncio.run(run_with_adk_chat())
        elif choice == 'q':
            print("\nGoodbye!")
        else:
            print("[?] Unknown choice, starting manual mode...")
            run_manual_mode()
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        stop_mqtt_listener()


if __name__ == "__main__":
    main()
