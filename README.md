# 🌱 Sistema IoT de Control de Riego Inteligente

Sistema multi-agente basado en Google ADK (Agent Development Kit) que controla automáticamente el riego de plantas utilizando un ESP32 con sensor DHT22, comunicación MQTT y agentes de IA para toma de decisiones.

## 📋 Tabla de Contenidos

- [Descripción General](#-descripción-general)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Componentes](#-componentes)
- [Requisitos](#-requisitos)
- [Instalación](#-instalación)
- [Configuración](#-configuración)
- [Ejecución](#-ejecución)
- [Modos de Operación](#-modos-de-operación)
- [ESP32 - Hardware y Simulación](#-esp32---hardware-y-simulación)
- [Lógica de Decisión](#-lógica-de-decisión)
- [Diagramas](#-diagramas)

## 🎯 Descripción General

Este proyecto implementa un sistema de riego inteligente que:

1. **Lee datos ambientales** de un sensor DHT22 conectado a un ESP32
2. **Transmite datos** vía MQTT a un broker público
3. **Analiza condiciones** usando múltiples agentes de IA (Google Gemini)
4. **Consulta pronóstico del clima** mediante un servidor MCP externo
5. **Toma decisiones automáticas** sobre cuándo regar
6. **Envía comandos** al ESP32 para activar/desactivar el riego


## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SISTEMA DE RIEGO IoT                         │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         MQTT          ┌──────────────────────────┐
│   ESP32 + DHT22  │◄──────────────────────►│   Broker MQTT Público    │
│                  │   (mqtt-dashboard.com) │  (mqtt-dashboard.com)    │
│  • Sensor Temp   │                        └──────────────────────────┘
│  • Sensor Humid  │                                    ▲
│  • Relay/Bomba   │                                    │
└──────────────────┘                                    │ MQTT Subscribe
                                                        │
                    ┌───────────────────────────────────┴─────────┐
                    │                                             │
            ┌───────▼────────┐                           ┌────────▼────────┐
            │  MCP Sensor    │                           │   MQTT Agent    │
            │    Server      │                           │                 │
            │                │                           │ • Envía comandos│
            │ • Recibe datos │                           │ • "regar"       │
            │ • Almacena     │                           │ • "no_regar"    │
            └───────┬────────┘                           └────────▲────────┘
                    │                                             │
                    │ Provee datos                                │
                    │                                             │
            ┌───────▼─────────────────────────────────────────────┴────────┐
            │              ORCHESTRATOR AGENT (Gemini 2.0)                 │
            │                                                               │
            │  • Evalúa condiciones (Temp > 25°C, Humedad < 50%)           │
            │  • Consulta pronóstico del clima                             │
            │  • Toma decisión de riego                                    │
            │  • Ejecuta comando                                           │
            └───────────────────────────────┬───────────────────────────────┘
                                            │
                                            │ Consulta clima
                                            │
                                    ┌───────▼────────┐
                                    │   MCP Agent    │
                                    │                │
                                    │ • Conecta con  │
                                    │   Weather MCP  │
                                    │ • API NWS      │
                                    └────────────────┘
```


## 🧩 Componentes

### 1. **ESP32 con MicroPython** (`esp32/`)

Hardware IoT que ejecuta el código de sensores y control:

- **Archivo**: `esp32/main.py`
- **Sensor**: DHT22 (temperatura y humedad)
- **Comunicación**: MQTT (publica datos, recibe comandos)
- **Control**: Relay/LED para activar bomba de riego
- **Diagrama**: `esp32/diagram.json` (para simulación en Wokwi)

**Funciones principales**:
- Lee temperatura y humedad cada 5 segundos
- Publica datos JSON al topic `fadena/test`
- Escucha comandos en el topic `esp32-sub`
- Controla relay según comando recibido

### 2. **MCP Sensor Server** (`mcp_sensor_server/`)

Servidor que actúa como puente entre MQTT y los agentes ADK:

- **Archivo**: `mcp_sensor_server/server.py`
- **Función**: Suscribe al broker MQTT y almacena datos de sensores
- **Expone herramientas**:
  - `get_sensor_data()`: Obtiene temp y humedad actuales
  - `get_temperature()`: Solo temperatura
  - `get_humidity()`: Solo humedad
  - `is_mqtt_connected()`: Estado de conexión

### 3. **MQTT Agent** (`mqtt_agent/`)

Agente ADK especializado en comunicación MQTT:

- **Archivo**: `mqtt_agent/agent.py`
- **Modelo**: Gemini 2.0 Flash
- **Función**: Envía comandos de riego al ESP32
- **Herramientas**:
  - `send_irrigation_command(command)`: Envía "regar", "no_regar" o "reset"
  - `get_mqtt_status()`: Estado del sistema MQTT

### 4. **MCP Agent** (`mcp_agent/`)

Agente que conecta con servidores MCP externos:

- **Archivo**: `mcp_agent/agent.py`
- **Modelo**: Gemini 2.0 Flash
- **Función**: Consulta pronóstico del clima
- **Servidor externo**: `external_mcp_server/MCP-main/weather.py`
- **API**: National Weather Service (NWS) de EE.UU.
- **Herramienta**: `get_weather_forecast_tool(location)`


### 5. **Orchestrator Agent** (`orchestrator_agent/`)

Agente principal que coordina todo el sistema:

- **Archivo**: `orchestrator_agent/agent.py`
- **Modelo**: Gemini 2.0 Flash
- **Función**: Toma decisiones inteligentes de riego
- **Herramientas**:
  - `evaluate_irrigation_need()`: Evalúa si se necesita regar
  - `execute_irrigation_decision()`: Evalúa y ejecuta automáticamente
  - `set_irrigation_thresholds()`: Ajusta umbrales
  - `get_system_status()`: Estado completo del sistema

### 6. **Aplicación Principal** (`main.py`)

Punto de entrada del sistema con interfaz interactiva:

- Inicializa todos los agentes
- Ofrece 3 modos de operación
- Gestiona el ciclo de vida del sistema

## 📦 Requisitos

### Software

- **Python**: >= 3.13
- **MicroPython**: Para ESP32 (incluido en simulador Wokwi)
- **Dependencias Python**:
  ```
  google-adk >= 0.5.0
  paho-mqtt >= 2.0.0
  python-dotenv >= 1.0.0
  mcp
  httpx
  nest_asyncio
  matplotlib >= 3.10.8
  numpy >= 2.4.0
  ```

### Hardware (Opcional - para implementación física)

- ESP32 DevKit C V4
- Sensor DHT22
- Relay 5V (para controlar bomba)
- Bomba de agua 5V/12V
- Cables jumper
- Fuente de alimentación

### Servicios Externos

- **Broker MQTT**: `mqtt-dashboard.com` (público, sin autenticación)
- **API Gemini**: Clave de API de Google AI Studio
- **Weather API**: National Weather Service (gratuito, sin clave)


## 🚀 Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd pyprograms
```

### 2. Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate  # En Linux/Mac
# o
.venv\Scripts\activate  # En Windows
```

### 3. Instalar dependencias

```bash
pip install -e .
```

O manualmente:

```bash
pip install google-adk paho-mqtt python-dotenv mcp httpx nest_asyncio matplotlib numpy
```

### 4. Configurar variables de entorno

Copia el archivo de ejemplo y edítalo:

```bash
cp .env_example .env
```

Edita `.env` y agrega tu clave de API de Gemini:

```env
# API Key para Google Gemini (OBLIGATORIO)
GOOGLE_API_KEY=tu_clave_api_aqui

# Configuración MQTT (opcional, usa valores por defecto)
MQTT_BROKER=mqtt-dashboard.com
MQTT_PORT=1883
MQTT_TOPIC_SENSOR=fadena/test
MQTT_TOPIC_COMMAND=esp32-sub
```

**Obtener clave de API de Gemini**:
1. Ve a [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Crea una nueva clave de API
3. Cópiala en el archivo `.env`


## ⚙️ Configuración

### Configuración del ESP32

Si usas hardware físico, edita `esp32/main.py`:

```python
# Configuración WiFi
WIFI_SSID = "TuRedWiFi"
WIFI_PASSWORD = "TuContraseña"

# Configuración MQTT (debe coincidir con .env)
MQTT_BROKER = "mqtt-dashboard.com"
CLIENT_ID = "tu_id_unico"  # Cambia esto
TOPIC_SUB = "esp32-sub"
TOPIC_PUB = "fadena/test"
DHT_PIN = 15  # Pin GPIO del DHT22
RELAY_PIN = 2  # Pin GPIO del relay
```

### Configuración de Topics MQTT

Asegúrate de que los topics coincidan en todos los archivos:

| Componente | Topic | Propósito |
|------------|-------|-----------|
| ESP32 | `fadena/test` | Publica datos de sensores |
| ESP32 | `esp32-sub` | Recibe comandos |
| MCP Sensor Server | `fadena/test` | Suscribe a datos |
| MQTT Agent | `esp32-sub` | Publica comandos |

## 🎮 Ejecución

### Paso 1: Ejecutar el ESP32

#### Opción A: Simulación en Wokwi (Recomendado para pruebas)

1. Ve a [Wokwi.com](https://wokwi.com/)
2. Crea un nuevo proyecto ESP32
3. Copia el contenido de `esp32/main.py` al editor
4. Copia el contenido de `esp32/diagram.json` al archivo `diagram.json`
5. Haz clic en "Start Simulation"

#### Opción B: Hardware físico

1. Instala MicroPython en tu ESP32
2. Copia `esp32/main.py` al ESP32 usando herramientas como `ampy` o Thonny
3. Reinicia el ESP32

### Paso 2: Ejecutar el sistema Python

```bash
python main.py
```

Verás el menú principal:

```
+==================================================================+
|         IoT Irrigation Control System - ADK Multi-Agent         |
+==================================================================+

[MENU] Select mode:
   [1] Manual mode - Interactive testing
   [2] Automatic mode - Continuous monitoring
   [3] Chat mode - Talk with AI agent
   [q] Quit

> Enter choice (1-3, q):
```


## 🎛️ Modos de Operación

### Modo 1: Manual (Interactivo)

Permite probar manualmente cada función del sistema:

```
[MANUAL] Running in MANUAL mode...
   Available commands:
   [1] Check system status
   [2] Evaluate irrigation need
   [3] Execute irrigation decision
   [4] Send 'regar' command
   [5] Send 'no_regar' command
   [6] Switch to automatic mode
   [q] Quit
```

**Ejemplo de uso**:
```
> Enter command (1-6, q): 1

[STATUS] System Status:
   MQTT Connected: True
   Broker: mqtt-dashboard.com:1883
   Temperature: 28.5°C
   Humidity: 45.2%
   Data age: 3.2s
```

### Modo 2: Automático

Monitorea continuamente y toma decisiones cada 10 segundos:

```
[AUTO] Running in AUTOMATIC mode...
   Monitoring sensor data and making irrigation decisions
   Press Ctrl+C to stop

--------------------------------------------------
[CHECK] Time: 14:23:45
   [DATA] Temperature: 28.5°C | Humidity: 45.2%
   [IRRIGATE] IRRIGATION NEEDED:
     - Humidity (45.2%) is below 50.0%
     - Temperature (28.5°C) is above 25.0°C
     - Forecast for San Pedro: Clear (No rain)
     -> Plants need water!
   [OK] Command sent: regar
   [IRRIGATE] Irrigation command sent - ESP32 will start watering
```

### Modo 3: Chat con IA

Interactúa naturalmente con el agente orquestador:

```
[CHAT] Running in CHAT mode with Orchestrator Agent...

You: ¿Cuál es el estado actual del sistema?
Agent: El sistema está funcionando correctamente. La temperatura actual 
es de 28.5°C y la humedad es del 45.2%. Según las condiciones actuales 
y el pronóstico sin lluvia, se recomienda activar el riego.

You: Activa el riego
Agent: Comando de riego enviado exitosamente al ESP32. La bomba se 
activará en breve.
```


## 🔌 ESP32 - Hardware y Simulación

### Diagrama de Conexiones

```
ESP32 DevKit C V4
┌─────────────────────┐
│                     │
│  3V3 ●──────────────┼──── VCC (DHT22)
│                     │
│  GND ●──────────────┼──── GND (DHT22)
│                     │
│  GPIO15 ●───────────┼──── DATA (DHT22)
│                     │
│  GPIO2 ●────────────┼──── IN (Relay/LED)
│                     │
└─────────────────────┘

DHT22 Sensor          Relay Module
┌──────────┐         ┌──────────┐
│   ┌──┐   │         │  ┌────┐  │
│   └──┘   │         │  │ ⚡ │  │──── Bomba
│  1 2 3 4 │         │  └────┘  │
└──┬─┬─┬───┘         └──────────┘
   │ │ │
  VCC│GND
    DATA
```

### Archivo `diagram.json` (Wokwi)

El archivo `esp32/diagram.json` define la simulación en Wokwi:

```json
{
  "version": 1,
  "author": "Diego Racero",
  "editor": "wokwi",
  "parts": [
    {
      "type": "board-esp32-devkit-c-v4",
      "id": "esp",
      "attrs": { "env": "micropython-20231227-v1.22.0" }
    },
    {
      "type": "wokwi-dht22",
      "id": "dht1"
    }
  ],
  "connections": [
    ["dht1:VCC", "esp:3V3", "red"],
    ["dht1:GND", "esp:GND.2", "black"],
    ["dht1:SDA", "esp:15", "green"]
  ]
}
```

**Características**:
- ESP32 con MicroPython v1.22.0
- DHT22 conectado al GPIO 15
- LED integrado (GPIO 2) simula el relay

### Código MicroPython (`esp32/main.py`)

**Funciones principales**:

1. **Conexión WiFi**: Se conecta a la red configurada
2. **Conexión MQTT**: Se conecta al broker público
3. **Lectura de sensores**: Lee DHT22 cada 5 segundos
4. **Publicación de datos**: Envía JSON con temperatura y humedad
5. **Recepción de comandos**: Escucha comandos de riego
6. **Control de relay**: Activa/desactiva según comando

**Formato de datos publicados**:
```json
{
  "temp": 28.5,
  "humidity": 45.2,
  "client_id": "fadena_id",
  "irrigation_active": false
}
```

**Comandos aceptados**:
- `regar`: Activa el relay (inicia riego)
- `no_regar`: Desactiva el relay (detiene riego)
- `reset`: Reinicia el ESP32


## 🧠 Lógica de Decisión

### Algoritmo de Decisión

El **Orchestrator Agent** evalúa tres condiciones:

```python
# Condiciones para REGAR
if (humidity < 50%) AND (temperature > 25°C) AND (no_rain_forecast):
    send_command("regar")
else:
    send_command("no_regar")
```

### Umbrales Configurables

| Parámetro | Valor por Defecto | Descripción |
|-----------|-------------------|-------------|
| `HUMIDITY_THRESHOLD` | 50% | Humedad mínima aceptable |
| `TEMPERATURE_THRESHOLD` | 25°C | Temperatura mínima para riego |

Puedes ajustar estos valores usando:
```python
set_irrigation_thresholds(humidity_threshold=45.0, temperature_threshold=28.0)
```

### Flujo de Decisión

```
┌─────────────────────────────────────────────────────────────┐
│                    INICIO DE EVALUACIÓN                     │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Obtener datos sensor │
                  │  (Temp & Humedad)    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  ¿Datos disponibles? │
                  └──────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                   NO                YES
                    │                 │
                    ▼                 ▼
            ┌──────────────┐   ┌─────────────────┐
            │ Retornar     │   │ Consultar       │
            │ ERROR        │   │ pronóstico clima│
            └──────────────┘   └────────┬────────┘
                                        │
                                        ▼
                             ┌──────────────────────┐
                             │ Humedad < 50%?       │
                             └──────────┬───────────┘
                                        │
                               ┌────────┴────────┐
                               │                 │
                              NO                YES
                               │                 │
                               │                 ▼
                               │      ┌──────────────────────┐
                               │      │ Temperatura > 25°C?  │
                               │      └──────────┬───────────┘
                               │                 │
                               │        ┌────────┴────────┐
                               │        │                 │
                               │       NO                YES
                               │        │                 │
                               │        │                 ▼
                               │        │      ┌──────────────────────┐
                               │        │      │ ¿Lluvia en forecast? │
                               │        │      └──────────┬───────────┘
                               │        │                 │
                               │        │        ┌────────┴────────┐
                               │        │        │                 │
                               │        │       YES               NO
                               │        │        │                 │
                               ▼        ▼        ▼                 ▼
                        ┌──────────────────────────┐    ┌──────────────────┐
                        │  Enviar "no_regar"       │    │ Enviar "regar"   │
                        │  (No se necesita riego)  │    │ (Activar riego)  │
                        └──────────────────────────┘    └──────────────────┘
```


## 📊 Diagramas

### Diagrama de Secuencia - Ciclo Completo

```
ESP32          MQTT Broker      MCP Sensor     Orchestrator    MCP Agent      MQTT Agent
  │                 │              Server          Agent          │              │
  │                 │                │               │            │              │
  │ Leer DHT22      │                │               │            │              │
  ├────────────┐    │                │               │            │              │
  │            │    │                │               │            │              │
  │◄───────────┘    │                │               │            │              │
  │                 │                │               │            │              │
  │ Publish datos   │                │               │            │              │
  ├────────────────►│                │               │            │              │
  │                 │                │               │            │              │
  │                 │ Subscribe      │               │            │              │
  │                 ├───────────────►│               │            │              │
  │                 │                │               │            │              │
  │                 │                │ Almacenar    │            │              │
  │                 │                ├──────────┐   │            │              │
  │                 │                │          │   │            │              │
  │                 │                │◄─────────┘   │            │              │
  │                 │                │               │            │              │
  │                 │                │               │ Evaluar    │              │
  │                 │                │               │ necesidad  │              │
  │                 │                │               ├──────────┐ │              │
  │                 │                │               │          │ │              │
  │                 │                │               │◄─────────┘ │              │
  │                 │                │               │            │              │
  │                 │                │ get_sensor_   │            │              │
  │                 │                │ data()        │            │              │
  │                 │                │◄──────────────┤            │              │
  │                 │                │               │            │              │
  │                 │                │ Retornar datos│            │              │
  │                 │                ├──────────────►│            │              │
  │                 │                │               │            │              │
  │                 │                │               │ Consultar  │              │
  │                 │                │               │ clima      │              │
  │                 │                │               ├───────────►│              │
  │                 │                │               │            │              │
  │                 │                │               │            │ Llamar MCP  │
  │                 │                │               │            │ Weather     │
  │                 │                │               │            ├──────────┐  │
  │                 │                │               │            │          │  │
  │                 │                │               │            │◄─────────┘  │
  │                 │                │               │            │              │
  │                 │                │               │ Forecast   │              │
  │                 │                │               │◄───────────┤              │
  │                 │                │               │            │              │
  │                 │                │               │ Decidir:   │              │
  │                 │                │               │ "regar"    │              │
  │                 │                │               ├──────────┐ │              │
  │                 │                │               │          │ │              │
  │                 │                │               │◄─────────┘ │              │
  │                 │                │               │            │              │
  │                 │                │               │ send_irrigation_command() │
  │                 │                │               ├──────────────────────────►│
  │                 │                │               │            │              │
  │                 │                │               │            │ Publish cmd │
  │                 │◄───────────────────────────────────────────────────────────┤
  │                 │                │               │            │              │
  │ Subscribe cmd   │                │               │            │              │
  │◄────────────────┤                │               │            │              │
  │                 │                │               │            │              │
  │ Activar relay   │                │               │            │              │
  ├──────────┐      │                │               │            │              │
  │          │      │                │               │            │              │
  │◄─────────┘      │                │               │            │              │
  │                 │                │               │            │              │
  │ 💧 RIEGO ACTIVO │                │               │            │              │
  │                 │                │               │            │              │
```


### Diagrama de Flujo de Datos

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FLUJO DE DATOS                              │
└─────────────────────────────────────────────────────────────────────┘

1. CAPTURA DE DATOS
   ┌──────────┐
   │  DHT22   │ ──► Temperatura: 28.5°C
   │  Sensor  │ ──► Humedad: 45.2%
   └────┬─────┘
        │
        ▼
   ┌──────────┐
   │  ESP32   │ ──► JSON: {"temp": 28.5, "humidity": 45.2, ...}
   └────┬─────┘
        │
        ▼

2. TRANSMISIÓN MQTT
   ┌──────────────┐
   │ MQTT Publish │ ──► Topic: "fadena/test"
   │   (QoS 2)    │ ──► Garantía de entrega
   └──────┬───────┘
          │
          ▼

3. RECEPCIÓN Y ALMACENAMIENTO
   ┌────────────────┐
   │ MCP Sensor     │ ──► Almacena en memoria
   │ Server         │ ──► Timestamp de actualización
   │ (Subscribe)    │ ──► Detecta datos obsoletos
   └────────┬───────┘
            │
            ▼

4. ANÁLISIS Y DECISIÓN
   ┌────────────────┐
   │ Orchestrator   │ ──► Lee datos del sensor
   │ Agent          │ ──► Consulta pronóstico
   │ (Gemini 2.0)   │ ──► Aplica lógica de decisión
   └────────┬───────┘     ──► Genera comando
            │
            ▼

5. EJECUCIÓN
   ┌────────────────┐
   │ MQTT Agent     │ ──► Publica comando
   └────────┬───────┘
            │
            ▼
   ┌────────────────┐
   │ MQTT Publish   │ ──► Topic: "esp32-sub"
   │   (QoS 2)      │ ──► Comando: "regar" o "no_regar"
   └────────┬───────┘
            │
            ▼

6. ACTUACIÓN
   ┌────────────────┐
   │ ESP32          │ ──► Recibe comando
   │ (Subscribe)    │ ──► Activa/desactiva GPIO 2
   └────────┬───────┘
            │
            ▼
   ┌────────────────┐
   │ Relay/Bomba    │ ──► 💧 Riego activado/desactivado
   └────────────────┘
```


### Arquitectura Multi-Agente ADK

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GOOGLE ADK MULTI-AGENT SYSTEM                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR AGENT                          │
│                        (Gemini 2.0 Flash)                           │
│                                                                     │
│  Rol: Coordinador principal y tomador de decisiones                │
│                                                                     │
│  Herramientas:                                                      │
│  • evaluate_irrigation_need()                                       │
│  • execute_irrigation_decision()                                    │
│  • set_irrigation_thresholds()                                      │
│  • get_system_status()                                              │
│                                                                     │
│  Lógica de Decisión:                                                │
│  IF (Humedad < 50%) AND (Temp > 25°C) AND (Sin lluvia)             │
│     THEN: Activar riego                                             │
│     ELSE: Mantener apagado                                          │
└────────────────────┬────────────────────────────┬───────────────────┘
                     │                            │
                     │ Usa herramientas           │ Usa herramientas
                     │                            │
        ┌────────────▼──────────┐    ┌────────────▼──────────┐
        │     MQTT AGENT        │    │     MCP AGENT         │
        │  (Gemini 2.0 Flash)   │    │  (Gemini 2.0 Flash)   │
        │                       │    │                       │
        │  Rol: Comunicación    │    │  Rol: Datos externos  │
        │  con ESP32            │    │  (Clima)              │
        │                       │    │                       │
        │  Herramientas:        │    │  Herramientas:        │
        │  • send_irrigation_   │    │  • get_weather_       │
        │    command()          │    │    forecast_tool()    │
        │  • get_mqtt_status()  │    │                       │
        │  • get_sensor_data()  │    │  Conecta con:         │
        │  • get_temperature()  │    │  • Weather MCP Server │
        │  • get_humidity()     │    │  • NWS API            │
        └───────────┬───────────┘    └───────────┬───────────┘
                    │                            │
                    │ Usa                        │ Usa
                    │                            │
        ┌───────────▼──────────┐    ┌────────────▼──────────┐
        │  MCP SENSOR SERVER   │    │ EXTERNAL MCP SERVER   │
        │  (No es agente ADK)  │    │ (weather.py)          │
        │                      │    │                       │
        │  • Listener MQTT     │    │  • Servidor MCP       │
        │  • Almacena datos    │    │  • API NWS            │
        │  • Expone tools      │    │  • Pronóstico clima   │
        └──────────────────────┘    └───────────────────────┘
```

**Características clave**:

1. **Separación de responsabilidades**: Cada agente tiene un rol específico
2. **Comunicación mediante herramientas**: Los agentes se comunican a través de function calls
3. **Modelo unificado**: Todos usan Gemini 2.0 Flash
4. **Escalabilidad**: Fácil agregar nuevos agentes o herramientas

