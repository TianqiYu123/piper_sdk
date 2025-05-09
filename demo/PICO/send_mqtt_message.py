import paho.mqtt.client as mqtt
import time
import json
import keyboard  # Requires the `keyboard` library
from math import pi

# MQTT Broker Information
MQTT_BROKER = "47.96.170.89"
MQTT_PORT = 8003
MQTT_TOPIC = "arm/pose/test"
MQTT_CLIENT_ID = "endpose_publisher"
d_l = 0.005
d_r = 0.05


# Initial pose values (x, y, z, rx, ry, rz) and grip
x = 0.068 #0.18
y = 0.0
z = 0.28  #0.365
rx = 0.0
ry = pi/2
rz = 0.0
trigger = 0.0   # 0 open, 1 close

def on_connect(client, userdata, flags, rc):
    """Callback function when the MQTT client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

client = mqtt.Client(client_id=MQTT_CLIENT_ID)
client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
except Exception as e:
    print(f"Connection error: {e}")
    exit()

client.loop_start()

try:
    while True:
        # Check for keyboard input to adjust pose values
        if keyboard.is_pressed('q'):
            x += d_l
        elif keyboard.is_pressed('a'):
            x -= d_l

        if keyboard.is_pressed('w'):
            y += d_l
        elif keyboard.is_pressed('s'):
            y -= d_l

        if keyboard.is_pressed('e'):
            z += d_l
        elif keyboard.is_pressed('d'):
            z -= d_l

        if keyboard.is_pressed('r'):
            rx += d_r
        elif keyboard.is_pressed('f'):
            rx -= d_r

        if keyboard.is_pressed('t'):
            ry += d_r
        elif keyboard.is_pressed('g'):
            ry -= d_r

        if keyboard.is_pressed('y'):
            rz += d_r
        elif keyboard.is_pressed('h'):
            rz -= d_r
        
        if keyboard.is_pressed('u'):
            trigger += d_r
        elif keyboard.is_pressed('j'):
            trigger -= d_r


        # Create the message with the current pose values (x, y, z, rx, ry, rz)
        message = {
            "info": [x, y, z, rx, ry, rz, 0.0, trigger, 0.0, 0.0, 0, 0, 0, 0, 0.0]
        }

        payload = json.dumps(message)

        client.publish(MQTT_TOPIC, payload)
        print(f"Published message to topic {MQTT_TOPIC}: {payload}")

        time.sleep(0.05)  # Small delay for keyboard input responsiveness

except KeyboardInterrupt:
    print("Exiting...")
finally:
    client.loop_stop()
    client.disconnect()
    print("Disconnected from MQTT Broker.")