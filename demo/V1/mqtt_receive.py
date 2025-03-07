import paho.mqtt.client as mqtt
import transforms3d.euler as euler
import numpy as np
import json

MQTT_BROKER = "47.96.170.89"  # Replace with your MQTT broker address
MQTT_PORT = 8003 # Default MQTT port.  If you are using a different port, change this.  8003 is typically a web port (HTTP).
MQTT_TOPIC = "arm/pose/#"  # Replace with the MQTT topic you are subscribing to
MQTT_CLIENT_ID = "endpose_reader" #A unique client ID is highly recommended for robust performance.


def on_connect(client, userdata, flags, rc):
    """Callback function when the MQTT client connects to the broker."""
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        print(f"Failed to connect, return code {rc}")


def process_message(payload):
    """Processes the MQTT message payload and returns the endpose and trigger.

    Args:
        payload: The MQTT message payload as a string.

    Returns:
        A tuple containing the endpose (x, y, z, rx, ry, rz) and the trigger (0 or 1), 
        or (None, None) if there's an error.
    """
    try:
        data = json.loads(payload)

        if "info" in data and isinstance(data["info"], list) and len(data["info"]) == 9:
            x, y, z, qx, qy, qz, qw, trigger,temp = data["info"]
            #print("trigger",trigger)
            # Convert quaternion to Euler angles (rx, ry, rz)
            # Note: The order of rotation is 'xyz' which is a common convention.
            rx, ry, rz = euler.quat2euler([qw, qx, qy, qz], 'sxyz')

            endpose = [x, y, z, rx, ry, rz]
            return endpose, int(trigger)
        else:
            print("Invalid data format: 'info' key missing, not a list, or not exactly 8 elements.")
            print(f"Received message: {payload}")  # print the entire message to help with debugging
            return None, None  # Indicate an error
    except json.JSONDecodeError:
        print("Error decoding JSON from MQTT message.")
        print(f"Received message: {payload}")
        return None, None  # Indicate an error
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Received message: {payload}")
        return None, None  # Indicate an error


class MQTTHandler:
    def __init__(self, broker, port, topic, client_id):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.endpose = None  # Store the most recently received endpose
        self.trigger = None  # Store the most recently received trigger

    def on_connect(self, client, userdata, flags, rc):
        """Callback function when the MQTT client connects to the broker."""
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(self.topic)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        """Callback function when a message is received from the MQTT broker."""
        endpose, trigger = process_message(msg.payload.decode())
        if endpose is not None:
            self.endpose = endpose
            self.trigger = trigger
            print(f"Received endpose: {self.endpose}, Trigger: {self.trigger}")

    def connect(self):
        """Connects to the MQTT broker and starts the event loop."""
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()  # Use loop_start for non-blocking operation
        except Exception as e:
            print(f"Connection error: {e}")

    def disconnect(self):
        """Disconnects from the MQTT broker."""
        self.client.loop_stop() # Stop the loop first
        self.client.disconnect()

    def get_endpose(self):
        """Returns the most recently received endpose and trigger.
        Returns (None, None) if no message has been received yet."""
        return self.endpose, self.trigger