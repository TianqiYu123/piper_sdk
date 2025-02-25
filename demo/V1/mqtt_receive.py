import paho.mqtt.client as mqtt
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
    """Processes the MQTT message payload and returns the endpose.

    Args:
        payload: The MQTT message payload as a string.

    Returns:
        A list representing the endpose (first 6 elements of 'info') or None if there's an error.
    """
    try:
        data = json.loads(payload)

        if "info" in data and isinstance(data["info"], list) and len(data["info"]) >= 6:
            endpose = data["info"][:6]  # Extract the first 6 elements
            return endpose
        else:
            print("Invalid data format: 'info' key missing, not a list, or less than 6 elements.")
            print(f"Received message: {payload}")  # print the entire message to help with debugging
            return None  # Indicate an error
    except json.JSONDecodeError:
        print("Error decoding JSON from MQTT message.")
        print(f"Received message: {payload}")
        return None  # Indicate an error
    except Exception as e:
        print(f"An error occurred: {e}")
        print(f"Received message: {payload}")
        return None #Indicate an error


class MQTTHandler:  #Encapsulate MQTT functionality in a class
    def __init__(self, broker, port, topic, client_id):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.endpose = None  # Store the most recently received endpose

    def on_connect(self, client, userdata, flags, rc):
        """Callback function when the MQTT client connects to the broker."""
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(self.topic)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        """Callback function when a message is received from the MQTT broker."""
        endpose = process_message(msg.payload.decode())
        if endpose:
            self.endpose = endpose
            #print(f"Received endpose: {self.endpose}")  # Optionally print here

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
        """Returns the most recently received endpose.  Returns None if no endpose has been received yet."""
        return self.endpose

