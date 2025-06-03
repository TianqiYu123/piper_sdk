#!/usr/bin/env python3
# -*-coding:utf8-*-
from math import pi
import sys
import select
import time
import json
from piper_sdk import *
import paho.mqtt.client as mqtt
from os.path import dirname, join, abspath
import threading  # Import threading
import logging  # Import logging

# Configure logging
#logging.basicConfig(level=logging.DEBUG,  # Set log level
#                    format='%(asctime)s - %(levelname)s - %(message)s')

# MQTT Configuration
MQTT_BROKER = "47.96.170.89"
MQTT_PORT = 8003
MQTT_TOPIC = "arm/pose/#"
MQTT_CLIENT_ID = "endpose_reader001"

# Robot Configuration
FACTOR = 57290
INITIAL_END_POSE = [0, 0, 0, 0, pi / 2, 0]
NUM_JOINTS = 6  # number of joints you are controlling

# Helper Functions (enable_fun, is_data_available, clamp) - No Changes Here
def enable_fun(piper: C_PiperInterface):
    enable_flag = False
    timeout = 5
    start_time = time.time()
    elapsed_time_flag = False
    while not enable_flag:
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = all(
            [
                getattr(piper.GetArmLowSpdInfoMsgs(), f"motor_{i+1}").foc_status.driver_enable_status
                for i in range(NUM_JOINTS)
            ]
        )  # use NUM_JOINTS
        print("使能状态:", enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0, 1000, 0x01, 0)
        print("--------------------")
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)

    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        exit(0)


def is_data_available():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [], 0)



def process_message(payload):
    try:
        data = json.loads(payload)
        if "info" in data and isinstance(data["info"], list) and len(data["info"]) == 15:
            (
                x,
                y,
                z,
                rx,
                ry,
                rz,
                tmp,  # Discard qx, qy, qz, qw
                trigger,
                joystickX,
                joystickY,
                joystickClick,
                buttonA,
                buttonB,
                grip,
                temp,
            ) = data["info"]
            # rx, ry, rz = euler.quat2euler([qw, qx, qy, qz], "sxyz") # NO NEED TO CALCULATE EULER
            endpose_delta = [x, y, z, rx, ry, rz]  # DIRECTLY ASSIGN THE FIRST 6 ELEMENTS
            logging.debug(f"Processed MQTT message: endpose_delta={endpose_delta}, trigger={trigger}, ...")
            return endpose_delta, trigger, joystickX, joystickY, joystickClick, buttonA, buttonB, grip
        else:
            logging.warning("Invalid data format")
            return None, None, None, None, None, None, None, None
    except (json.JSONDecodeError, Exception) as e:
        logging.error(f"Error processing message: {e}", exc_info=True) # Include traceback
        return None, None, None, None, None, None, None, None


class MQTTHandler:
    def __init__(self, broker, port, topic, client_id):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        #self.client.on_message = self.on_message # Removed direct assignment
        self.client.on_message = self._on_message_callback # Use internal callback
        self.endpose_delta = None
        self.trigger = None
        self.joystickX = None
        self.joystickY = None
        self.joystickClick = None
        self.buttonA = None
        self.buttonB = None
        self.grip = None
        self.message_received = False  # Flag to track if a message has been processed
        self._lock = threading.Lock() # Add a lock
        self.message_count = 0  # Track the number of messages received
        self.processed_count = 0 # Track number of messages processed

        # Configure MQTT client options:
        self.client.max_inflight_messages_set(1000) # allows up to 1000 messages inflight
        self.client.reconnect_delay_set(min_delay=1, max_delay=120) #configure reconnect backoff

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT Broker!")
            client.subscribe(self.topic, qos=1)  # Set QoS to 1
        else:
            logging.error(f"Failed to connect, return code {rc}")
            # Optionally, attempt to reconnect here or set a flag to retry later

    def _on_message_callback(self, client, userdata, msg):
        """Internal callback to handle messages in a thread-safe manner."""
        self.message_count += 1  # Increment message counter
        logging.debug(f"Message received. Total messages received: {self.message_count}")
        # Spin off a thread to handle the message
        threading.Thread(target=self._process_message, args=(msg,)).start()

    def _process_message(self, msg):
        """Processes the MQTT message and updates internal state."""
        start_time = time.time() # Time the processing

        try:
            endpose_delta, trigger, joystickX, joystickY, joystickClick, buttonA, buttonB, grip = process_message(
                msg.payload.decode()
            )
            with self._lock: # Acquire the lock
                self.endpose_delta = endpose_delta
                self.trigger = trigger
                self.joystickX = joystickX
                self.joystickY = joystickY
                self.joystickClick = joystickClick
                self.buttonA = buttonA
                self.buttonB = buttonB
                self.grip = grip
                self.message_received = True  # Set the flag when a message is received
                self.processed_count += 1

        except Exception as e:
            logging.error(f"Error in _process_message: {e}", exc_info=True)
        finally:
             pass # Release the lock.  Lock is handled by 'with'
        processing_time = time.time() - start_time
        logging.debug(f"Message processed.  Processing time: {processing_time:.4f} seconds. Total messages processed: {self.processed_count}")

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            logging.error(f"Connection error: {e}", exc_info=True)

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def get_righthand(self):
        with self._lock: # Acquire the lock
            # Create a copy of the variables to avoid race conditions
            endpose_delta = self.endpose_delta
            trigger = self.trigger
            joystickX = self.joystickX
            joystickY = self.joystickY
            joystickClick = self.joystickClick
            buttonA = self.buttonA
            buttonB = self.buttonB
            grip = self.grip
            message_received = self.message_received
            # Reset message_received flag
            self.message_received = False

        return (
            endpose_delta,
            trigger,
            joystickX,
            joystickY,
            joystickClick,
            buttonA,
            buttonB,
            grip,
            message_received  # Return the value of message_received
        )



import subprocess

def main():
    try:
        subprocess.run(["bash", "can_activate.sh", "can0", "1000000"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running can_activate.sh: {e}")
        return

    mqtt_handler = MQTTHandler(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CLIENT_ID)
    mqtt_handler.connect()
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)

    #acc_limit = 270  # Define the acceleration limit
    acc_limit = 200
    piper.JointConfig(joint_num=7, set_zero=0, acc_param_is_effective=0xAE, max_joint_acc=acc_limit, clear_err=0xAE)
    time.sleep(1)


    current_end_pose = INITIAL_END_POSE[:]

    # Initialize joint angles
    joint_0 = 0
    joint_1 = 0
    joint_2 = 0
    joint_4 = 0


    try:
        while True:
            (
                endpose_delta,
                trigger,
                joystickX,
                joystickY,
                joystickClick,
                buttonA,
                buttonB,
                grip,
                message_received # Get the message_received flag
            ) = mqtt_handler.get_righthand()

            # REMOVED BUTTON A AND BUTTON B CONTROL

            if message_received and endpose_delta is not None: # Only process if a message was received and processed
                # No need to reset mqtt_handler.message_received here, done in get_righthand()
                try:
                    endpose_delta = [float(x) for x in endpose_delta]

                    if len(endpose_delta) != 6:
                        logging.warning(f"Error: Incorrect endpose length ({len(endpose_delta)}), expected 6.")
                        continue

                except (ValueError, TypeError) as e:
                    logging.error(f"Error unpacking MQTT: {e}", exc_info=True)
                    continue

                # Accumulate the delta values to the current end pose
                new_end_pose = [current_end_pose[i] + endpose_delta[i] for i in range(6)]

                current_end_pose = new_end_pose[:]

                # Direct Joint Control based on Z-axis rotation
                z_rotation_delta = endpose_delta[5]
                z_delta = endpose_delta[2]

                joint_0 += z_rotation_delta * FACTOR # Map z-axis rotation to joint0
                joint_1 += z_delta * 1 * FACTOR # Scale the increment for joint 1
                joint_2 -= z_delta * 2 * FACTOR # Scale the increment for joint 2
                joint_4 += z_delta * 1 * FACTOR # Scale the increment for joint 4
                print("joint_0:", joint_0)
                print("joint_1:", joint_1)
                print("joint_2:", joint_2)
                print("joint_4:", joint_4)


                # Convert to int and Send Motor signal
                joint_0_send = round(joint_0)  # was joint_angles[0]
                joint_1_send = round(joint_1)  # was joint_angles[1]
                joint_2_send = round(joint_2)  # was joint_angles[2]
                joint_3_send = 0 #  fixed
                joint_4_send = round(joint_4)  # was joint_angles[4]
                joint_5_send = 0  # Joint Angle fixed = 0

                piper.MotionCtrl_2(0x01, 0x01, 90, 0x00)
                piper.JointCtrl(joint_0_send, joint_1_send, joint_2_send, joint_3_send, joint_4_send, joint_5_send)


                old_end_pose = current_end_pose[:]

            time.sleep(0.01)  # Add a small delay to avoid excessive CPU usage

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        mqtt_handler.disconnect()
        piper.DisconnectPort()


if __name__ == "__main__":
    main()