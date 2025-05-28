#!/usr/bin/env python3
# -*-coding:utf8-*-
import numpy as np
from math import pi
import transforms3d.euler as euler
from scipy.spatial.transform import Rotation
import sys
import select
import time
import json
from piper_sdk import *
import pinocchio as pin
from scipy.optimize import minimize
import paho.mqtt.client as mqtt
import os
from os.path import dirname, join, abspath
import random  # Import the random module
import threading  # Import threading
import logging  # Import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Set log level
                    format='%(asctime)s - %(levelname)s - %(message)s')

# MQTT Configuration
MQTT_BROKER = "47.96.170.89"
MQTT_PORT = 8003
MQTT_TOPIC = "arm/pose/#"
MQTT_CLIENT_ID = "endpose_reader"

# Robot Configuration
FACTOR = 57290
INITIAL_END_POSE = [0.1, 0, 0.1, 0, pi / 2, 0]
NUM_JOINTS = 6  # number of joints you are controlling

# End Effector Limits
X_LIMIT = [0.0, 0.3]
Y_LIMIT = [-0.15, 0.15]
Z_LIMIT = [0.1, 0.6]
RX_LIMIT = [-pi / 2, pi / 2]
RY_LIMIT = [pi / 2 - 0.1, pi / 2 + 0.1]
RZ_LIMIT = [-pi, pi]

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

def clamp(value, limits):
    """Clamp a value within the given limits."""
    lower, upper = limits
    return max(lower, min(value, upper))


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

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            logging.info("Connected to MQTT Broker!")
            client.subscribe(self.topic, qos=1)  # Set QoS to 1
        else:
            logging.error(f"Failed to connect, return code {rc}")

    def _on_message_callback(self, client, userdata, msg):
        """Internal callback to handle messages in a thread-safe manner."""
        # Spin off a thread to handle the message
        threading.Thread(target=self._process_message, args=(msg,)).start()

    def _process_message(self, msg):
        """Processes the MQTT message and updates internal state."""
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
        except Exception as e:
            logging.error(f"Error in _process_message: {e}", exc_info=True)
        finally:
             pass # Release the lock.  Lock is handled by 'with'

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


def cost_function(q, model, data, target_pose):
    """Cost function for IK optimization.  Measures the distance between the current end-effector pose and the target pose."""
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    current_pose = data.oMf[model.getFrameId("gripper_base")].homogeneous  # change tool_frame to your frame name.

    # Calculate position error
    position_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])

    # Calculate orientation error (using rotation matrix difference)
    rotation_error = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3])
    # print("Error",position_error + rotation_error)
    return position_error + rotation_error * 0.1


def inverse_kinematics(model, data, target_pose, q_init=None):
    """Inverse kinematics using optimization."""
    if q_init is None:
        q_init = np.zeros(model.nq)  # Initial guess for joint angles

    # Optimization bounds (joint limits) - Optional, but highly recommended
    bounds = [(model.lowerPositionLimit[i], model.upperPositionLimit[i]) for i in range(model.nq)]

    start_time = time.time()

    # result = minimize(cost_function, q_init, args=(model, data, target_pose), method="SLSQP", bounds=bounds, tol=1e-4)  # tol is tolerance
    result = minimize(cost_function, q_init, args=(model, data, target_pose), method="SLSQP", bounds=bounds,
                      tol=1e-4)  # tol is tolerance

    elapsed_time = time.time() - start_time
    if result.success:
        return result.x, True, elapsed_time
    else:
        return None, False, elapsed_time


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


    # Load the robot model using pinocchio
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "PICO")  # Adjust path as necessary
    # model_path = join(pinocchio_model_dir, "/")
    model_path = pinocchio_model_dir
    # /home/dm/robot_arm/piper_sdk_PICO/demo/PICO/piper_description/urdf/piper_description.urdf
    mesh_dir = pinocchio_model_dir
    urdf_filename = "piper_description.urdf"  # Use your robot's URDF
    urdf_model_path = join(join(model_path, "piper_description/urdf"), urdf_filename)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)
    data = model.createData()

    current_end_pose = INITIAL_END_POSE[:]
    old_end_pose = INITIAL_END_POSE[:]
    last_joint_angles = np.zeros(model.nq)

    # Define the maximum random delta for each component of the end pose
    MAX_DELTA_POSITION = 0.001  # Adjust as needed (e.g., 1mm)
    MAX_DELTA_ORIENTATION = 0.001  # Adjust as needed (e.g., 0.1 degrees in radians)

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

                # Add a small random delta to each component of the new end pose
                random_delta = [
                    random.uniform(-MAX_DELTA_POSITION, MAX_DELTA_POSITION),  # x
                    random.uniform(-MAX_DELTA_POSITION, MAX_DELTA_POSITION),  # y
                    random.uniform(-MAX_DELTA_POSITION, MAX_DELTA_POSITION),  # z
                    random.uniform(-MAX_DELTA_ORIENTATION, MAX_DELTA_ORIENTATION),  # rx
                    random.uniform(-MAX_DELTA_ORIENTATION, MAX_DELTA_ORIENTATION),  # ry
                    random.uniform(-MAX_DELTA_ORIENTATION, MAX_DELTA_ORIENTATION),  # rz
                ]
                new_end_pose = [new_end_pose[i] + random_delta[i] for i in range(6)]


                # Apply limits to the new end pose
                new_end_pose[0] = clamp(new_end_pose[0], X_LIMIT)
                new_end_pose[1] = clamp(new_end_pose[1], Y_LIMIT)
                new_end_pose[2] = clamp(new_end_pose[2], Z_LIMIT)
                new_end_pose[3] = clamp(new_end_pose[3], RX_LIMIT)
                new_end_pose[4] = clamp(new_end_pose[4], RY_LIMIT)
                new_end_pose[5] = clamp(new_end_pose[5], RZ_LIMIT)

                current_end_pose = new_end_pose[:]

                # IK to find the joint angles, send angles to robot directly
                target_pose = np.eye(4)
                target_pose[:3, :3] = Rotation.from_euler("xyz", current_end_pose[3:]).as_matrix()
                target_pose[:3, 3] = current_end_pose[:3]

                # Print the target pose (position and orientation)
                logging.debug("Target Pose:")
                logging.debug(f"Position: {target_pose[:3, 3]}")
                logging.debug(f"Orientation (Euler XYZ): {current_end_pose[3:]}")

                joint_angles, success, elapsed_time = inverse_kinematics(
                    model, data, target_pose, last_joint_angles
                )

                if success:
                    logging.debug("IK Successful!")  # Print if IK was successful
                    last_joint_angles = joint_angles

                    # Convert to int and Send Motor signal
                    joint_0 = round(joint_angles[0] * FACTOR)
                    joint_1 = round(joint_angles[1] * FACTOR)
                    joint_2 = round(joint_angles[2] * FACTOR)
                    joint_3 = round(joint_angles[3] * FACTOR)
                    joint_4 = round(joint_angles[4] * FACTOR)
                    joint_5 = round((joint_angles[5]) * FACTOR)  # Joint Angle is not always = 0
                    joint_6 = round((1 - grip) / 5 * 1000 * 1000)  # Grip control

                    print("joint_0:", joint_0)
                    print("joint_1:", joint_1)
                    print("joint_2:", joint_2)
                    print("joint_4:", joint_4)

                    piper.MotionCtrl_2(0x01, 0x01, 50, 0x00)
                    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

                else:
                    logging.warning("IK Failed")  # Print if IK failed
                old_end_pose = current_end_pose[:]

            time.sleep(0.01)  # Add a small delay to avoid excessive CPU usage

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        mqtt_handler.disconnect()
        piper.DisconnectPort()


if __name__ == "__main__":
    main()