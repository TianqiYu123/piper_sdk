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
import subprocess

from pinocchio.visualize import MeshcatVisualizer

MQTT_BROKER = "47.96.170.89"
MQTT_PORT = 8003
MQTT_TOPIC = "arm/pose/robot_test_arm"
MQTT_CLIENT_ID = "endpose_reader"

# FACTOR = 57324.840764
FACTOR = 57290
# INITIAL_END_POSE = [150, 0, 260, 0, pi/2, 0]
# INITIAL_END_POSE = [0, 0, 0, 0, pi / 2, 0]
INITIAL_END_POSE = [0.2, 0, 0.18, 0, pi / 2, 0]
NUM_JOINTS = 6  # number of joints you are controlling


def enable_fun(piper: C_PiperInterface):
    """Enables the arm motors and gripper."""
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


def left_to_right_hand(x_left, y_left, z_left, rx_left, ry_left, rz_left):
    """Converts coordinates from a left-handed to a right-handed coordinate system."""
    x_right = z_left
    y_right = -x_left
    z_right = y_left
    r_left = Rotation.from_euler("zyx", [rz_left, ry_left, rx_left])
    rot_matrix_left = r_left.as_matrix()
    transform_matrix = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]])
    transform_matrix_inv = transform_matrix.T
    rot_matrix_right = transform_matrix @ rot_matrix_left @ transform_matrix_inv
    r_right = Rotation.from_matrix(rot_matrix_right)
    rz_right, ry_right, rx_right = r_right.as_euler("zyx")
    return [x_right, y_right, z_right, rx_right, ry_right, rz_right]


def process_message(payload):
    """Processes an MQTT message payload and extracts end-effector pose and other data."""
    try:
        data = json.loads(payload)
        if "info" in data and isinstance(data["info"], list) and len(data["info"]) == 15:
            (
                x,
                y,
                z,
                qx,
                qy,
                qz,
                qw,
                trigger,
                joystickX,
                joystickY,
                joystickClick,
                buttonA,
                buttonB,
                grip,
                temp,
            ) = data["info"]
            rx, ry, rz = euler.quat2euler([qw, qx, qy, qz], "sxyz")
            rz = 0  # limit rotation
            rx = 0  # limit rotation
            endpose = left_to_right_hand(x, y, z, rx, ry, rz)
            return endpose, int(trigger), joystickX, joystickY, joystickClick, buttonA, buttonB, grip
        else:
            print("Invalid data format")
            return None, None, None, None, None, None, None, None
    except (json.JSONDecodeError, Exception) as e:
        print(f"Error: {e}")
        return None, None, None, None, None, None, None, None


class MQTTHandler:
    def __init__(self, broker, port, topic, client_id):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.client_id = client_id
        self.client = mqtt.Client(client_id=self.client_id)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.endpose = None
        self.trigger = None
        self.joystickX = None
        self.joystickY = None
        self.joystickClick = None
        self.buttonA = None
        self.buttonB = None
        self.grip = None

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(self.topic)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        endpose, trigger, joystickX, joystickY, joystickClick, buttonA, buttonB, grip = process_message(
            msg.payload.decode()
        )
        self.endpose = endpose
        self.trigger = trigger
        self.joystickX = joystickX
        self.joystickY = joystickY
        self.joystickClick = joystickClick
        self.buttonA = buttonA
        self.buttonB = buttonB
        self.grip = grip

    def connect(self):
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()
        except Exception as e:
            print(f"Connection error: {e}")

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()

    def get_righthand(self):
        return (
            self.endpose,
            self.trigger,
            self.joystickX,
            self.joystickY,
            self.joystickClick,
            self.buttonA,
            self.buttonB,
            self.grip,
        )


def cost_function(q, model, data, target_pose):
    """Cost function for IK optimization.  Measures the distance between the current end-effector pose and the target pose."""
    # Create the full joint angle vector, setting the fixed joints to 0
    q_full = np.zeros(model.nq)
    q_full[0] = q[0]  # Joint 0
    q_full[1] = q[1]  # Joint 1
    q_full[2] = q[2]  # Joint 2
    q_full[3] = 0  # Joint 3 is fixed at 0
    q_full[4] = q[3]  # Joint 4
    q_full[5] = 0  # Joint 5 is fixed at 0

    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    current_pose = data.oMf[model.getFrameId("gripper_base")].homogeneous  # change tool_frame to your frame name.

    # Calculate position error
    position_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])

    # Calculate orientation error (using rotation matrix difference)
    rotation_error = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3])
    # print("Error",position_error + rotation_error)
    # return position_error + rotation_error*0.12
    return position_error + rotation_error * 0.2


def inverse_kinematics(model, data, target_pose, q_init=None):
    """Inverse kinematics using optimization."""
    if q_init is None:
        q_init = np.zeros(4)  # Initial guess for joint angles, now just 4 joints.

    # Optimization bounds (joint limits) - Optional, but highly recommended
    bounds = [(model.lowerPositionLimit[i], model.upperPositionLimit[i]) for i in [0, 1, 2, 4]]  # Only bounds for active joints

    start_time = time.time()

    result = minimize(
        cost_function, q_init, args=(model, data, target_pose), method="SLSQP", bounds=bounds, tol=1e-4
    )  # tol is tolerance
    elapsed_time = time.time() - start_time
    if result.success:
        return result.x, True, elapsed_time
    else:
        return None, False, elapsed_time


def run_can_activation_script():
    """Runs the can_activate.sh script."""
    try:
        subprocess.run(["bash", "can_activate.sh", "can0", "1000000"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running can_activate.sh: {e}")
        sys.exit(1)  # Exit if the script fails


def initialize_robot(piper: C_PiperInterface):
    """Initializes the robot arm: enables motors, moves to initial pose."""
    piper.EnableArm(7)
    enable_fun(piper=piper)

    # Move to initial pose
    target_joint_angles = np.zeros(NUM_JOINTS)  # Return to zero angles

    # Convert to int and Send Motor signal
    joint_0 = round(target_joint_angles[0] * FACTOR)
    joint_1 = round(target_joint_angles[1] * FACTOR)
    joint_2 = round(target_joint_angles[2] * FACTOR)
    joint_3 = round(target_joint_angles[3] * FACTOR)
    joint_4 = round(target_joint_angles[4] * FACTOR)
    joint_5 = round((target_joint_angles[5]) * FACTOR)  # Joint Angle is not always = 0
    joint_6 = round(0.2 * 1000 * 1000)

    piper.MotionCtrl_2(0x01, 0x01, 80, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    time.sleep(0.5)  # Wait for the robot to reach the initial pose

    acc_limit = 500  # Define the acceleration limit
    piper.JointConfig(joint_num=7, set_zero=0, acc_param_is_effective=0xAE, max_joint_acc=acc_limit, clear_err=0xAE)
    time.sleep(0.5)


def load_pinocchio_model():
    """Loads the robot model using pinocchio."""
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "PICO")  # Adjust path as necessary
    model_path = pinocchio_model_dir
    mesh_dir = pinocchio_model_dir
    urdf_filename = "piper_description.urdf"  # Use your robot's URDF
    urdf_model_path = join(join(model_path, "piper_description/urdf"), urdf_filename)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)
    data = model.createData()
    return model, collision_model, visual_model, data


def setup_meshcat_visualizer(model, collision_model, visual_model):
    """Initializes and sets up the Meshcat visualizer."""
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install Python meshcat")
        print(err)
        sys.exit(0)
    viz.loadViewerModel()
    return viz


def move_to_initial_pose(piper):
        """Moves the robot to a predefined initial pose."""
        target_joint_angles = [0.0, 0.992, -0.40, 0.0, -0.50, 0.0]
        joint_0 = round(target_joint_angles[0] * FACTOR)
        joint_1 = round(target_joint_angles[1] * FACTOR)
        joint_2 = round(target_joint_angles[2] * FACTOR)
        joint_3 = round(target_joint_angles[3] * FACTOR)  # Force to 0
        joint_4 = round(target_joint_angles[4] * FACTOR)
        joint_5 = round((target_joint_angles[5]) * FACTOR)  # Force to 0
        joint_6 = round(0.2 * 1000 * 1000)

        piper.MotionCtrl_2(0x01, 0x01, 95, 0x00)
        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
        piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
        time.sleep(2)


def main():
    """Main control loop for the robot arm."""

    run_can_activation_script()

    mqtt_handler = MQTTHandler(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CLIENT_ID)
    mqtt_handler.connect()

    piper = C_PiperInterface("can0")
    piper.ConnectPort()

    initialize_robot(piper)

    # Load Pinocchio model and setup Meshcat
    model, collision_model, visual_model, data = load_pinocchio_model()
    viz = setup_meshcat_visualizer(model, collision_model, visual_model)

    # Initial pose and calibration
    current_end_pose = INITIAL_END_POSE[:]
    old_end_pose = INITIAL_END_POSE[:]
    calibration_pose = None
    last_joint_angles = np.zeros(4)  # Changed to reflect 4 active joints
    trigger_state = 0

    time.sleep(2)

    try:
        while True:
            # Get data from MQTT
            (
                endpose_delta,
                trigger,
                joystickX,
                joystickY,
                joystickClick,
                buttonA,
                buttonB,
                grip,
            ) = mqtt_handler.get_righthand()

            # Handle button A (restart)
            if buttonA == 1:
                print("Button A pressed. Restarting robot arm control...")
                mqtt_handler.disconnect()
                piper.DisconnectPort()
                mqtt_handler.connect()
                piper.ConnectPort()
                initialize_robot(piper)
                current_end_pose = INITIAL_END_POSE[:]
                old_end_pose = INITIAL_END_POSE[:]
                calibration_pose = None
                last_joint_angles = np.zeros(4)

            # Handle button B (disable)
            if buttonB == 1:
                print("Button B pressed. Disabling robot arm...")
                piper.DisableArm(7)

            # Process endpose data when available
            if endpose_delta is not None and trigger is not None:
                try:
                    endpose_delta = [float(x) for x in endpose_delta]
                    trigger = int(trigger)

                    if len(endpose_delta) != 6:
                        print(f"Error: Incorrect endpose length ({len(endpose_delta)}), expected 6.")
                        continue

                except (ValueError, TypeError) as e:
                    print(f"Error unpacking MQTT: {e}")
                    continue

                # Handle trigger state
                if trigger == 0:  # Trigger Released
                    if trigger_state == 1:
                        print("Trigger released. Returning to initial pose...")
                        move_to_initial_pose(piper)

                        current_end_pose = INITIAL_END_POSE[:]
                        old_end_pose = INITIAL_END_POSE[:]
                        calibration_pose = None
                    trigger_state = 0

                elif trigger == 1:  # Trigger Pressed
                    trigger_state = 1

                    # Calibrate on first trigger press
                    if calibration_pose is None:
                        calibration_pose = endpose_delta[:]
                        print("Trigger pressed. Calibrating to current pose as zero point...")

                    # Calculate deltas from calibrated pose
                    delta_x = endpose_delta[0] - calibration_pose[0]
                    delta_y = endpose_delta[1] - calibration_pose[1]
                    delta_z = endpose_delta[2] - calibration_pose[2]
                    delta_rx = endpose_delta[3] - calibration_pose[3]
                    delta_ry = endpose_delta[4] - calibration_pose[4]
                    delta_rz = endpose_delta[5] - calibration_pose[5]

                    if delta_z < -0.1:
                        delta_z = -0.1

                    new_end_pose = [
                        INITIAL_END_POSE[0] + delta_x,
                        INITIAL_END_POSE[1] + delta_y,
                        INITIAL_END_POSE[2] + delta_z,
                        INITIAL_END_POSE[3] + delta_rx,
                        INITIAL_END_POSE[4] + delta_ry,
                        INITIAL_END_POSE[5] + delta_rz,
                    ]

                    current_end_pose = new_end_pose[:]

                    # IK to find the joint angles, send angles to robot directly
                    target_pose = np.eye(4)
                    target_pose[:3, :3] = Rotation.from_euler("xyz", current_end_pose[3:]).as_matrix()
                    target_pose[:3, 3] = current_end_pose[:3]

                    joint_angles, success, elapsed_time = inverse_kinematics(
                        model, data, target_pose, last_joint_angles
                    )

                    if success:
                        last_joint_angles = joint_angles

                        # Convert to int and Send Motor signal
                        joint_0 = round(joint_angles[0] * FACTOR)
                        joint_1 = round(joint_angles[1] * FACTOR)
                        joint_2 = round(joint_angles[2] * FACTOR)
                        joint_3 = round(0 * FACTOR)  # Fixed Joint 3
                        joint_4 = round(joint_angles[3] * FACTOR)  # Joint 4 from result.x (remember, q_init is now 4 elements)
                        joint_5 = round(0 * FACTOR)  # Fixed Joint 5
                        joint_6 = round((1 - grip) / 5 * 1000 * 1000)

                        # Visualize the robot in Meshcat
                        joint_angles_full = np.zeros(model.nq)
                        joint_angles_full[0] = joint_angles[0]
                        joint_angles_full[1] = joint_angles[1]
                        joint_angles_full[2] = joint_angles[2]
                        joint_angles_full[3] = 0  # Fixed
                        joint_angles_full[4] = joint_angles[3]
                        joint_angles_full[5] = 0  # Fixed

                        viz.display(joint_angles_full)

                        piper.MotionCtrl_2(0x01, 0x01, 90, 0x00)
                        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                        piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

                    else:
                        print(f"IK Failed")
                    old_end_pose = current_end_pose[:]

            time.sleep(0.01)  # Add a small delay to avoid excessive CPU usage

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        mqtt_handler.disconnect()
        piper.DisconnectPort()


if __name__ == "__main__":
    main()