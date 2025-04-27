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

from pinocchio.visualize import MeshcatVisualizer

MQTT_BROKER = "47.96.170.89"
MQTT_PORT = 8003
MQTT_TOPIC = "arm/pose/#"
MQTT_CLIENT_ID = "endpose_reader"

#FACTOR = 57324.840764
FACTOR = 57290
#INITIAL_END_POSE = [150, 0, 260, 0, pi/2, 0]
INITIAL_END_POSE = [0, 0, 0, 0, pi / 2, 0]
NUM_JOINTS = 6  # number of joints you are controlling


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


def left_to_right_hand(x_left, y_left, z_left, rx_left, ry_left, rz_left):
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
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    current_pose = data.oMf[model.getFrameId("gripper_base")].homogeneous  # change tool_frame to your frame name.

    # Calculate position error
    position_error = np.linalg.norm(current_pose[:3, 3] - target_pose[:3, 3])

    # Calculate orientation error (using rotation matrix difference)
    rotation_error = np.linalg.norm(current_pose[:3, :3] - target_pose[:3, :3])
    # print("Error",position_error + rotation_error)
    return position_error + rotation_error


def inverse_kinematics(model, data, target_pose, q_init=None):
    """Inverse kinematics using optimization."""
    if q_init is None:
        q_init = np.zeros(model.nq)  # Initial guess for joint angles

    # Optimization bounds (joint limits) - Optional, but highly recommended
    bounds = [(model.lowerPositionLimit[i], model.upperPositionLimit[i]) for i in range(model.nq)]

    start_time = time.time()

    result = minimize(
        cost_function, q_init, args=(model, data, target_pose), method="SLSQP", bounds=bounds, tol=1e-4
    )  # tol is tolerance
    elapsed_time = time.time() - start_time
    if result.success:
        return result.x, True, elapsed_time
    else:
        return None, False, elapsed_time


def main():
    mqtt_handler = MQTTHandler(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CLIENT_ID)
    mqtt_handler.connect()
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
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
    time.sleep(2)  # Wait for the robot to reach the initial pose

    acc_limit = 270  # Define the acceleration limit
    piper.JointConfig(joint_num=7, set_zero=0, acc_param_is_effective=0xAE, max_joint_acc=acc_limit, clear_err=0xAE)
    time.sleep(1)

    '''
    piper.JointMaxAccConfig(1, acc_limit)
    time.sleep(0.05)
    piper.JointMaxAccConfig(2, acc_limit)
    time.sleep(0.05)
    piper.JointMaxAccConfig(3, acc_limit)
    time.sleep(0.05)
    piper.JointMaxAccConfig(4, acc_limit)
    time.sleep(0.05)
    piper.JointMaxAccConfig(5, acc_limit)
    time.sleep(0.05)
    piper.JointMaxAccConfig(6, acc_limit)
    time.sleep(0.05)
    '''

    # Load the robot model using pinocchio
    pinocchio_model_dir = join(dirname(dirname(str(abspath(__file__)))), "PICO")  # Adjust path as necessary
    #model_path = join(pinocchio_model_dir, "/")
    model_path = pinocchio_model_dir
    #/home/dm/robot_arm/piper_sdk_PICO/demo/PICO/piper_description/urdf/piper_description.urdf
    mesh_dir = pinocchio_model_dir
    urdf_filename = "piper_description.urdf"  # Use your robot's URDF
    urdf_model_path = join(join(model_path, "piper_description/urdf"), urdf_filename)
    model, collision_model, visual_model = pin.buildModelsFromUrdf(urdf_model_path, mesh_dir)
    data = model.createData()

    # Initialize the Meshcat visualizer
    viz = MeshcatVisualizer(model, collision_model, visual_model)
    try:
        viz.initViewer(open=True)
    except ImportError as err:
        print("Error while initializing the viewer. It seems you should install Python meshcat")
        print(err)
        sys.exit(0)
    viz.loadViewerModel()

    current_end_pose = INITIAL_END_POSE[:]
    old_end_pose = INITIAL_END_POSE[:]
    calibration_pose = None
    last_joint_angles = np.zeros(model.nq)

    trigger_state = 0

    time.sleep(2)

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
            ) = mqtt_handler.get_righthand()

            if buttonA == 1:
                print("Button A pressed. Restarting robot arm control...")
                # Reinitialize the robot arm control
                mqtt_handler.disconnect()
                piper.DisconnectPort()
                mqtt_handler.connect()
                piper.ConnectPort()
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
                time.sleep(2)  # Wait for the robot to reach the initial pose

                acc_limit = 270  # Define the acceleration limit
                piper.JointConfig(joint_num=7, set_zero=0, acc_param_is_effective=0xAE, max_joint_acc=acc_limit, clear_err=0xAE)
                time.sleep(1)

                current_end_pose = INITIAL_END_POSE[:]
                old_end_pose = INITIAL_END_POSE[:]
                calibration_pose = None
                last_joint_angles = np.zeros(model.nq)

            if buttonB == 1:
                print("Button B pressed. Disabling robot arm...")
                piper.DisableArm(7)

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

                if trigger == 0:
                    if trigger_state == 1:
                        print("Trigger released. Returning to initial pose...")
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

                        current_end_pose = INITIAL_END_POSE[:]
                        old_end_pose = INITIAL_END_POSE[:]
                        calibration_pose = None
                    trigger_state = 0
                    if trigger_state == 1:
                        print("Trigger released. Returning to initial pose...")
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

                        current_end_pose = INITIAL_END_POSE[:]
                        old_end_pose = INITIAL_END_POSE[:]
                        calibration_pose = None
                    trigger_state = 0

                elif trigger == 1:
                    trigger_state = 1

                    if calibration_pose is None:
                        calibration_pose = endpose_delta[:]
                        print("Trigger pressed. Calibrating to current pose as zero point...")

                    delta_x = endpose_delta[0] - calibration_pose[0]  # * 1000
                    delta_y = endpose_delta[1] - calibration_pose[1]  # * 1000
                    delta_z = endpose_delta[2] - calibration_pose[2]  # * 1000
                    delta_rx = endpose_delta[3] - calibration_pose[3]  # * 1
                    delta_ry = endpose_delta[4] - calibration_pose[4]  # * 1
                    delta_rz = endpose_delta[5] - calibration_pose[5]  # * 1

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
                        joint_3 = round(joint_angles[3] * FACTOR)
                        joint_4 = round(joint_angles[4] * FACTOR)
                        joint_5 = round((joint_angles[5]) * FACTOR)  # Joint Angle is not always = 0
                        joint_6 = round((1 - grip) / 5 * 1000 * 1000)

                        # Visualize the robot in Meshcat
                        viz.display(joint_angles)

                        # piper.SearchAllMotorMaxAccLimit()
                        # print(piper.GetAllMotorMaxAccLimit())
                        piper.MotionCtrl_2(0x01, 0x01, 90, 0x00)
                        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                        piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

                    else:
                        print(f"IK Failed")
                    old_end_pose = current_end_pose[:]

            time.sleep(0.01) # Add a small delay to avoid excessive CPU usage

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        mqtt_handler.disconnect()
        piper.DisconnectPort()


if __name__ == "__main__":
    main()
