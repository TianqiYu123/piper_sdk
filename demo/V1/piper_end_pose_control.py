#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import numpy as np
from math import pi
from spatialmath import SE3
from scipy.spatial.transform import Rotation
import roboticstoolbox as rtb 
import sys
import select
from mqtt_receive import MQTTHandler

from typing import (
    Optional,
)
import time
from piper_sdk import *

MQTT_BROKER = "47.96.170.89"  #MQTT broker address
MQTT_PORT = 8003 # Default MQTT port.  If you are using a different port, change this.  8003 is typically a web port (HTTP).
MQTT_TOPIC = "arm/pose/#"  # MQTT topic you are subscribing to
MQTT_CLIENT_ID = "endpose_reader" #client ID


class RobotArmIK:
    """
    A class to perform inverse kinematics on a 6-DOF robot arm,
    using the last successful joint angles as the initial guess for the next calculation.
    """

    def __init__(self):
        """
        Initializes the RobotArmIK with DH parameters and robot model.
        """
        radian1 = pi / 180

        # DH Parameters and Joint Limits
        d1 = 123
        d2 = 0
        d3 = 0
        d4 = 250.75
        d5 = 0
        d6 = 91
        a1 = 0
        a2 = 0
        a3 = 285.03
        a4 = -21.98
        a5 = 0
        a6 = 0
        alpha1 = 0 * radian1
        alpha2 = -90 * radian1
        alpha3 = 0 * radian1
        alpha4 = 90 * radian1
        alpha5 = -90 * radian1
        alpha6 = 90 * radian1

        lim1_min = -154 * radian1
        lim1_max = 154 * radian1
        lim2_min = 0 * radian1
        lim2_max = 195 * radian1
        lim3_min = -175 * radian1
        lim3_max = 0 * radian1
        lim4_min = -100 * radian1
        lim4_max = 112 * radian1
        lim5_min = -75 * radian1
        lim5_max = 75 * radian1
        lim6_min = -100 * radian1
        lim6_max = 100 * radian1

        # DH Parameters for each joint
        L1 = rtb.RevoluteMDH(d= d1, a= a1, alpha= alpha1, qlim= [lim1_min, lim1_max])
        L2 = rtb.RevoluteMDH(d= d2, a= a2, alpha= alpha2, offset= -174.22 * radian1, qlim= [lim2_min, lim2_max])
        L3 = rtb.RevoluteMDH(d= d3, a= a3, alpha= alpha3, offset= -100.78 * radian1, qlim= [lim3_min, lim3_max])
        L4 = rtb.RevoluteMDH(d= d4, a= a4, alpha= alpha4, qlim= [lim4_min, lim4_max])
        L5 = rtb.RevoluteMDH(d= d5, a= a5, alpha= alpha5, qlim= [lim5_min, lim5_max])
        L6 = rtb.RevoluteMDH(d= d6, a= a6, alpha= alpha6, qlim= [lim6_min, lim6_max])

        # Create the serial chain (robot arm)
        self.robot = rtb.DHRobot([L1, L2, L3, L4, L5, L6], name="Arm")

        # Update qlim with user-provided values
        self.robot.qlim = [[-2.68780705,  0. ,        -3.05432619, -1.74532925, -1.30899694, -1.74532925],
                            [ 2.68780705,  3.40339204,  0.        ,  1.95476876,  1.30899694,  1.74532925]]

        self.last_successful_q = [0, 0, 0, 0, 0, 0]  # Initialize with a default initial guess


    def left_to_right_hand(self, x_left, y_left, z_left, rx_left, ry_left, rz_left):
        """
        Converts a pose (x, y, z, rx, ry, rz) from a left-handed coordinate system to a right-handed
        coordinate system using a custom coordinate axis transformation, while eliminating the initial
        rotation introduced by the coordinate axis transformation itself.

        Args:
            x_left, y_left, z_left: Position coordinates in the left-handed coordinate system.
            rx_left, ry_left, rz_left: Euler angles (Roll, Pitch, Yaw) in radians in the left-handed
                                        coordinate system (ZYX order).

        Returns:
            x_right, y_right, z_right, rx_right, ry_right, rz_right:
            The pose (position and Euler angles) in the right-handed coordinate system.
        """

        # Position transformation
        x_right = z_left
        y_right = -x_left
        z_right = y_left

        # Rotation matrix transformation
        r_left = Rotation.from_euler('zyx', [rz_left, ry_left, rx_left])  # Note: scipy's Euler angle order (ZYX)
        rot_matrix_left = r_left.as_matrix()

        # Direction transformation matrix
        transform_matrix = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, 1, 0]
        ])

        # Inverse transformation matrix, to eliminate the initial rotation
        transform_matrix_inv = transform_matrix.T  # The inverse of an orthogonal matrix equals its transpose

        rot_matrix_right = transform_matrix @ rot_matrix_left @ transform_matrix_inv  # Inverse transform, rotation, forward transform
        r_right = Rotation.from_matrix(rot_matrix_right)
        rz_right, ry_right, rx_right = r_right.as_euler('zyx')  # Note Euler angle order
        return [x_right, y_right, z_right, rx_right, ry_right, rz_right]
    
    
    def inverse_kinematics(self, endpos):
        """
        Calculates the inverse kinematics for the given end-effector pose, using the last
        successful joint angles as the initial guess.

        Args:
            endpos (list or tuple): End-effector pose [x, y, z, rx, ry, rz] in meters and radians.
                                     rx, ry, rz are Euler angles (RPY - Roll, Pitch, Yaw).

        Returns:
            tuple: A tuple containing:
                - theta (numpy.ndarray): Joint angles if successful, None otherwise.
                - success (bool): True if IK solution is found, False otherwise.
                - message (str): A message indicating the result of the IK process.
                - elapsed_time (float): The time taken for the IK calculation in seconds.
        """
        start_time = time.time()  # Record the start time

        try:
            #print("endpos",endpos)
            endpos_new = self.left_to_right_hand(endpos[0], endpos[1], endpos[2], endpos[3], endpos[4], endpos[5])

            #print("endpos2",endpos_new)
            print("endpos",endpos)
            #x, y, z, rx, ry, rz = endpos  # Unpack end-effector pose
            x, y, z, rx, ry, rz = endpos_new  # Unpack end-effector pose

            # Create the SE3 transformation matrix using RPY angles
            Tep = SE3.Trans(x, y, z) * SE3.RPY(rx, ry, rz)  # important, you can chose RPY or Euler

            # Solve inverse kinematics using the last successful joint angles as the initial guess
            #sol = self.robot.ik_LM(Tep, q0=self.last_successful_q, ilimit=100, slimit=100, tol=1e-2, k=0.5)
            sol = self.robot.ik_LM(Tep, q0=[0,0,0,0,0,0], ilimit=100, slimit=100, tol=1e-3, k=0.5)
            theta = sol[0]  # Joint angles
            success = sol[1]  # Success flag (1 if successful, 0 otherwise)

            if success == 1:
                # Check if the IK solution is within the joint limits
                if np.all((self.robot.qlim[0] <= theta) & (theta <= self.robot.qlim[1])):
                    elapsed_time = time.time() - start_time  # Calculate the elapsed time
                    self.last_successful_q = theta # update last sucessful statec
                    #self.last_successful_q = [0,0,0,0,0,0]
                    return theta, True, "IK solution found within joint limits.", elapsed_time
                else:
                    elapsed_time = time.time() - start_time  # Calculate the elapsed time
                    return None, False, "IK solution found, but violates joint limits.", elapsed_time
            else:
                elapsed_time = time.time() - start_time  # Calculate the elapsed time
                return None, False, "IK solver failed to converge.", elapsed_time

        except Exception as e:
            elapsed_time = time.time() - start_time  # Calculate the elapsed time even if an exception occurs
            return None, False, f"An error occurred: {e}", elapsed_time

def enable_fun(piper:C_PiperInterface):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
            piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)

def is_data_available():
    """Check if there's data waiting to be read from stdin (keyboard)."""
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

if __name__ == "__main__":
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    #piper.GripperCtrl(0,1000,0x01, 0)
    factor = 57324.840764

    arm_ik = RobotArmIK()

    # Initial end-effector pose
    x = 0.0
    y = 260.0
    z = 55.0
    rx = np.pi / 2
    ry = 0.0
    rz = 0.0

    step_size_pos = 10  # Adjustment step for position (x, y, z)
    step_size_rot = np.pi / 36  # Adjustment step for rotation (rx, ry, rz) - 5 degrees

    print("\nControl the end-effector pose:")
    print("  q/a: Adjust X ({}). Limits: -650 to 650".format(x))
    print("  w/s: Adjust Y ({}). Limits: -650 to 650".format(y))
    print("  e/d: Adjust Z ({}). Limits: -650 to 650".format(z))
    print("  r/f: Adjust Rx ({}). Limits: -pi to pi".format(rx))
    print("  t/g: Adjust Ry ({}). Limits: -pi to pi".format(ry))
    print("  y/h: Adjust Rz ({}). Limits: -pi to pi".format(rz))
    print("  Press Ctrl+C to exit.")

    try:
        while True:
            if is_data_available():
                key = sys.stdin.read(1).lower()  # Read one character

                if key == 'q':
                    x += step_size_pos
                    x = min(max(x, -650), 650)
                elif key == 'a':
                    x -= step_size_pos
                    x = min(max(x, -650), 650)
                elif key == 'w':
                    y += step_size_pos
                    y = min(max(y, -650), 650)
                elif key == 's':
                    y -= step_size_pos
                    y = min(max(y, -650), 650)
                elif key == 'e':
                    z += step_size_pos
                    z = min(max(z, -650), 650)
                elif key == 'd':
                    z -= step_size_pos
                    z = min(max(z, -650), 650)
                elif key == 'r':
                    rx += step_size_rot
                    rx = min(max(rx, -np.pi), np.pi)
                elif key == 'f':
                    rx -= step_size_rot
                    rx = min(max(rx, -np.pi), np.pi)
                elif key == 't':
                    ry += step_size_rot
                    ry = min(max(ry, -np.pi), np.pi)
                elif key == 'g':
                    ry -= step_size_rot
                    ry = min(max(ry, -np.pi), np.pi)
                elif key == 'y':
                    rz += step_size_rot
                    rz = min(max(rz, -np.pi), np.pi)
                elif key == 'h':
                    rz -= step_size_rot
                    rz = min(max(rz, -np.pi), np.pi)
                elif ord(key) == 3:  # Check for Ctrl+C (ASCII code 3)
                    raise KeyboardInterrupt
                else:
                    print("Invalid command.")
                    continue

                # Update the end_pose
                end_pose = [x, y, z, rx, ry, rz]

                # Calculate inverse kinematics
                joint_angles, success, message, elapsed_time = arm_ik.inverse_kinematics(end_pose)  # Capture elapsed_time

                if not success or joint_angles is None:  # Check for IK failure
                    print(f"IK Failed. Skipping this pose.")
                    print(f"Message: {message}")
                    continue # Skip the rest of the loop

                print(f"End Pose: {end_pose}")
                print(f"Success: {success}")
                print(f"Message: {message}")
                print(f"Execution Time: {elapsed_time:.4f} seconds")

                joint_0 = round(joint_angles[0]*factor)
                joint_1 = round(joint_angles[1]*factor)
                joint_2 = round(joint_angles[2]*factor)
                joint_3 = round(joint_angles[3]*factor)
                joint_4 = round(joint_angles[4]*factor)
                joint_5 = round(joint_angles[5]*factor)

                # piper.MotionCtrl_1()
                piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
                piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                #piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                time.sleep(0.005)

            else:
                time.sleep(0.01)  # Small delay to avoid busy-waiting

    except KeyboardInterrupt:
        print("\nExiting program.")
'''
if __name__ == "__main__":
    piper = C_PiperInterface("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    #piper.GripperCtrl(0,1000,0x01, 0)
    factor = 57324.840764

    arm_ik = RobotArmIK()

    # Example end-effector poses (x, y, z, rx, ry, rz) - meters and radians
    end_pose = [0, 260, 55, np.pi/2, 0, 0]#initial pose
    #end_pose = [-322, -67.7, 347, np.pi/2, 0, 0]

    # Calculate inverse kinematics
    joint_angles, success, message, elapsed_time = arm_ik.inverse_kinematics(end_pose)  # Capture elapsed_time
    print(f"End Pose: {end_pose}")
    print(f"Success: {success}")
    print(f"Message: {message}")
    print(f"Execution Time: {elapsed_time:.4f} seconds")  # Print the elapsed time, formatted to 4 decimal places

    joint_0 = round(joint_angles[0]*factor)
    joint_1 = round(joint_angles[1]*factor)
    joint_2 = round(joint_angles[2]*factor)
    joint_3 = round(joint_angles[3]*factor)
    joint_4 = round(joint_angles[4]*factor)
    joint_5 = round(joint_angles[5]*factor)

    # piper.MotionCtrl_1()
    piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
    piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
    #piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    time.sleep(0.005)
    pass
'''
        
