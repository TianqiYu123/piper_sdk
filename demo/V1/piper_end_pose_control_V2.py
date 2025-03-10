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
            print("endpos_new",endpos_new)
            #x, y, z, rx, ry, rz = endpos  # Unpack end-effector pose
            x, y, z, rx, ry, rz = endpos_new  # Unpack end-effector pose

            # Create the SE3 transformation matrix using RPY angles
            Tep = SE3.Trans(x, y, z) * SE3.RPY(rx, ry, rz)  # important, you can chose RPY or Euler

            # Solve inverse kinematics using the last successful joint angles as the initial guess
            #sol = self.robot.ik_LM(Tep, q0=self.last_successful_q, ilimit=100, slimit=100, tol=1e-2, k=0.5)
            #sol = self.robot.ik_LM(Tep, q0=[0,0,0,0,0,0], ilimit=500, slimit=200, tol=1e-1, k=0.5)
            sol = self.robot.ik_LM(Tep, q0=[0,0,0,0,0,0], ilimit=500, slimit=300, tol=1e-1, mask = [1,1,1,0.6,0.6,0.6],joint_limits = 1,k=0.5,method = 'chan')
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

#FACTOR = 57324.840764  # A constant, keep it uppercase.
FACTOR = 1000
# Initial end pose
INITIAL_END_POSE = [0, 260, 55, np.pi/2, 0, 0]  # x, y, z, rx, ry, rz (meters and radians)

#Calibration trigger State
TRIGGER_INACTIVE = 0
TRIGGER_ACTIVE = 1

def main():
    mqtt_handler = MQTTHandler(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CLIENT_ID)
    mqtt_handler.connect()
    piper = C_PiperInterface("can0")  # Assuming can0 is constant, otherwise, make it configurable
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper) # Assuming enable_fun does not need endpose.

    arm_ik = RobotArmIK()

    current_end_pose = INITIAL_END_POSE[:]  # Create a copy
    calibration_pose = None # will store the calibration snapshot

    trigger_state = TRIGGER_INACTIVE #Initial state
    #endpose_delta = None #initial delta -- Removed, since data is now parsed directly from MQTT message

    time.sleep(2)  # Give MQTT client time to connect and receive initial messages

    try:
        while True:
            #get data from MQTT
            endpose_delta, endpose_trigger = mqtt_handler.get_endpose()
            # Assuming get_endpose function returns 7 values, with index 6 being the trigger.
            if endpose_delta is not None and endpose_trigger is not None: #check validity first
                try:
                    #endpose_delta = data[:6] #first 6 is delta pose -- Removed
                    #endpose_trigger = int(data[6]) #trigger is the last value -- Removed
                    endpose_delta = [float(x) for x in endpose_delta] # cast to float
                    endpose_trigger = int(endpose_trigger) # cast to int

                    if len(endpose_delta) != 6:
                        print(f"Error: Received delta endpose with incorrect length ({len(endpose_delta)}). Expected 6.")
                        continue #skip this iteration

                except (ValueError, TypeError) as e:
                    print(f"Error unpacking/converting MQTT {e}")
                    continue #skip this iteration

                #print(f"Raw endpose_delta: {endpose_delta}, endpose_trigger: {endpose_trigger}")  # Debug Print

                if endpose_trigger == TRIGGER_INACTIVE: #Trigger is released.
                    if trigger_state == TRIGGER_ACTIVE:  #Was the robot in motion before releasing trigger?
                        print("Trigger released. Returning to initial pose...")
                        # Move back to the initial pose
                        endpos_righthand = arm_ik.left_to_right_hand(INITIAL_END_POSE[0], INITIAL_END_POSE[1], INITIAL_END_POSE[2], INITIAL_END_POSE[3], INITIAL_END_POSE[4], INITIAL_END_POSE[5])
                        X = round(endpos_righthand[0]*FACTOR)
                        Y = round(endpos_righthand[1]*FACTOR)
                        Z = round(endpos_righthand[2]*FACTOR)
                        RX = round(endpos_righthand[3]*FACTOR*180/np.pi)
                        RY = round(endpos_righthand[4]*FACTOR*180/np.pi)
                        RZ = round(endpos_righthand[5]*FACTOR*180/np.pi)
                        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                        piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
                        time.sleep(0.01)
                        '''
                        joint_angles, success, message, elapsed_time = arm_ik.inverse_kinematics(INITIAL_END_POSE)

                        if success:
                            joint_0 = round(joint_angles[0] * FACTOR)
                            joint_1 = round(joint_angles[1] * FACTOR)
                            joint_2 = round(joint_angles[2] * FACTOR)
                            joint_3 = round(joint_angles[3] * FACTOR)
                            joint_4 = round(joint_angles[4] * FACTOR)
                            joint_5 = round(joint_angles[5] * FACTOR)
                            print(f"return to initial pose: {joint_0, joint_1, joint_2, joint_3, joint_4, joint_5}") 
                            piper.MotionCtrl_2(0x01, 0x01, 80, 0x00)
                            piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                            time.sleep(0.005)
                        else:
                            print(f"Failed to return to initial pose: {message}") #failure message
                        '''

                        current_end_pose = INITIAL_END_POSE[:] #Reset to origin
                        calibration_pose = None # Clear Calibration
                    trigger_state = TRIGGER_INACTIVE #Reset Trigger state


                elif endpose_trigger == TRIGGER_ACTIVE: #Trigger is pressed.
                    trigger_state = TRIGGER_ACTIVE  #set it to active, so when release it can go back to origin

                    if calibration_pose is None: #calibration hasn't happened yet, get the end pose

                        #Here, you'll grab endpose_delta upon first pressing trigger, so will calibrate the machine from this location.
                        #After setting calibration_pose = endpose_delta, calculate motion as (next endpose - previous endpose) + initial pose.
                        calibration_pose = endpose_delta[:] # Take snapshot from first measurement

                        print("Trigger pressed. Calibrating to current pose as zero point...")


                    # Calculate delta from calibration pose
                    delta_x = (endpose_delta[0] - calibration_pose[0])*1000 #mm
                    delta_y = (endpose_delta[1] - calibration_pose[1])*1000 #mm
                    delta_z = (endpose_delta[2] - calibration_pose[2])*1000 #mm
                    delta_rx = (endpose_delta[3] - calibration_pose[3])*1 #rad
                    delta_ry = (endpose_delta[4] - calibration_pose[4])*1 #rad
                    delta_rz = (endpose_delta[5] - calibration_pose[5])*1 #rad


                    # Apply the calibrated delta to the INITIAL END POSE.
                    new_end_pose = [INITIAL_END_POSE[0] + delta_x,
                                    INITIAL_END_POSE[1] + delta_y,
                                    INITIAL_END_POSE[2] + delta_z,
                                    INITIAL_END_POSE[3] + delta_rx,
                                    INITIAL_END_POSE[4] + delta_ry,
                                    INITIAL_END_POSE[5] + delta_rz]

                    current_end_pose = new_end_pose[:] #copy to memory

                    print(f"Received delta from MQTT: {endpose_delta}")
                    print(f"Calculated New endpose: {current_end_pose}")

                    endpos_righthand = arm_ik.left_to_right_hand(current_end_pose[0], current_end_pose[1], current_end_pose[2], current_end_pose[3], current_end_pose[4], current_end_pose[5])
                    X = round(endpos_righthand[0]*FACTOR)
                    Y = round(endpos_righthand[1]*FACTOR)
                    Z = round(endpos_righthand[2]*FACTOR)
                    RX = round(endpos_righthand[3]*FACTOR*180/np.pi)
                    RY = round(endpos_righthand[4]*FACTOR*180/np.pi)
                    RZ = round(endpos_righthand[5]*FACTOR*180/np.pi)
                    print(X,Y,Z,RX,RY,RZ)
                    # piper.MotionCtrl_1()
                    piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
                    piper.EndPoseCtrl(X,Y,Z,RX,RY,RZ)
                    #piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                    time.sleep(0.01)

                    '''
                    # Calculate inverse kinematics
                    joint_angles, success, message, elapsed_time = arm_ik.inverse_kinematics(current_end_pose)

                    if success:
                        joint_0 = round(joint_angles[0] * FACTOR)
                        joint_1 = round(joint_angles[1] * FACTOR)
                        joint_2 = round(joint_angles[2] * FACTOR)
                        joint_3 = round(joint_angles[3] * FACTOR)
                        joint_4 = round(joint_angles[4] * FACTOR)
                        joint_5 = round(joint_angles[5] * FACTOR)
                        piper.MotionCtrl_2(0x01, 0x01, 99, 0x00)
                        piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                        time.sleep(0.001)
                    else:
                        print(f"IK Failed message: {message}")'
                    '''

                else:
                    print(f"Invalid endpose_trigger value: {endpose_trigger}. Expected 0 or 1.")

            else:
                print("No data received from MQTT yet.") # data here means that endpose and trigger are not none, so if here, these value are not none
            time.sleep(0.001) # Adjust the sleep time as needed

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        mqtt_handler.disconnect()
        piper.DisconnectPort()

if __name__ == "__main__":
    main()

        
        
