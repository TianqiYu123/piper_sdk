#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
import numpy as np
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb 
import sys
import select
from mqtt_receive import MQTTHandler
from mqtt_receive import RobotArmIK
import time
from piper_sdk import *

MQTT_BROKER = "47.96.170.89"  #MQTT broker address
MQTT_PORT = 8003 # Default MQTT port.  If you are using a different port, change this.  8003 is typically a web port (HTTP).
MQTT_TOPIC = "arm/pose/#"  # MQTT topic you are subscribing to
MQTT_CLIENT_ID = "endpose_reader" #client ID

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

FACTOR = 57324.840764  # A constant, keep it uppercase.

# Number of interpolation points
N_SAMPLES = 10  # Adjust for smoother motion

# Initial end pose
INITIAL_END_POSE = [55, 0, 260,0,  np.pi/2, 0]

def main():
    mqtt_handler = MQTTHandler(MQTT_BROKER, MQTT_PORT, MQTT_TOPIC, MQTT_CLIENT_ID)
    mqtt_handler.connect()
    piper = C_PiperInterface("can0")  # Assuming can0 is constant, otherwise, make it configurable
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper) # Assuming enable_fun does not need endpose.

    arm_ik = RobotArmIK()

    current_end_pose = INITIAL_END_POSE[:]  # Create a copy
    old_end_pose = INITIAL_END_POSE[:]
    calibration_pose = None # will store the calibration snapshot

    trigger_state = 0 #Initial state
    #endpose_delta = None #initial delta -- Removed, since data is now parsed directly from MQTT message

    time.sleep(2)  # Give MQTT client time to connect and receive initial messages


    try:
        while True:
            endpose_delta, trigger, joystickX, joystickY, joystickClick, buttonA, buttonB, grip = mqtt_handler.get_righthand()

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

                if trigger == 0:  # If the trigger is released, return to the initial pose
                    if trigger_state == 1:
                        print("Trigger released. Returning to initial pose...")
                        T0 = SE3.Trans(*current_end_pose[:3]) * SE3.RPY(*current_end_pose[3:])
                        T1 = SE3.Trans(*INITIAL_END_POSE[:3]) * SE3.RPY(*INITIAL_END_POSE[3:])

                        trajectory = rtb.tools.trajectory.ctraj(T0, T1, N_SAMPLES)  # Generate smooth trajectory

                        for Tep in trajectory:
                            joint_angles, success, message, elapsed_time = arm_ik.inverse_kinematics(Tep)
                            if success:
                                joint_0 = round(joint_angles[0] * FACTOR)
                                joint_1 = round(joint_angles[1] * FACTOR)
                                joint_2 = round(joint_angles[2] * FACTOR)
                                joint_3 = round(joint_angles[3] * FACTOR)
                                joint_4 = round(joint_angles[4] * FACTOR)
                                #joint_5 = round((joint_angles[5] + 1.2653) * FACTOR)
                                joint_5 = round((joint_angles[5]) * FACTOR)
                                joint_6 = round(0.2 * 1000 * 1000)

                                piper.MotionCtrl_2(0x01, 0x01, 80, 0x00)
                                piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                                piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                                #time.sleep(0.001)  # Adjust for smooth motion

                        current_end_pose = INITIAL_END_POSE[:]  # Reset to origin
                        old_end_pose = INITIAL_END_POSE[:]
                        calibration_pose = None  # Clear calibration
                    trigger_state = 0  # Reset trigger state

                elif trigger == 1:  # If trigger is pressed
                    trigger_state = 1

                    if calibration_pose is None:
                        calibration_pose = endpose_delta[:]
                        print("Trigger pressed. Calibrating to current pose as zero point...")

                    # Compute new target pose
                    delta_x = (endpose_delta[0] - calibration_pose[0]) * 1000  # Convert to mm
                    delta_y = (endpose_delta[1] - calibration_pose[1]) * 1000
                    delta_z = (endpose_delta[2] - calibration_pose[2]) * 1000
                    delta_rx = (endpose_delta[3] - calibration_pose[3]) * 1  # In radians
                    delta_ry = (endpose_delta[4] - calibration_pose[4]) * 1
                    delta_rz = (endpose_delta[5] - calibration_pose[5]) * 1

                    new_end_pose = [
                        INITIAL_END_POSE[0] + delta_x,
                        INITIAL_END_POSE[1] + delta_y,
                        INITIAL_END_POSE[2] + delta_z,
                        INITIAL_END_POSE[3] + delta_rx,
                        INITIAL_END_POSE[4] + delta_ry,
                        INITIAL_END_POSE[5] + delta_rz,
                    ]

                    current_end_pose = new_end_pose[:]

                    # Compute trajectory
                    #T0 = SE3.Trans(*current_end_pose[:3]) * SE3.RPY(*current_end_pose[3:])
                    #T1 = SE3.Trans(*new_end_pose[:3]) * SE3.RPY(*new_end_pose[3:])
                    T0 = SE3.Trans(*old_end_pose[:3]) * SE3.RPY(*old_end_pose[3:])
                    T1 = SE3.Trans(*current_end_pose[:3]) * SE3.RPY(*current_end_pose[3:])
                    trajectory = rtb.tools.trajectory.ctraj(T0, T1, N_SAMPLES)

                    for Tep in trajectory:
                        joint_angles, success, message, elapsed_time = arm_ik.inverse_kinematics(Tep)
                        if success:
                            joint_0 = round(joint_angles[0] * FACTOR)
                            joint_1 = round(joint_angles[1] * FACTOR)
                            joint_2 = round(joint_angles[2] * FACTOR)
                            joint_3 = round(joint_angles[3] * FACTOR)
                            joint_4 = round(joint_angles[4] * FACTOR)
                            joint_5 = round((joint_angles[5]) * FACTOR)
                            #joint_5 = round((joint_angles[5] + 1.2653) * FACTOR)
                            joint_6 = round((1 - grip) / 5 * 1000 * 1000)

                            piper.MotionCtrl_2(0x01, 0x01, 80, 0x00)
                            piper.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)
                            piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                            #time.sleep(0.001)

                        else:
                            print(f"IK Failed message: {message}")
                    old_end_pose = current_end_pose[:]

            #time.sleep(0.001)

    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        mqtt_handler.disconnect()
        piper.DisconnectPort()

if __name__ == "__main__":
    main()

        
        
