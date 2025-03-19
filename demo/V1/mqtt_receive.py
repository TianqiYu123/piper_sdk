import paho.mqtt.client as mqtt
import transforms3d.euler as euler
import numpy as np
import json
from scipy.spatial.transform import Rotation
from math import pi
from spatialmath import SE3
import roboticstoolbox as rtb 

import time
from piper_sdk import *

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

def left_to_right_hand(x_left, y_left, z_left, rx_left, ry_left, rz_left):
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

        if "info" in data and isinstance(data["info"], list) and len(data["info"]) == 15:
            x, y, z, qx, qy, qz, qw, trigger,joystickX,joystickY,joystickClick,buttonA,buttonB,grip,temp = data["info"]
            #print("trigger",trigger)
            # Convert quaternion to Euler angles (rx, ry, rz)
            # Note: The order of rotation is 'xyz' which is a common convention.
            rx, ry, rz = euler.quat2euler([qw, qx, qy, qz], 'sxyz')

            #endpose = [x, y, z, rx, ry, rz]
            endpose = left_to_right_hand(x, y, z, rx, ry, rz)
            return endpose, int(trigger), joystickX, joystickY, int(joystickClick), int(buttonA), int(buttonB), int(grip)
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
        L6 = rtb.RevoluteMDH(d= d6, a= a6, alpha= alpha6, offset= -72.5 * radian1, qlim= [lim6_min, lim6_max])

        # Create the serial chain (robot arm)
        self.robot = rtb.DHRobot([L1, L2, L3, L4, L5, L6], name="Arm")

        # Update qlim with user-provided values
        self.robot.qlim = [[-2.68780705,  0. ,        -3.05432619, -1.74532925, -1.30899694, -1.74532925],
                            [ 2.68780705,  3.40339204,  0.        ,  1.95476876,  1.30899694,  1.74532925]]

        self.last_successful_q = [0, 0, 0, 0, 0, 0]  # Initialize with a default initial guess


    
    def inverse_kinematics(self, Tep):
        """
        Calculates the inverse kinematics for the given end-effector pose, using the last
        successful joint angles as the initial guess.

        Args:
            endpos (list or tuple): End-effector pose [x, y, z, rx, ry, rz] in meters and radians.
                                     rx, ry, rz are Euler angles (RPY - Roll, Pitch, Yaw).

        Returns:
                        current_end_pose = INITIAL_END_POSE[:] #Reset to origin
                        calibration_pose = None # Clear Calibration
                    trigger_state = 0 #Reset Trigger state

        """
        start_time = time.time()  # Record the start time

        try:
            #print("endpos",endpos)
            #endpos_new = self.left_to_right_hand(endpos[0], endpos[1], endpos[2], endpos[3], endpos[4], endpos[5])

            #print("endpos2",endpos_new)
            #print("endpos_righthand",int(endpos[0]),int(endpos[1]),int(endpos[2]),endpos[3],endpos[4],endpos[5])
            #x, y, z, rx, ry, rz = endpos  # Unpack end-effector pose

            # Create the SE3 transformation matrix using RPY angles
            #Tep = SE3.Trans(x, y, z) * SE3.RPY(rx, ry, rz)  # important, you can chose RPY or Euler

            # Solve inverse kinematics using the last successful joint angles as the initial guess
            #sol = self.robot.ik_LM(Tep, q0=[0,0,0,0,0,0], ilimit=500, slimit=200, tol=1e-1, k=0.5)
            #sol = self.robot.ik_LM(Tep, q0=[0,0,0,0,0,0], ilimit=500, slimit=300, tol=1e-2, mask = [1,1,1,100,100,100],joint_limits = 1,k=0.5,method = 'chan')
            sol = self.robot.ik_LM(Tep, q0=self.last_successful_q, ilimit=500, slimit=300, tol=5e-3, mask = [1,1,1,100,100,100],joint_limits = 1,k=0.5,method = 'chan')
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
        self.joystickX = None
        self.joystickY = None
        self.joystickClick = None
        self.buttonA = None
        self.buttonB = None
        self.grip = None

    def on_connect(self, client, userdata, flags, rc):
        """Callback function when the MQTT client connects to the broker."""
        if rc == 0:
            print("Connected to MQTT Broker!")
            client.subscribe(self.topic)
        else:
            print(f"Failed to connect, return code {rc}")

    def on_message(self, client, userdata, msg):
        """Callback function when a message is received from the MQTT broker."""
        endpose, trigger,joystickX,joystickY,joystickClick,buttonA,buttonB,grip = process_message(msg.payload.decode())
        if endpose is not None:
            self.endpose = endpose
            self.trigger = trigger
            self.joystickX = joystickX
            self.joystickY = joystickY
            self.joystickClick = joystickClick
            self.buttonA = buttonA
            self.buttonB = buttonB
            self.grip = grip
            #print(f"Received endpose: {self.endpose}, Trigger: {self.trigger}")

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

    def get_righthand(self):
        """Returns the most recently received endpose and trigger.
        Returns (None, None) if no message has been received yet."""
        return self.endpose, self.trigger, self.joystickX, self.joystickY, self.joystickClick, self.buttonA, self.buttonB, self.grip