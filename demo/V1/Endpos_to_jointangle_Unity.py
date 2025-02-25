
import time
import numpy as np
from math import pi
from spatialmath import SE3
from scipy.spatial.transform import Rotation
import roboticstoolbox as rtb 
#https://github.com/petercorke/robotics-toolbox-python
#If use anaconda: 
#conda install conda-forge::roboticstoolbox-python


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
            x, y, z, rx, ry, rz = endpos_new  # Unpack end-effector pose


            # Create the SE3 transformation matrix using RPY angles
            Tep = SE3.Trans(x, y, z) * SE3.RPY(rx, ry, rz)  # important, you can chose RPY or Euler

            # Solve inverse kinematics using the last successful joint angles as the initial guess
            sol = self.robot.ik_LM(Tep, q0=self.last_successful_q, ilimit=100, slimit=100, tol=1e-2, k=0.5)
            theta = sol[0]  # Joint angles
            success = sol[1]  # Success flag (1 if successful, 0 otherwise)

            if success == 1:
                # Check if the IK solution is within the joint limits
                if np.all((self.robot.qlim[0] <= theta) & (theta <= self.robot.qlim[1])):
                    elapsed_time = time.time() - start_time  # Calculate the elapsed time
                    self.last_successful_q = theta # update last sucessful state
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





######################################################################################################
####################################### Example usage:################################################
######################################################################################################
if __name__ == "__main__":
    # Create an instance of the RobotArmIK class
    arm_ik = RobotArmIK()

    # Example end-effector poses (x, y, z, rx, ry, rz) - meters and radians
    end_poses = [
        [170.694303, 411.179277, -462.358694, 0, 0, 0],# Example values  
        [170, 411.179277, -462.358694, 0, 0, 0],  # Slightly different pose
        [170, 411.179277, -462, 0, 0, 0],  # Another pose
        #[170, 411, -462, 0, 0, 0]  # Another pose
    ]

    for end_pose in end_poses:
        # Calculate inverse kinematics
        joint_angles, success, message, elapsed_time = arm_ik.inverse_kinematics(end_pose)  # Capture elapsed_time

        # Print results
        print(f"End Pose: {end_pose}")
        print(f"Success: {success}")
        print(f"Message: {message}")
        print(f"Execution Time: {elapsed_time:.4f} seconds")  # Print the elapsed time, formatted to 4 decimal places

        if success:
            print(f"Joint Angles: {joint_angles}")

            # Verify the solution using forward kinematics
            T = arm_ik.robot.fkine(joint_angles)
            print("------------------check result by forward kinematics------------------")
            print(T)
        else:
            print("Inverse kinematics failed.")
        print("-------------------------------------")