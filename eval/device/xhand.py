import os
import time
import numpy as np
from xhand_controller import xhand_control
from device.robot_base import RobotBase

# Set LD_LIBRARY_PATH environment variable to include library directory for XHand controller
# This ensures the dynamic libraries can be found at runtime
script_dir = os.path.dirname(os.path.realpath(__file__))
xhandcontrol_library_dir = os.path.join(script_dir, "lib")
os.environ["LD_LIBRARY_PATH"] = (
    xhandcontrol_library_dir + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
)
print(f"LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}\n")


class XHand(RobotBase):
    def __init__(self, name="xhand", control_mode="position", serial_port="ttyUSB1", protocol="RS485", baud_rate=3000000, 
                 force_update=False, joint1_offset=1.4, apply_offset_by_default=True):
        """
        Initialize XHand robotic hand controller.
        
        Args:
            protocol: Communication protocol (currently only "RS485" supported)
            serial_port: Serial port for connection to the hand device
            baud_rate: Communication baud rate
            force_update: Whether to force state updates when reading
            joint1_offset: Calibration offset value for joint 1
            apply_offset_by_default: Whether to automatically apply joint1 offset
        """
        # Initialize the RobotBase class
        super().__init__(name=name, control_mode=control_mode)
        
        # XHand-specific initialization parameters
        self.force_update = force_update
        self.joint1_offset = joint1_offset  # Store the offset for joint 1
        self.apply_offset_by_default = apply_offset_by_default  # Whether to apply offset by default
        self._hand_id = None
        self._device = xhand_control.XHandControl()
        
        # Setup joint names and configuration
        self._setup_joint_names()
        
        # Setup joint movement limits
        # self._setup_joint_limits()
        
        # Connect to the device
        assert protocol in ["RS485"], "Unsupported protocol specified."
        self.enumerate_devices("RS485")
        device_identifier = {
            "protocol": protocol,
            "serial_port": f"/dev/{serial_port}",
            "baud_rate": baud_rate
        }
        self.open_device(device_identifier)

        # Retrieve the hand ID and device information
        self.list_hands_id()
        self.get_sdk_version()
        self.read_version()
        self.get_hand_type()
        self.get_serial_number()

        # Initialize the default hand command for 12 fingers with preset gains and values
        print("Initializing hand command...")
        self._hand_command = self._init_default_command()
        self.send_command(self._hand_command)
        time.sleep(1)
        
        # Initialize observation state
        self.init_obs = self.get_obs()
    
    def _setup_joint_names(self):
        """Setup joint names for the robotic hand"""
        joint_names = [f"xhand_joint_{i}" for i in range(12)]
        self.joint_names = {
            "all": joint_names,
            "fingers": joint_names
        }
    
    # Implementation of RobotBase required methods
    def set_action(self, action: np.ndarray) -> None:
        """
        Implement RobotBase's set_action method to control the robot's movement
        
        Args:
            action: (12,) array containing target position for each joint
        """
        assert action.shape == (12,), f"Expected action shape (12,), got {action.shape}"
        return self.set_hand_position(action)
    
    def clean_warning_error(self) -> None:
        """
        Clear warning and error states from the hand device
        Resets sensors to help clear error conditions
        """
        print("Cleaning warning and error states...")
        # Reset sensors to clear potential errors
        self.reset_sensor()
    
    def get_qpos(self) -> np.ndarray:
        """
        Get current joint positions
        
        Returns:
            (12,) array containing current position values for all joints
        """
        positions = self.get_hand_position()
        if positions is None:
            # Return zero positions if reading fails
            return np.zeros(12)
        return positions
    
    def get_obs(self) -> dict:
        """
        Get robot observation state
        
        Returns:
            Dictionary containing qpos (positions), qvel (velocities), qacc (accelerations)
            Note: velocities and accelerations are zeroed as XHand doesn't provide this data
        """
        qpos = self.get_qpos()
        # XHand doesn't provide velocity and acceleration data, so return zero arrays
        obs = {
            "qpos": qpos,
        }
        return obs
    
    def reset(self) -> None:
        """
        Reset robot to initial state
        Sets all joints to default positions and updates initial observation state
        """
        print("Resetting XHand to initial state...")
        # Set to initial position
        initial_positions = np.full(12, 0.1)  # Use default values from _init_position
        self.set_hand_position(initial_positions)
        time.sleep(1)  # Allow time for movement to complete
        # Save initial observation state
        self.init_obs = self.get_obs()
    
    def stop(self) -> None:
        """
        Stop all movement and close the device connection
        Sets hand to safe mode before closing
        """
        print("Stopping XHand...")
        self.set_hand_mode(0)  # Use safe mode
        self.close_device()
    
    @property
    def active_joint_names(self) -> list[str]:
        """
        Returns the names of active joints
        
        Returns:
            List of joint names
        """
        return self.joint_names["all"]
    
    @property
    def joint_limits(self) -> np.ndarray:
        """
        Returns the movement limits for all joints
        
        Returns:
            Array of joint limits with shape (12, 2) - [min, max] for each joint
        """
        return self._joint_limits

    # XHand-specific methods
    def _init_position(self):
        """
        Initialize default position values for all 12 joints
        
        Returns:
            A numpy array with 12 elements containing default position values for each joint
        """
        # Default position for all joints is 0.1
        positions = np.full(12, 0.1)
        
        # No offset is applied here as this returns the "true" desired positions
        # The offset application happens when these positions are used
        return positions
    
    def _init_default_command(self):
        """
        Initialize default command structure for all 12 joints
        Uses positions from _init_position() and applies necessary offsets
        
        Returns:
            A HandCommand_t structure with initialized values for all joints
        """
        command = xhand_control.HandCommand_t()
        
        # Get the default positions
        default_positions = self._init_position()
        
        # Create 12 FingerCommand_t entries if they don't exist
        for i in range(12):
            # If the list is not long enough, append a new instance
            if len(command.finger_command) <= i:
                command.finger_command.append(xhand_control.FingerCommand_t())
            cmd = command.finger_command[i]
            cmd.id = i
            cmd.kp = 100      # Proportional gain for PID control
            cmd.ki = 0        # Integral gain for PID control
            cmd.kd = 0        # Derivative gain for PID control
            
            # Apply position - with special handling for joint 1 offset
            if i == 1 and self.apply_offset_by_default:  # Joint 1 with offset
                cmd.position = default_positions[i] - self.joint1_offset  # Subtract offset for joint 1
                print(f"Applying initial offset to joint 1: {default_positions[i]} -> {cmd.position}")
            else:
                cmd.position = default_positions[i]  # Default position for other joints
            
            cmd.tor_max = 300  # Maximum torque
            cmd.mode = 3       # Control mode (3 = position control)
        return command

    def enumerate_devices(self, protocol: str):
        """
        Enumerate available devices for the specified protocol
        
        Args:
            protocol: Communication protocol ("RS485" or "EtherCAT")
            
        Returns:
            List of available devices for the protocol
        """
        devices = self._device.enumerate_devices(protocol)
        print(f"Available XHand devices for {protocol}: {devices}\n")
        return devices

    def open_device(self, device_identifier: dict):
        """
        Open connection to the device based on the protocol provided
        
        Args:
            device_identifier: Dictionary containing connection parameters
                For RS485: protocol, serial_port, baud_rate
                For EtherCAT: protocol only
        """
        protocol = device_identifier.get("protocol")
        if protocol == "RS485":
            baud_rate = int(device_identifier.get("baud_rate", 3000000))
            serial_port = device_identifier.get("serial_port")
            rsp = self._device.open_serial(serial_port, baud_rate)
            print(f"Opened RS485 device on {serial_port} at {baud_rate}: {rsp.error_code == 0}\n")
        elif protocol == "EtherCAT":
            devices = self.enumerate_devices("EtherCAT")
            if not devices:
                print("No EtherCAT devices found.\n")
                return
            rsp = self._device.open_ethercat(devices[0])
            print(f"Opened EtherCAT device: {rsp.error_code == 0}\n")
        else:
            print("Unsupported protocol specified.")

    def list_hands_id(self):
        """
        Retrieve available hand IDs and update the active hand ID
        
        Returns:
            List of available hand IDs
        """
        hands = self._device.list_hands_id()
        if hands:
            self._hand_id = hands[0]
            print(f"Current Hand ID: {self._hand_id}\n")
        else:
            print("No hand IDs found.\n")
        return hands

    def get_sdk_version(self):
        """
        Retrieve and print the SDK software version
        
        Returns:
            SDK version string
        """
        sdk_version = self._device.get_sdk_version()
        print(f"Software SDK Version: {sdk_version}\n")
        return sdk_version

    def read_version(self):
        """
        Read and print the version information of the hand device
        
        Returns:
            Hardware version information
        """
        joint_id = 0
        error_struct, version = self._device.read_version(self._hand_id, joint_id)
        print(f"Hardware SDK Version: {version}\n")
        return version

    def get_hand_type(self):
        """
        Retrieve and print the device's hand type
        
        Returns:
            Hand type identifier
        """
        error_struct, hand_type = self._device.get_hand_type(self._hand_id)
        print(f"Hand Type: {hand_type}\n")
        return hand_type

    def get_serial_number(self):
        """
        Retrieve and print the device's serial number
        
        Returns:
            Device serial number
        """
        error_struct, serial_number = self._device.get_serial_number(self._hand_id)
        print(f"Serial Number: {serial_number}\n")
        return serial_number

    def set_hand_position(self, positions, apply_offset=None):
        """
        Set positions for all 12 joints of the hand
        
        Args:
            positions: Array with 12 elements containing position values for each joint
            apply_offset: If True, subtract the offset from joint 1's position.
                         If None, use the default setting from initialization.
                
        Returns:
            The result of the send_command operation
        """
        # Validate input
        if not isinstance(positions, np.ndarray):
            positions = np.array(positions)
        
        if positions.size != 12:
            raise ValueError(f"Expected 12 position values, got {positions.size}")
        
        # Create a copy to avoid modifying the input array
        adjusted_positions = positions.copy()
        
        # Decide whether to apply the offset
        apply_offset = self.apply_offset_by_default if apply_offset is None else apply_offset
        
        # Apply offset correction for joint 1 (index 1) if requested
        if apply_offset:
            adjusted_positions[1] = positions[1] - self.joint1_offset
            print(f"Applying offset to joint 1: {positions[1]} -> {adjusted_positions[1]}")
        
        # Update positions in the command structure
        for i in range(12):
            self._hand_command.finger_command[i].position = float(adjusted_positions[i])
        
        # Send the command to the hand
        result = self.send_command(self._hand_command)
        return result

    def get_hand_position(self, apply_offset=None):
        """
        Get current position values for all 12 joints of the hand
        
        Args:
            apply_offset: If True, add the offset to joint 1's position.
                         If None, use the default setting from initialization.
        
        Returns:
            Array with 12 elements containing current position values of each joint.
            Returns None if there was an error reading the state.
        """
        # Read the current state of the hand
        error_struct, state = self._device.read_state(self._hand_id, False)
        if error_struct.error_code != 0:
            print(f"Error reading state: {error_struct.error_code}\n")
            return None
        
        # Create an array to store positions
        positions = np.zeros(12)
        
        # Extract position values from each finger state
        for i in range(12):
            if i < len(state.finger_state):
                positions[i] = state.finger_state[i].position
        
        # Decide whether to apply the offset
        apply_offset = self.apply_offset_by_default if apply_offset is None else apply_offset
        
        # Apply offset correction for joint 1 (index 1) if requested
        if apply_offset:
            positions[1] = positions[1] + self.joint1_offset
            print(f"Applying offset to joint 1: {positions[1] - self.joint1_offset} -> {positions[1]}")
        
        print(f"Current hand positions: {positions}\n")
        return positions

    def send_command(self, hand_command=None):
        """
        Send a hand command to the device
        
        Args:
            hand_command: Command structure to send, or None to use default command
            
        Returns:
            Result of the command operation
        """
        if hand_command is None:
            hand_command = self._hand_command
        result = self._device.send_command(self._hand_id, hand_command)
        print(f"Send Command Successful: {result.error_code == 0}\n")
        return result

    def set_hand_mode(self, mode: int):
        """
        Set the control mode for all fingers on the hand
        
        Args:
            mode: Control mode (0=idle/safe mode, 3=position control, etc.)
        """
        hand_mode = xhand_control.HandCommand_t()
        for i in range(12):
            if len(hand_mode.finger_command) <= i:
                hand_mode.finger_command.append(xhand_control.FingerCommand_t())
            cmd = hand_mode.finger_command[i]
            cmd.id = i
            cmd.kp = 0      # Zero gains for safe operation
            cmd.ki = 0
            cmd.kd = 0
            cmd.position = 0
            cmd.tor_max = 0  # Zero torque for safe operation
            cmd.mode = mode
        self._device.send_command(self._hand_id, hand_mode)
        time.sleep(1)
        print(f"Hand mode set to {mode}.\n")

    def close_device(self):
        """
        Close the connection to the device
        Sets the hand to safe mode before closing the connection
        """
        self.set_hand_mode(0)  # Set to safe mode before closing
        self._device.close_device()
        print("Device connection closed.\n")

    def reset_sensor(self, sensor_id=17):
        """
        Reset the sensor specified by sensor_id
        
        Args:
            sensor_id: ID of sensor to reset (default: 17)
            
        Returns:
            Result of the reset operation
        """
        result = self._device.reset_sensor(self._hand_id, sensor_id)
        print(f"Reset Sensor Successful: {result.error_code == 0}\n")
        return result