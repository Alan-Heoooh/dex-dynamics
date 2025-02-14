import pyrealsense2 as rs
import numpy as np
import cv2
import yaml
from pupil_apriltags import Detector
from transforms3d.quaternions import mat2quat, quat2mat

# Define the size of the AprilTag in meters (e.g., 5cm = 0.05m)
TAG_SIZE = 0.172  

# Specify RealSense Camera Serial Number (Set this if multiple cameras are connected)
CAMERA_SERIAL_NUMBER = "102122060842"  # Replace with your camera's serial number
if CAMERA_SERIAL_NUMBER == "102422074156":
    camera_sn = "cam_0"
elif CAMERA_SERIAL_NUMBER == "337322071340":
    camera_sn = "cam_1"
elif CAMERA_SERIAL_NUMBER == "102122060842":
    camera_sn = "cam_3"
else:
    raise ValueError("Invalid camera serial number. Please set the correct serial number.")


transformation_data_path = '/home/coolbot/data/calib/transform_data.yml'
with open(transformation_data_path, 'r') as f:
    transformation_data = yaml.load(f, Loader=yaml.FullLoader)

# Initialize RealSense pipeline with specific serial number
pipeline = rs.pipeline()
config = rs.config()
config.enable_device(CAMERA_SERIAL_NUMBER)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
profile = pipeline.start(config)

# Get camera intrinsics
intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
camera_matrix = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                          [0, intrinsics.fy, intrinsics.ppy],
                          [0, 0, 1]])
                          
dist_coeffs = np.zeros(5)  # Assuming no distortion

# Initialize AprilTag detector
detector = Detector(families="tag36h11")

def invert_pose(R, t):
    """Compute the inverse transformation (camera w.r.t. tag)."""
    R_inv = R.T  # Transpose of rotation matrix is its inverse
    t_inv = -R_inv @ t  # Invert translation
    return R_inv, t_inv

# Store transformations over time
transformation_history = []

def convert_to_ros_coordinates(T_cam):
    """Convert the camera's position and rotation to ROS camera_link convention."""
    # optical_to_ros = np.array([
    #     [0.0, 0.0, 1.0, 0.0],
    #     [-1.0, 0.0, 0.0, 0.0],
    #     [0.0, -1.0, 0.0, 0.0],
    #     [0.0, 0.0, 0.0, 1.0],])

    color_frame_to_link = np.array(transformation_data[f'{camera_sn}_color_frame_to_{camera_sn}_link']['transformation'])
    color_optical_to_color_frame = np.array(transformation_data[f'{camera_sn}_color_optical_frame_to_{camera_sn}_color_frame']['transformation'])

    T_link = color_frame_to_link @ color_optical_to_color_frame @ T_cam    

    return T_link

try:
    while True:
        # Get frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags
        tags = detector.detect(gray, estimate_tag_pose=True, 
                               camera_params=(intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy),
                               tag_size=TAG_SIZE)

        for tag in tags:
            # Pose of the tag in the camera frame
            R_tag = np.array(tag.pose_R)
            t_tag = np.array(tag.pose_t).flatten()
            T_tag = np.eye(4)
            T_tag[:3, :3] = R_tag
            T_tag[:3, 3] = t_tag

            # Pose of the tag in the ros frame
            T_tag_ros = convert_to_ros_coordinates(T_tag)

            t_tag_ros, R_tag_ros = T_tag_ros[:3, 3], T_tag_ros[:3, :3]

            # Compute camera extrinsics (inverse transformation)
            # R_cam, t_cam = invert_pose(R_tag, t_tag)
            R_ros, t_ros = invert_pose(R_tag_ros, t_tag_ros)

            # Convert to ROS coordinates
            # t_ros, R_ros = convert_to_ros_coordinates(t_cam, R_cam)

            # Store transformation matrix
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = R_ros
            transformation_matrix[:3, 3] = t_ros

            transformation_history.append(transformation_matrix)

            # Print results in ROS frame
            print(f"\nTag ID: {tag.tag_id}")
            print(f"Translation (ROS):\n{t_ros}")
            print(f"Rotation (ROS):\n{R_ros}")
            print(f"Transformation Matrix (ROS):\n{transformation_matrix}")

            # Draw detected tag on image
            for idx in range(len(tag.corners)):
                pt1 = tuple(tag.corners[idx].astype(int))
                pt2 = tuple(tag.corners[(idx + 1) % len(tag.corners)].astype(int))
                cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)

            # Draw the tag center
            cX, cY = int(tag.center[0]), int(tag.center[1])
            cv2.circle(color_image, (cX, cY), 5, (0, 0, 255), -1)
            cv2.putText(color_image, f"ID: {tag.tag_id}", (cX - 10, cY - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Show the image
        cv2.imshow("AprilTag Detection", color_image)

        # Exit loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    if transformation_history:
        # Compute average transformation
        avg_transformation = np.mean(transformation_history, axis=0)

        # Extract position and orientation in ROS format
        position = avg_transformation[:3, 3].tolist()
        rotation_vector = mat2quat(avg_transformation[:3, :3])
        orientation = rotation_vector.flatten().tolist()

        # Convert to YAML format
        transformation_data = {
            f"cam_{CAMERA_SERIAL_NUMBER}": {
                "orientation": orientation,
                "position": position,
                "transformation": avg_transformation.tolist()
            }
        }

        # Save to a local YAML file
        yaml_filename = "camera_extrinsics_link.yml"
        with open(yaml_filename, "a") as yaml_file:
            yaml.dump(transformation_data, yaml_file, default_flow_style=False)

        print(f"\nAverage Transformation saved to {yaml_filename}")

    # Clean up
    pipeline.stop()
    cv2.destroyAllWindows()
