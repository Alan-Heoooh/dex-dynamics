import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
import yaml
import numpy as np
from transforms3d.quaternions import quat2mat  # For converting quaternion to rotation matrix


class TFListener(Node):
    def __init__(self):
        super().__init__('tf_listener_node')

        # Create the tf buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Timer to get the transform once and exit
        self.timer = self.create_timer(1.0, self.get_transform_once)

        self.target_frame = 'cam_0_depth_frame'
        self.source_frame = 'cam_0_depth_optical_frame'

    def get_transform_once(self):
        try:
            # Get the transform between frames
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,  # target frame
                self.source_frame,  # source frame
                rclpy.time.Time()  # latest transform available
            )

            # Extract translation
            translation = transform.transform.translation
            x, y, z = translation.x, translation.y, translation.z

            # Extract orientation (quaternion)
            rotation = transform.transform.rotation
            qw, qx, qy, qz = rotation.w, rotation.x, rotation.y, rotation.z  # Changed order to w, x, y, z

            # Convert quaternion (w, x, y, z) to 3x3 rotation matrix using transforms3d
            rot_matrix = quat2mat([qw, qx, qy, qz])

            # Build the 4x4 transformation matrix
            trans_matrix = np.identity(4)
            trans_matrix[0:3, 0:3] = rot_matrix  # Set rotation
            trans_matrix[0:3, 3] = [x, y, z]  # Set translation

            # Store data to a YAML file
            self.save_to_yaml(x, y, z, qw, qx, qy, qz, trans_matrix)

            self.get_logger().info("Transformation saved to transform_data.yml")

            # Shutdown the program after saving the file
            rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f'Could not get transform: {e}')
            rclpy.shutdown()  # Still shutdown even if there’s an error

    def save_to_yaml(self, x, y, z, qw, qx, qy, qz, matrix):
        data = {f'{self.source_frame}_to_{self.target_frame}': {
            'position': {
                'x': x,
                'y': y,
                'z': z
            },
            'orientation': {
                'w': qw,
                'x': qx,
                'y': qy,
                'z': qz
            },
            'transformation': matrix.tolist()  # Convert to list for YAML serialization
        }}

        with open('transform_data.yml', 'a') as file:
            yaml.dump(data, file, default_flow_style=False)


def main(args=None):
    rclpy.init(args=args)
    node = TFListener()
    rclpy.spin(node)  # Only spins until the transform is found and program shuts down


if __name__ == '__main__':
    main()
