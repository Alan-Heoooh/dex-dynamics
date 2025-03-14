import os
import sys
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer


class ImageSubscriberNode(Node):
    NUM_CAMERAS = 4

    def __init__(self, camera_names, save_dir):
        super().__init__('image_subscriber_node')
        assert len(camera_names) == self.NUM_CAMERAS, f"Does not support cameras more than {self.NUM_CAMERAS}"

        self.camera_names = camera_names
        self.save_dir = save_dir
        self.counters = {camera_name: 0 for camera_name in camera_names}  # Initialize counters

        for camera_name in camera_names:
            os.makedirs(f'{save_dir}/{camera_name}/color', exist_ok=True)
            os.makedirs(f'{save_dir}/{camera_name}/depth', exist_ok=True)

        self.cv_bridge = CvBridge()
        self.subscriber1_rgb = Subscriber(self, Image, f'/camera/{camera_names[0]}/color/image_raw')
        self.subscriber2_rgb = Subscriber(self, Image, f'/camera/{camera_names[1]}/color/image_raw')
        self.subscriber3_rgb = Subscriber(self, Image, f'/camera/{camera_names[2]}/color/image_raw')
        self.subscriber4_rgb = Subscriber(self, Image, f'/camera/{camera_names[3]}/color/image_raw')


        self.subscriber1_depth = Subscriber(self, Image, f'/camera/{camera_names[0]}/aligned_depth_to_color/image_raw')
        self.subscriber2_depth = Subscriber(self, Image, f'/camera/{camera_names[1]}/aligned_depth_to_color/image_raw')
        self.subscriber3_depth = Subscriber(self, Image, f'/camera/{camera_names[2]}/aligned_depth_to_color/image_raw')
        self.subscriber4_depth = Subscriber(self, Image, f'/camera/{camera_names[3]}/aligned_depth_to_color/image_raw')


        # self.subscriber1_points = Subscriber(self, PointCloud2, f'/camera/{camera_names[0]}/depth/color/points')
        # self.subscriber2_points = Subscriber(self, PointCloud2, f'/camera/{camera_names[1]}/depth/color/points')
        # self.subscriber3_points = Subscriber(self, PointCloud2, f'/camera/{camera_names[2]}/depth/color/points')

        self.synchronizer = ApproximateTimeSynchronizer(
            [self.subscriber1_rgb, self.subscriber2_rgb, self.subscriber3_rgb, self.subscriber4_rgb,
             self.subscriber1_depth, self.subscriber2_depth, self.subscriber3_depth, self.subscriber4_depth,
             # self.subscriber1_points, self.subscriber2_points, self.subscriber3_points
             ],
            queue_size=100, slop=0.1)
        self.synchronizer.registerCallback(self.callback)

    def rgb_msg_to_numpy(self, img):
        return self.cv_bridge.imgmsg_to_cv2(img, desired_encoding='bgr8')

    def depth_msg_to_numpy(self, dimg):
        return self.cv_bridge.imgmsg_to_cv2(dimg, desired_encoding='passthrough')

    def callback(self, img1, img2, img3, img4, dimg1, dimg2, dimg3, dimg4):
        # Convert ROS images to OpenCV format
        cv_img1 = self.rgb_msg_to_numpy(img1)
        cv_img2 = self.rgb_msg_to_numpy(img2)
        cv_img3 = self.rgb_msg_to_numpy(img3)
        cv_img4 = self.rgb_msg_to_numpy(img4)

        depth_image1 = self.depth_msg_to_numpy(dimg1)
        depth_image2 = self.depth_msg_to_numpy(dimg2)
        depth_image3 = self.depth_msg_to_numpy(dimg3)
        depth_image4 = self.depth_msg_to_numpy(dimg4)

        timestamp1_ms = img1.header.stamp.sec * 1000 + img1.header.stamp.nanosec / 1e6
        timestamp2_ms = img2.header.stamp.sec * 1000 + img2.header.stamp.nanosec / 1e6
        timestamp3_ms = img3.header.stamp.sec * 1000 + img3.header.stamp.nanosec / 1e6

        # Save color and depth images with sequential numbers
        self.save_images(cv_img1, depth_image1, self.camera_names[0])
        self.save_images(cv_img2, depth_image2, self.camera_names[1])
        self.save_images(cv_img3, depth_image3, self.camera_names[2])
        self.save_images(cv_img4, depth_image4, self.camera_names[3])


        return cv_img1, cv_img2, cv_img3, cv_img4, depth_image1, depth_image2, depth_image3, depth_image4

    def save_images(self, color_image, depth_image, camera_name, timestamp_ms=None):
        # Get current counter and pad the number to 6 digits
        counter = self.counters[camera_name]
        if timestamp_ms is not None:
            filename = f"{timestamp_ms:013.0f}.png"
        else:
            filename = f"{counter:06d}.png"

        # Update counters
        self.counters[camera_name] += 1

        # Save the color and depth images
        color_image_path = f'{self.save_dir}/{camera_name}/color/color_{filename}'
        depth_image_path = f'{self.save_dir}/{camera_name}/depth/depth_{filename}'

        cv2.imwrite(color_image_path, color_image)
        cv2.imwrite(depth_image_path, depth_image)

        print(f"Saved {color_image_path} and {depth_image_path}")


# def main(args=None):
#     if args is None:
#         args = sys.argv
#     save_dir = '.'
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     rclpy.init(args=args)
#     node = ImageSubscriberNode(camera_names=['cam_0', 'cam_1', 'cam_2', 'cam_3'], save_dir=save_dir)
#     rclpy.spin(node)
#     rclpy.shutdown()


def main(args=None):
    if args is None:
        args = sys.argv
    
    if len(args) < 2:
        print("Usage: python3 image_extraction_script.py <save_directory>")
        return
    
    save_dir = args[1]  # Get save directory from command-line arguments
    os.makedirs(save_dir, exist_ok=True)
    
    rclpy.init(args=args)
    node = ImageSubscriberNode(camera_names=['cam_0', 'cam_1', 'cam_2', 'cam_3'], save_dir=save_dir)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()