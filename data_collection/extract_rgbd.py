import os
import cv2
import numpy as np
import open3d as o3d
import rosbag2_py

width = 1280
height =  720

bag_file = '/home/coolbot/record_test/record_test_0.db3'
output_dir = '/home/coolbot/record_test/record_test_rgbd'

color_topics = '/camera/cam_{}/color/image_raw'
depth_topics = '/camera/cam_{}/depth/aligned_depth_to_color/image_raw'

# color_topic = '/camera/cam_2/color/image_raw'
# depth_topic = '/camera/cam_2/depth//camera/cam_2/aligned_depth_to_color/image_raw'

n_cameras = 4
color_topics_dict = {camera: color_topics.format(camera) for camera in range(n_cameras)}
depth_topics_dict = {camera: depth_topics.format(camera) for camera in range(n_cameras)}
for camera in range(n_cameras):
    color_topic = color_topics.format(camera)
    depth_topic = depth_topics.format(camera)


# color_metadata_topic = '/camera/camera/color/camera_info'
# depth_metadata_topic = '/camera/camera/depth/camera_info'

os.makedirs(output_dir, exist_ok=True)
# os.makedirs(os.path.join(output_dir, 'color'), exist_ok=True)
# os.makedirs(os.path.join(output_dir, 'depth'), exist_ok=True)
# os.makedirs(os.path.join(output_dir, 'color_metadata'), exist_ok=True)
# os.makedirs(os.path.join(output_dir, 'depth_metadata'), exist_ok=True)
for camera in range(n_cameras):
    os.makedirs(os.path.join(output_dir, 'cam_{}'.format(camera), 'color'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cam_{}'.format(camera), 'depth'), exist_ok=True)


reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

# read messages
while reader.has_next():
    # bridge = CvBridge()
    topic, data, t = reader.read_next()
    # change t type to int64
    timestamp_ms = int(t / 1e6)
    t = int(t)
    # if topic == color_topic:
    #     color_img = np.frombuffer(data, dtype=np.uint8)#.reshape(720, 1280, 3)
    #     color_img = color_img[-(480 * 640 * 3):]  # Trim any extra padding
    #     color_img = color_img.reshape(480, 640, 3)
    #     cv2.imwrite(os.path.join(output_dir, 'color', '{}.png'.format(timestamp_ms)), color_img)
    # elif topic == depth_topic:
    #     depth_img = np.frombuffer(data,dtype=np.uint16)
    #     depth_img = depth_img[-(720 * 1280):]
    #     depth_img = depth_img.reshape(720, 1280)
    #     depth_img = depth_img.astype(np.float32) / 1000.0
    #     cv2.imwrite(os.path.join(output_dir, 'depth', '{}.png'.format(timestamp_ms)), depth_img)
    if topic in color_topics_dict.values():
        camera = int(topic[12])
        color_img = np.frombuffer(data, dtype=np.uint8)
        color_img = color_img[-(width * height * 3):]
        color_img = color_img.reshape(height, width, 3)
        cv2.imwrite(os.path.join(output_dir, 'cam_{}'.format(camera) ,'color', '{}.png'.format(timestamp_ms)), color_img)
    elif topic in depth_topics_dict.values():
        camera = int(topic[12])
        depth_img = np.frombuffer(data, dtype=np.uint16)
        depth_img = depth_img[-(width * height):]
        depth_img = depth_img.reshape(height, width)
        depth_img = depth_img.astype(np.float32) / 1000.0
        cv2.imwrite(os.path.join(output_dir, 'cam_{}'.format(camera), 'depth', '{}.png'.format(timestamp_ms)), depth_img)
    

print('Done!')