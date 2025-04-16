import pyrealsense2 as rs
import numpy as np
import cv2

class Realsense:
    def __init__(self):
        pipeline = rs.pipeline()
        self.pipeline = pipeline

        rs_config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = rs_config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))
        rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        self.depth_scale = depth_scale
        clipping_distance_in_meters = 1 #1 meter
        clipping_distance = clipping_distance_in_meters / depth_scale
        align_to = rs.stream.color
        align = rs.align(align_to)

        self.align = align
    
    def get_image(self):
        pipeline = self.pipeline
        align = self.align

        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        fx = color_intrinsics.fx
        fy = color_intrinsics.fy
        ppx = color_intrinsics.ppx
        ppy = color_intrinsics.ppy
        cam_K = np.array([
            [fx, 0, ppx],
            [0, fy, ppy],
            [0, 0, 1]
        ])

        if not aligned_depth_frame or not color_frame:
            return None 
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * self.depth_scale * 1000).astype(np.float32)
        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.1) | (depth>=np.inf)] = 0

        return color, depth, cam_K

    def close(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
