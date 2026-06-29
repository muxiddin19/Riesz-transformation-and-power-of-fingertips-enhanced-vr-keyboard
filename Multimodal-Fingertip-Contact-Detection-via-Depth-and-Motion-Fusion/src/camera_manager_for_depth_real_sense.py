import cv2
import numpy as np
import pyrealsense2 as rs


class RealSenseDepthFrame:
    """Wrapper for RealSense depth frame with RealSense-like interface."""
    
    def __init__(self, depth_frame, depth_scale):
        self.depth_frame = depth_frame
        self.depth_scale = depth_scale
        self.depth_map = np.asanyarray(depth_frame.get_data()) * depth_scale
        self.width = depth_frame.get_width()
        self.height = depth_frame.get_height()
    
    def get_distance(self, x, y):
        """Get depth at pixel coordinates (x, y) in meters."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return float(self.depth_frame.get_distance(x, y))
        return 0.0


class CameraManager:
    """RealSense D405 camera manager."""
    
    def __init__(self, color_width=640, color_height=480, depth_width=640, depth_height=480, fps=30):
        self.color_width = color_width
        self.color_height = color_height
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.fps = fps
        
        self.pipeline = None
        self.config = None
        self.align = None
        self.depth_scale = None
        
        # Depth estimator (not used for RealSense, but kept for compatibility)
        self.depth_estimator = None
        
        # Frame skipping for performance
        self.frame_count = 0
        self.last_depth_frame = None
    
    def get_resolution(self):
        return self.color_width, self.color_height, self.depth_width, self.depth_height, self.fps
    
    def set_depth_estimator(self, depth_estimator):
        """Set depth estimator (not used for RealSense hardware depth)."""
        self.depth_estimator = depth_estimator
        print("Depth estimator set (not used with RealSense hardware depth)")

    def start_stream(self):
        """Initialize RealSense D405 stream."""
        print("Starting RealSense D405 stream...")
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # Enable streams
        self.config.enable_stream(rs.stream.depth, self.depth_width, self.depth_height, rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.color_width, self.color_height, rs.format.bgr8, self.fps)
        
        try:
            self.pipeline.start(self.config)
        except Exception as e:
            print(f"Error: Could not start RealSense: {e}")
            return False
        
        # Align depth to color
        self.align = rs.align(rs.stream.color)
        
        # Get depth scale
        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        print(f"RealSense D405 started: {self.color_width}x{self.color_height} @ {self.fps}fps")
        print(f"Depth scale: {self.depth_scale}")
        
        return True
    
    def get_frames(self):
        """
        Get color frame and depth frame from RealSense.
        Returns: (color_image, depth_frame, depth_dimensions)
        """
        if self.pipeline is None:
            return None, None, None
        
        # Wait for frames
        frames = self.pipeline.wait_for_frames()
        
        # Align depth to color
        aligned_frames = self.align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None, None
        
        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())
        
        # Create depth frame wrapper
        rs_depth_frame = RealSenseDepthFrame(depth_frame, self.depth_scale)
        
        depth_dimensions = (self.depth_width, self.depth_height)
        
        return color_image, rs_depth_frame, depth_dimensions
    
    def stop_stream(self):
        """Stop RealSense stream."""
        if self.pipeline is not None:
            print("Stopping RealSense stream.")
            self.pipeline.stop()
            self.pipeline = None