import cv2
import numpy as np
from collections import deque

class RieszVisualizer:
    """
    Visualizes the four Riesz Kernel tracks (R0, R1, R2, R3) in real-time.
    Designed to match the supervisor's 'Four Tracks' diagram.
    """
    def __init__(self, history_len=60, width=200, height=300):
        self.history_len = history_len
        self.width = width
        self.height = height
        self.track_height = height // 4
        
        # Buffers for the 4 tracks
        self.tracks = {
            'R0': deque(maxlen=history_len), # Depth
            'R1': deque(maxlen=history_len), # Velocity
            'R2': deque(maxlen=history_len), # Acceleration
            'R3': deque(maxlen=history_len)  # Jerk
        }
        
        # Colors matching the diagram
        self.colors = {
            'R0': (80, 40, 0),    # Dark Navy (BGR)
            'R1': (0, 150, 0),   # Green
            'R2': (200, 100, 0), # Blue
            'R3': (0, 165, 255)  # Orange/Yellow
        }

    def update(self, r0, r1, r2, r3):
        """Update the tracks with new values."""
        self.tracks['R0'].append(r0)
        self.tracks['R1'].append(r1)
        self.tracks['R2'].append(r2)
        self.tracks['R3'].append(r3)

    def draw_to_window(self):
        """Creates a standalone window and draws the monitor inside it."""
        # Create a blank dark canvas for the standalone window
        monitor_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        monitor_frame[:] = (20, 20, 20) # Background
        
        # Draw each track
        for i, (key, data) in enumerate(self.tracks.items()):
            if len(data) < 2: continue
            
            ty = i * self.track_height
            # Draw baseline
            cv2.line(monitor_frame, (0, ty + self.track_height//2), (self.width, ty + self.track_height//2), (60, 60, 60), 1)
            # Label
            cv2.putText(monitor_frame, key, (10, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Normalize and draw the graph
            vals = list(data)
            max_val = max(abs(min(vals)), abs(max(vals)), 1e-5)
            
            # Special scaling for R3 to show the "Impulse Spike"
            if key == 'R3':
                max_val = max(max_val, 100000) # Keep scale consistent for Jerk
            
            for j in range(len(vals) - 1):
                x1 = (j * self.width // self.history_len)
                x2 = ((j + 1) * self.width // self.history_len)
                
                # Center-aligned vertical scaling
                y1 = ty + (self.track_height // 2) - int((vals[j] / max_val) * (self.track_height // 2.2))
                y2 = ty + (self.track_height // 2) - int((vals[j+1] / max_val) * (self.track_height // 2.2))
                
                cv2.line(monitor_frame, (x1, y1), (x2, y2), self.colors[key], 2)
            
            # Draw separator lines
            if i < 3:
                cv2.line(monitor_frame, (0, ty + self.track_height), (self.width, ty + self.track_height), (80, 80, 80), 1)

        cv2.imshow('Riesz Kernel Monitor', monitor_frame)

    def draw(self, frame):
        """Draw the monitor overlay on the frame (Original method)."""
        h, w = frame.shape[:2]
        
        # Create a semi-transparent side panel for the monitor
        panel_x = w - self.width - 10
        panel_y = 50
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + self.width, panel_y + self.height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw each track
        for i, (key, data) in enumerate(self.tracks.items()):
            if len(data) < 2: continue
            
            ty = panel_y + i * self.track_height
            # Draw baseline
            cv2.line(frame, (panel_x, ty + self.track_height//2), (panel_x + self.width, ty + self.track_height//2), (60, 60, 60), 1)
            # Label
            cv2.putText(frame, key, (panel_x + 5, ty + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            
            # Normalize and draw the graph
            vals = list(data)
            max_val = max(abs(min(vals)), abs(max(vals)), 1e-5)
            
            # Special scaling for R3 to show the "Impulse Spike"
            if key == 'R3':
                max_val = max(max_val, 100000) # Keep scale consistent for Jerk
            
            for j in range(len(vals) - 1):
                x1 = panel_x + (j * self.width // self.history_len)
                x2 = panel_x + ((j + 1) * self.width // self.history_len)
                
                # Center-aligned vertical scaling
                y1 = ty + (self.track_height // 2) - int((vals[j] / max_val) * (self.track_height // 2.2))
                y2 = ty + (self.track_height // 2) - int((vals[j+1] / max_val) * (self.track_height // 2.2))
                
                cv2.line(frame, (x1, y1), (x2, y2), self.colors[key], 2)

        # Border
        cv2.rectangle(frame, (panel_x, panel_y), (panel_x + self.width, panel_y + self.height), (150, 150, 150), 1)
        return frame