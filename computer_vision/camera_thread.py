import threading
import cv2
import queue
import time
import platform

# --- BACKGROUND CAMERA THREAD ---
class VideoCaptureThread(threading.Thread):
    def __init__(self, video_source, raw_queue, running_event):
        super().__init__()
        self.video_source = video_source
        self.raw_queue = raw_queue
        self.running_event = running_event
        self.daemon = True  # Allows thread to exit when the main program closes

    def run(self):
        # Initialize camera with performance tweaks for Raspberry Pi
        if platform.system() == 'Linux':
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(self.video_source, cv2.CAP_DSHOW)
        
        # Optimize resolution for the Pi's CPU
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        while self.running_event.is_set():
            ret, frame = cap.read()
            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                # Capture high-precision timestamp immediately after read returns
                timestamp = time.perf_counter()
                
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                
                # Bundle them together
                data = {"timestamp": timestamp, "data": frame}
                # If the queue is full, pop the oldest frame to prevent memory bloat
                if self.raw_queue.full():
                    try:
                        self.raw_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                # Push the fresh frame onto the queue
                self.raw_queue.put(data)
            else:
                # Brief pause if the frame wasn't read successfully
                threading.Event().wait(0.01)

        cap.release()
