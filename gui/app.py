import os
import sys
import tkinter as tk
import queue
import threading
import cv2
import customtkinter as ctk
import time
from PIL import Image, ImageTk
from computer_vision.camera_thread import VideoCaptureThread
from computer_vision.opencv_thread import OpenCVThread
from computer_vision.sync_thread import SyncManagerThread
from computer_vision.calibration import Calibrator
from computer_vision.evaluation_thread import EvalThread

# --- MAIN CUSTOMTKINTER APPLICATION ---
class App(ctk.CTk):
    def __init__(self, video_source1=0, video_source2=2):
        super().__init__()

        # Configure Main Window
        self.title("FormSnitch")
        self.geometry("720x1280")
        self.update_idletasks()
        self.config(cursor="none")
        self.attributes("-fullscreen", True)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Threading Control Variables
        self.raw_queue1 = queue.Queue(maxsize=10)
        self.raw_queue2 = queue.Queue(maxsize=10)
        self.sync_queue1 = queue.Queue(maxsize=1)
        self.sync_queue2 = queue.Queue(maxsize=1)
        self.processed_queue1 = queue.Queue(maxsize=1)
        self.processed_queue2 = queue.Queue(maxsize=1)
        self.output_image_queue = queue.Queue(maxsize=1)
        
        self.running_event = threading.Event()
        self.ml_running_event = threading.Event()
        self.show_camera1 = threading.Event()
        self.running_event.set()
        self.ml_running_event.set()
        self.show_camera1.set()

        # Create Calibrator Object
        self.calibrator = Calibrator()
        self.calibrator.load_calibration()  # Load latest calibration results
        
        # Create the video capture background threads
        self.video_thread1 = VideoCaptureThread(video_source1, self.raw_queue1, self.running_event)
        self.video_thread2 = VideoCaptureThread(video_source2, self.raw_queue2, self.running_event)
        
        # Create Synchronization Thread
        self.sync_thread = SyncManagerThread(self.raw_queue1, self.raw_queue2, self.sync_queue1, self.sync_queue2, self.running_event, 0.01)
        
        # Create ML Threads
        self.opencv_thread1 = OpenCVThread(self.sync_queue1, self.processed_queue1, self.ml_running_event)
        self.opencv_thread2 = OpenCVThread(self.sync_queue2, self.processed_queue2, self.ml_running_event)
        
        # Create Evaluation Thread
        self.eval_thread = EvalThread(self.processed_queue1, self.processed_queue2, self.output_image_queue, self.calibrator, self.running_event, (640,480))
        
        # Start Threads
        self.video_thread1.start()
        self.video_thread2.start()
        self.sync_thread.start()
        self.opencv_thread1.start()
        self.opencv_thread2.start()
        self.eval_thread.start()

        # Configure Grid Layout (2 Columns: Left for Video, Right for Controls)
        self.grid_rowconfigure(0, weight=4)  # Video column takes up more space
        self.grid_rowconfigure(1, weight=1, minsize=150)  # Bottombar column
        self.grid_columnconfigure(0, weight=1)

        # Create UI Elements
        self._create_widgets()

        # Handle window X close button gracefully
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.bind("<Escape>", self.exit_fullscreen)

        # Start the GUI update loop for the video stream
        self.update_video_feed()

    def _create_widgets(self):
        # Video Display Area
        self.video_label = ctk.CTkLabel(self, text="Loading Camera Feed...", fg_color="#1a1a1a")
        self.video_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        #Calibration instructions
        self.calib_instruction_label = ctk.CTkLabel(
        self,
        text="",
        font=ctk.CTkFont(size=28, weight="bold"),
        text_color="white",
        fg_color="#000000"
        )

        self.calib_instruction_label.grid(
        row=0,
        column=0,
        padx=40,
        pady=40,
        sticky="s"
        )

        # Control Bottombar 
        self.bottombar = ctk.CTkFrame(self, corner_radius=10)
        self.bottombar.grid(row=1, column=0, padx=(0, 20), pady=20, sticky="nsew")
        self.bottombar.grid_columnconfigure(0, weight=1)
        self.bottombar.grid_rowconfigure(0, weight=1)
        self.bottombar.grid_columnconfigure((1, 2, 3), weight=5)
        self.bottombar.grid_rowconfigure((0,1), weight=1)

        # Bottombar Title
        self.bottombar_label = ctk.CTkLabel(self.bottombar, text="CONTROLS", font=ctk.CTkFont(size=16, weight="bold"))
        self.bottombar_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        self.switch_view_btn = ctk.CTkButton(
            self.bottombar, 
            text="Switch Camera", 
            fg_color="#1f538d", 
            hover_color="#14375e",
            command=self.switch_view
        )
        self.switch_view_btn.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.calib_btn = ctk.CTkButton(
            self.bottombar, 
            text="Calibrate Cameras", 
            fg_color="#1f538d", 
            hover_color="#14375e",
            command=self.calibrate_cam
        )
        self.calib_btn.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")

        # System Shutdown Button
        self.shutdown_btn = ctk.CTkButton(
            self.bottombar, 
            text="System Shutdown", 
            fg_color="#942a2a", 
            hover_color="#661c1c",
            command=self.shutdown_pi
        )
        self.shutdown_btn.grid(row=0, column=3, padx=20, pady=20, sticky="nsew")
        
        self.dist_slider = ctk.CTkSlider(self.bottombar, from_=1000, to=5000, command=self.on_slider_move)
        self.dist_slider.set(3000)
        self.dist_slider.grid(row=1, column=2, padx=20, pady=20, sticky="nsew")
        
        self.yaw_slider = ctk.CTkSlider(self.bottombar, from_=-180, to=180, command=self.on_slider_move)
        self.yaw_slider.set(0)
        self.yaw_slider.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        
    def on_slider_move(self, value):
        yaw = self.yaw_slider.get()
        dist = self.dist_slider.get()
        self.eval_thread.update_view_angles(pitch=0, yaw=yaw, roll=0, distance=dist)

    def update_video_feed(self):
        """Checks the queue for new frames and updates the UI."""
        if self.running_event.is_set():
            try:
                # Grab the latest frame from the thread queue
                frame = self.output_image_queue.get_nowait()
                
                # Dynamically match widget size while preserving aspect ratio roughly
                img_width, img_height = self.video_label.winfo_width(), self.video_label.winfo_height()
                if img_width < 10: img_width = 480  # Fallback defaults for first frame
                if img_height < 10: img_height = 640
                
                frame_resized = cv2.resize(frame, (img_width, img_height), interpolation=cv2.INTER_LINEAR)
                
                # Convert OpenCV BGR format to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to CustomTkinter CTkImage
                pil_img = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(img_width, img_height))
                
                # Update label image
                self.video_label.configure(image=ctk_img, text="")
                self.video_label.image = ctk_img  # Keep a reference to prevent garbage collection
                
            except queue.Empty:
                pass

            self.after(30, self.update_video_feed)
    
    def exit_fullscreen(self, event=None):
        self.attributes("-fullscreen", False)
        # You can also set a default window size to fall back to
        self.geometry("720x1280")

    def switch_view(self):
        viewport = self.eval_thread.get_viewport()
        if not self.calibrator.is_calibrated() and viewport == 1 or viewport == 2:
            self.eval_thread.set_viewport(0)
        else:
            self.eval_thread.set_viewport(viewport+1)
        print("Switch Action Triggered!")
    
    def _set_calibration_instruction(self, text):
        self.calib_instruction_label.configure(text=text)

    def calibrate_cam(self):
        self.calib_btn.configure(state="disabled", text="Calibrating...")
        self.ml_running_event.clear()
        calib_thread = threading.Thread(target=self._run_calibration_async, daemon=True)
        calib_thread.start()
        print("Calibrate Action Triggered!")
        
    def _run_calibration_async(self):
        self.after(0, lambda: self.calib_btn.configure(text="Prepare checkerboard..."))
        self.after(0, lambda: self._set_calibration_instruction(
            "Paruoškite šachmatų lentą.\nLaikykite ją taip, kad ją matytų abi kameros."
        ))

        time.sleep(5)

        success = self.calibrator.calibrate(
            self.sync_queue1,
            self.sync_queue2,
            progress_callback=self._update_calibration_progress
        )

        self.after(0, lambda: self._on_calibration_complete(success))

    def _update_calibration_progress(self, current, total):
        if current <= 30:
            instruction = (
                f"1 etapas: laikykite lentą ramiai.\n"
                f"Kadras {current}/{total}"
            )
        elif current <= 80:
            instruction = (
                f"2 etapas: lėtai judinkite lentą į kairę ir dešinę.\n"
                f"Kadras {current}/{total}"
            )
        else:
            instruction = (
                f"3 etapas: lėtai judinkite lentą aukštyn ir žemyn.\n"
                f"Kadras {current}/{total}"
            )

        self.after(
            0,
            lambda: (
                self.calib_btn.configure(text=f"Calibrating... {current}/{total}"),
                self._set_calibration_instruction(instruction)
            )
        )

    def _on_calibration_complete(self, success):
        self.ml_running_event.set()
        self.calib_btn.configure(state="normal", text="Calibrate Cameras")
        
    def shutdown_pi(self):
        """Safely stops threads and shuts down the Raspberry Pi operating system."""
        print("Shutting down system...")
        # Executes the Linux shutdown command
        # os.system("systemctl poweroff")
        self.on_closing()

    def on_closing(self):
        """Cleans up background threads safely before closing the window."""
        self.running_event.clear()
        self.ml_running_event.set()
        if self.video_thread1.is_alive():
            self.video_thread1.join(timeout=1.0)
        if self.video_thread2.is_alive():
            self.video_thread2.join(timeout=1.0)
        self.destroy()
        sys.exit(0)
