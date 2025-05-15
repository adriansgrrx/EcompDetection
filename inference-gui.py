import os
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, scrolledtext
from tkinter import font as tkfont
from PIL import Image, ImageTk
import threading
import queue
import serial
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict
import sys

class ComponentDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("EComp Detector")
        
        # Make window fullscreen and set background
        self.root.attributes('-fullscreen', True)
        self.root.configure(background='#2E2E2E')
        
        # Get screen dimensions
        self.screen_width = 800  # As per your LCD display width
        self.screen_height = 500  # As per your LCD display height
        
        # Create a style for ttk widgets
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TFrame', background='#2E2E2E')
        self.style.configure('TButton', font=('Arial', 12, 'bold'), background='#4CAF50')
        self.style.configure('Stats.TLabel', font=('Arial', 10), background='#2E2E2E', foreground='white')
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'), background='#2E2E2E', foreground='#4CAF50')
        
        # Initialize detection variables
        self.detection_running = False
        self.cap = None
        self.detection_thread = None
        self.update_thread = None
        self.stop_event = threading.Event()
        
        # Component detection variables - INITIALIZE THESE FIRST
        self.original_classes = ['BJT', 'LED', 'burnt', 'capacitor', 'cracked', 'faded', 'missing-leg', 'resistor', 'rust']
        self.defect_classes = {'burnt', 'cracked', 'faded', 'missing-leg', 'rust'}
        self.count_classes = ['BJT', 'LED', 'capacitor', 'resistor', 'defective', 'unknown']
        self.counts = {cls: 0 for cls in self.count_classes}
        self.seen_centroids = {cls: [] for cls in self.count_classes}
        self.active_objects = defaultdict(list)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid with 3 columns and 1 row
        self.main_frame.columnconfigure(0, weight=1)  # Column 1 - Stats
        self.main_frame.columnconfigure(1, weight=3)  # Column 2 - Video & Log
        self.main_frame.columnconfigure(2, weight=1)  # Column 3 - Controls
        self.main_frame.rowconfigure(0, weight=1)
        
        # Create frames for each column
        self.stats_frame = ttk.Frame(self.main_frame, style='TFrame')
        self.stats_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)
        
        self.video_log_frame = ttk.Frame(self.main_frame, style='TFrame')
        self.video_log_frame.grid(row=0, column=1, sticky='nsew', padx=10, pady=10)
        self.video_log_frame.rowconfigure(0, weight=5)  # 70% for video
        self.video_log_frame.rowconfigure(1, weight=5)  # 30% for log
        self.video_log_frame.columnconfigure(0, weight=1)
        
        self.control_frame = ttk.Frame(self.main_frame, style='TFrame')
        self.control_frame.grid(row=0, column=2, sticky='nsew', padx=10, pady=10)
        
        # Create video frame (70% of column 2)
        self.video_frame = ttk.Frame(self.video_log_frame, style='TFrame')
        self.video_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # Create log frame (30% of column 2)
        self.log_frame = ttk.Frame(self.video_log_frame, style='TFrame')
        self.log_frame.grid(row=1, column=0, sticky='nsew', padx=5, pady=5)
        
        # Set up UI components
        self.setup_stats_panel()
        self.setup_video_panel()
        self.setup_log_panel()
        self.setup_control_panel()
        
        # YOLO model
        try:
            self.model = None  # Will be loaded when detection starts
            self.log_message("GUI initialized. Ready to start detection.")
        except Exception as e:
            self.log_message(f"Error initializing: {str(e)}")
        
        # Arduino setup
        self.arduino = None
        self.component_queue = queue.Queue()
        self.cooldown_tracker = defaultdict(lambda: 0)
        self.cooldown_period = 10  # seconds
        
        # Frame display variables
        self.frame_w = 440
        self.frame_h = 440
        
        # Trigger rectangle settings
        self.rect_w, self.rect_h = 300, 480
        self.rect_x1 = (self.frame_w - self.rect_w) // 2
        self.rect_y1 = (self.frame_h - self.rect_h) // 2
        self.rect_x2 = self.rect_x1 + self.rect_w
        self.rect_y2 = self.rect_y1 + self.rect_h
        
        # Unknown component tracking
        self.last_detection_time = time.time()
        self.unknown_component_sent = False
        self.no_detection_threshold = 5.0
        
        # Tracking parameters
        self.min_distance = 30
        self.centroid_timeout = 10.0
        
        # Update stats initially
        self.update_stats()

    def setup_stats_panel(self):
        # Create header for stats
        ttk.Label(self.stats_frame, text="ECompDetect             ", style='Header.TLabel').pack(pady=(10, 20))
        
        # Create labels for each component type
        self.stat_labels = {}
        for component in self.count_classes:
            frame = ttk.Frame(self.stats_frame, style='TFrame')
            frame.pack(fill=tk.X, pady=5)
            
            ttk.Label(frame, text=f"{component}:", style='Stats.TLabel').pack(side=tk.LEFT, padx=5)
            count_label = ttk.Label(frame, text="0", style='Stats.TLabel')
            count_label.pack(side=tk.RIGHT, padx=5)
            
            self.stat_labels[component] = count_label
        
        # FPS display
        self.fps_frame = ttk.Frame(self.stats_frame, style='TFrame')
        self.fps_frame.pack(fill=tk.X, pady=(20, 5))
        ttk.Label(self.fps_frame, text="FPS:", style='Stats.TLabel').pack(side=tk.LEFT, padx=5)
        self.fps_label = ttk.Label(self.fps_frame, text="0.0", style='Stats.TLabel')
        self.fps_label.pack(side=tk.RIGHT, padx=5)
        
        # Status display
        self.status_frame = ttk.Frame(self.stats_frame, style='TFrame')
        self.status_frame.pack(fill=tk.X, pady=(20, 5))
        ttk.Label(self.status_frame, text="Status:", style='Stats.TLabel').pack(side=tk.LEFT, padx=5)
        self.status_label = ttk.Label(self.status_frame, text="IDLE", style='Stats.TLabel')
        self.status_label.pack(side=tk.RIGHT, padx=5)

    def setup_video_panel(self):
        # Create canvas for video display
        self.video_canvas = tk.Canvas(self.video_frame, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Display placeholder text
        self.video_canvas.create_text(
            self.screen_width // 3, self.screen_height // 3,
            text="Camera feed will appear here when detection starts",
            fill="white", font=('Arial', 12)
        )

    def setup_log_panel(self):
        # Create log area
        self.log_area = scrolledtext.ScrolledText(
            self.log_frame, 
            state='disabled',
            height=8,
            bg="#1E1E1E",
            fg="#FFFFFF",
            font=("Consolas", 10)
        )
        self.log_area.pack(fill=tk.BOTH, expand=True)

    def setup_control_panel(self):
        # Create control buttons with some space in between
        ttk.Label(self.control_frame, text="Controls", style='Header.TLabel').pack(pady=(10, 30))
        
        # Start button
        self.start_button = ttk.Button(
            self.control_frame, 
            text="Start",
            command=self.start_detection,
            style='TButton'
        )
        self.start_button.pack(pady=(0, 20), fill=tk.X, padx=20)
        
        # Stop button
        self.stop_button = ttk.Button(
            self.control_frame, 
            text="Stop",
            command=self.stop_detection,
            state=tk.DISABLED,
            style='TButton'
        )
        self.stop_button.pack(pady=(0, 20), fill=tk.X, padx=20)
        
        # Exit button
        self.exit_button = ttk.Button(
            self.control_frame, 
            text="Quit",
            command=self.shutdown_app,
            style='TButton'
        )
        self.exit_button.pack(pady=(0, 20), fill=tk.X, padx=20)

    def log_message(self, message):
        """Add a message to the log area with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        # Enable text widget, insert message, then disable it again
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, log_message)
        self.log_area.see(tk.END)  # Auto-scroll to the end
        self.log_area.config(state=tk.DISABLED)

    def update_stats(self):
        """Update the statistics display"""
        for component, count in self.counts.items():
            self.stat_labels[component].config(text=str(count))

    def update_status(self, status):
        """Update the status display"""
        self.status_label.config(text=status)

    def update_fps(self, fps):
        """Update the FPS display"""
        self.fps_label.config(text=f"{fps:.1f}")

    def get_centroid(self, xyxy):
        """Calculate centroid from bounding box coordinates"""
        x1, y1, x2, y2 = xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def start_arduino_communication(self):
        """Start Arduino communication in a separate thread"""
        try:
            self.arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
            time.sleep(2)  # Allow Arduino time to reset
            self.log_message("Arduino connection established")
            
            # Start Arduino communication thread
            self.arduino_thread = threading.Thread(target=self.arduino_communication, daemon=True)
            self.arduino_thread.start()
            
        except Exception as e:
            self.log_message(f"Arduino connection failed: {str(e)}")
            self.arduino = None

    def arduino_communication(self):
        """Arduino communication thread function"""
        while not self.stop_event.is_set():
            try:
                component = self.component_queue.get(timeout=1)
                if component and self.arduino:
                    self.log_message(f"Sending to Arduino: {component}")
                    self.arduino.write(component.encode())

                    # Wait for Arduino reply
                    response_timeout = time.time() + 2
                    while time.time() < response_timeout and not self.stop_event.is_set():
                        if self.arduino.in_waiting > 0:
                            response = self.arduino.readline().decode().strip()
                            self.log_message(f"Arduino response: {response}")
                            if response == "DONE":
                                break
                    else:
                        if not self.stop_event.is_set():
                            self.log_message("Arduino response timeout.")
            except queue.Empty:
                time.sleep(0.5)
            except Exception as e:
                self.log_message(f"Arduino communication error: {str(e)}")
                time.sleep(1)

    def start_detection(self):
        """Start the component detection process"""
        if self.detection_running:
            return
        
        self.detection_running = True
        self.stop_event.clear()
        
        # Update UI state
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("ON")
        
        # Start Arduino communication
        self.start_arduino_communication()
        
        # Load YOLO model
        try:
            self.log_message("Loading YOLO model...")
            self.model = YOLO('ecomp-detect-yolov8n_edgetpu.tflite', task='detect')
            self.log_message("YOLO model loaded successfully")
        except Exception as e:
            self.log_message(f"Failed to load YOLO model: {str(e)}")
            self.stop_detection()
            return
        
        # Open camera
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.log_message("Failed to open webcam.")
                self.stop_detection()
                return
            
            # Set fixed frame dimensions
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_h)
            
            # Verify settings
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.log_message(f"Frame dimensions: Width = {actual_w}, Height = {actual_h}")
            
        except Exception as e:
            self.log_message(f"Camera error: {str(e)}")
            self.stop_detection()
            return
        
        # Start detection thread
        self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.detection_thread.start()
        
        # Start update thread for UI
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()
        
        self.log_message("Detection Starting...")

    def update_loop(self):
        """Update UI elements periodically"""
        while not self.stop_event.is_set():
            self.update_stats()
            time.sleep(0.5)

    def detection_loop(self):
        """Main detection loop running in a separate thread"""
        last_fps_update = time.time()
        fps_values = []
        
        while not self.stop_event.is_set():
            try:
                tic = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    self.log_message("Failed to read frame from camera")
                    time.sleep(0.5)
                    continue
                
                # Run detection
                result = self.model(frame, imgsz=256, conf=0.7, verbose=False)
                detections = result[0].boxes
                annotated_frame = result[0].plot()
                
                current_time = time.time()
                
                # Check for any detections in the rectangle
                found_detection_in_rect = False
                
                for cls in self.count_classes:
                    self.seen_centroids[cls] = [(cx, cy, t) for cx, cy, t in self.seen_centroids[cls] 
                                               if current_time - t < self.centroid_timeout]
                
                # Cleanup active_objects after timeout
                for cls in list(self.active_objects.keys()):
                    self.active_objects[cls] = [(cx, cy, t) for cx, cy, t in self.active_objects[cls] 
                                               if current_time - t < self.centroid_timeout]
                
                if detections is not None and detections.xyxy is not None and len(detections.xyxy) > 0:
                    for i, box in enumerate(detections.xyxy):
                        cls_id = int(detections.cls[i].item())
                        original_class = self.original_classes[cls_id]
                        class_name = 'defective' if original_class in self.defect_classes else original_class
                        
                        if class_name not in self.counts:
                            continue
                        
                        x1, y1, x2, y2 = map(int, box)
                        cx, cy = self.get_centroid((x1, y1, x2, y2))
                        cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)
                        
                        if self.rect_x1 <= cx <= self.rect_x2 and self.rect_y1 <= cy <= self.rect_y2:
                            found_detection_in_rect = True
                            self.last_detection_time = current_time
                            self.unknown_component_sent = False  # Reset flag when a detection is found
                            
                            already_tracked = False
                            for prev_cx, prev_cy, _ in self.active_objects[class_name]:
                                distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                                if distance < self.min_distance:
                                    already_tracked = True
                                    break
                            
                            if not already_tracked:
                                self.counts[class_name] += 1
                                self.seen_centroids[class_name].append((cx, cy, current_time))
                                self.active_objects[class_name].append((cx, cy, current_time))
                                self.log_message(f"Detected {class_name} - Count: {self.counts[class_name]}")
                                
                                if current_time - self.cooldown_tracker[class_name] > self.cooldown_period:
                                    self.cooldown_tracker[class_name] = current_time
                                    if class_name == 'BJT':
                                        self.component_queue.put('A')
                                    elif class_name == 'LED':
                                        self.component_queue.put('B')
                                    elif class_name == 'capacitor':
                                        self.component_queue.put('C')
                                    elif class_name == 'defective':
                                        self.component_queue.put('D')
                                    elif class_name == 'resistor':
                                        self.component_queue.put('E')
                
                # Handle unknown component detection
                elapsed_time = current_time - self.last_detection_time
                
                # Display time since last detection
                cv2.putText(annotated_frame, f'Time since last: {elapsed_time:.1f}s', 
                            (10, self.frame_h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Check if we need to handle an unknown component
                if elapsed_time >= self.no_detection_threshold and not self.unknown_component_sent:
                    # Extract the rectangle region
                    roi = frame[self.rect_y1:self.rect_y2, self.rect_x1:self.rect_x2]
                    
                    # Convert to grayscale and apply blur to reduce noise
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (21, 21), 0)
                    
                    # Use a simple threshold to find contours
                    _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Check if there are any significant contours
                    significant_contours = [c for c in contours if cv2.contourArea(c) > 500]
                    
                    if significant_contours:
                        self.log_message("Unknown component detected")
                        self.counts['unknown'] += 1
                        self.component_queue.put('U')  # Send 'U' for unknown component
                        self.unknown_component_sent = True
                        self.last_detection_time = current_time  # Reset timer
                
                # Draw central trigger rectangle
                cv2.rectangle(annotated_frame, (self.rect_x1, self.rect_y1), 
                              (self.rect_x2, self.rect_y2), (255, 0, 0), 2)
                
                # REMOVED: Detection counts on frame
                # No longer displaying counts directly on the OpenCV frame
                
                # Calculate and display FPS
                fps = 1.0 / (time.time() - tic)
                fps_values.append(fps)
                
                # Update FPS display every second
                if time.time() - last_fps_update > 1.0:
                    avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
                    self.root.after(0, self.update_fps, avg_fps)
                    fps_values = []
                    last_fps_update = time.time()
                
                cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (annotated_frame.shape[1] - 150, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                
                # Resize to fit canvas
                canvas_width = self.video_canvas.winfo_width()
                canvas_height = self.video_canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
                
                photo = ImageTk.PhotoImage(image=img)
                
                # Update canvas with new image
                self.video_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.video_canvas.image = photo  # Keep a reference to prevent garbage collection
                
            except Exception as e:
                self.log_message(f"Detection error: {str(e)}")
                time.sleep(0.5)
        
        self.log_message("Detection loop stopped.")

    def stop_detection(self):
        """Stop the detection process"""
        if not self.detection_running:
            return
        
        self.log_message("Stopping detection...")
        self.stop_event.set()
        
        # Close camera
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        # Close Arduino connection
        if self.arduino is not None:
            self.arduino.close()
            self.arduino = None
        
        # Wait for threads to finish
        if self.detection_thread is not None:
            self.detection_thread.join(timeout=2.0)
        
        if self.update_thread is not None:
            self.update_thread.join(timeout=2.0)
        
        # Reset UI
        self.detection_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.update_status("IDLE")
        self.fps_label.config(text="0.0")
        
        # Clear video canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            self.screen_width // 3, self.screen_height // 3,
            text="Camera feed will appear here when detection starts",
            fill="white", font=('Arial', 12)
        )
        
        self.log_message("Detection stopped.")

    def shutdown_app(self):
        """Shutdown the application"""
        self.log_message("Shutting down...")
        
        # Stop detection if running
        if self.detection_running:
            self.stop_detection()
        
        # Allow time for logs to be displayed
        self.root.after(1000, self.root.destroy)

if __name__ == "__main__":
    root = tk.Tk()
    app = ComponentDetectionGUI(root)
    root.mainloop()