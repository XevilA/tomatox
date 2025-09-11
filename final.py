#!/usr/bin/env python3
"""
Main Application - Tomato Harvesting System with Real-time Detection
ESP32 Servo Control with Detection Frames
"""

import sys
import json
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import logging

# PyQt6 imports
from PyQt6.QtCore import (Qt, QThread, QTimer, pyqtSignal, QObject, 
                         QRect, QRectF, QPointF)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSlider, QGroupBox, QGridLayout, QFileDialog,
                             QMessageBox, QTextEdit, QCheckBox, QSpinBox)
from PyQt6.QtGui import (QImage, QPixmap, QPainter, QPen, QColor, QFont,
                        QBrush)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. Install with: pip install tensorflow")


@dataclass
class Detection:
    """Detection result with bounding box"""
    x: int
    y: int
    width: int
    height: int
    label: str
    confidence: float
    color: tuple


class TomatoDetector(QObject):
    """AI Model for tomato detection with bounding boxes"""
    detection_complete = pyqtSignal(list, float)  # detections, fps
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = None
        self.labels = ['unripe', 'ripe', 'rotten']
        self.colors = {
            'unripe': (0, 255, 0),     # Green
            'ripe': (0, 0, 255),       # Red
            'rotten': (139, 69, 19),   # Brown
            'unknown': (128, 128, 128) # Gray
        }
        self.confidence_threshold = 0.5
        
    def load_model(self, model_path: str) -> bool:
        """Load TensorFlow model"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
            
        try:
            self.model = keras.models.load_model(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded: {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_labels(self, labels_path: str) -> bool:
        """Load label names from text file"""
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.labels)} labels")
            return True
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return False
    
    def detect_tomatoes(self, frame: np.ndarray) -> List[Detection]:
        """Detect tomatoes and return bounding boxes"""
        detections = []
        height, width = frame.shape[:2]
        
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define HSV ranges for tomatoes
        ranges = {
            'unripe': [(35, 50, 50), (85, 255, 255)],    # Green
            'ripe': [(0, 50, 50), (10, 255, 255)],       # Red lower
            'ripe2': [(170, 50, 50), (180, 255, 255)],   # Red upper
            'rotten': [(10, 50, 20), (25, 255, 150)]     # Brown
        }
        
        # Process each color range
        for label, (lower, upper) in ranges.items():
            if label == 'ripe2':
                label = 'ripe'  # Merge red ranges
                
            # Create mask
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            
            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area (adjust based on camera distance)
                if area > 1000:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on shape and area
                    confidence = self.calculate_confidence(contour, area, w, h)
                    
                    if confidence > self.confidence_threshold:
                        # If using AI model, classify the ROI
                        if self.model:
                            roi = frame[y:y+h, x:x+w]
                            label, confidence = self.classify_roi(roi)
                        
                        # Create detection
                        detection = Detection(
                            x=x, y=y, width=w, height=h,
                            label=label,
                            confidence=confidence,
                            color=self.colors.get(label, self.colors['unknown'])
                        )
                        detections.append(detection)
        
        # Non-maximum suppression to remove overlapping boxes
        detections = self.non_max_suppression(detections)
        
        return detections
    
    def calculate_confidence(self, contour, area, w, h) -> float:
        """Calculate detection confidence based on shape"""
        # Check circularity (tomatoes are roughly circular)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Check aspect ratio (should be close to 1 for circles)
        aspect_ratio = w / h if h > 0 else 0
        aspect_score = 1.0 - abs(1.0 - aspect_ratio)
        
        # Combine scores
        confidence = (circularity * 0.6 + aspect_score * 0.4)
        
        return min(confidence, 1.0)
    
    def classify_roi(self, roi: np.ndarray) -> Tuple[str, float]:
        """Classify region of interest using AI model"""
        if not self.model or roi.size == 0:
            return "unknown", 0.0
            
        try:
            # Preprocess for model
            roi_resized = cv2.resize(roi, (224, 224))
            roi_normalized = roi_resized.astype(np.float32) / 255.0
            roi_batch = np.expand_dims(roi_normalized, axis=0)
            
            # Predict
            predictions = self.model.predict(roi_batch, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            label = self.labels[class_idx] if class_idx < len(self.labels) else "unknown"
            
            return label, confidence
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "unknown", 0.0
    
    def non_max_suppression(self, detections: List[Detection], 
                           iou_threshold: float = 0.5) -> List[Detection]:
        """Remove overlapping detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            # Keep the highest confidence detection
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [d for d in detections 
                         if self.calculate_iou(best, d) < iou_threshold]
        
        return keep
    
    def calculate_iou(self, det1: Detection, det2: Detection) -> float:
        """Calculate Intersection over Union"""
        # Calculate intersection
        x1 = max(det1.x, det2.x)
        y1 = max(det1.y, det2.y)
        x2 = min(det1.x + det1.width, det2.x + det2.width)
        y2 = min(det1.y + det1.height, det2.y + det2.height)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = det1.width * det1.height
        area2 = det2.width * det2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """Process frame and return annotated image with detections"""
        start_time = time.time()
        
        # Detect tomatoes
        detections = self.detect_tomatoes(frame)
        
        # Draw detection boxes
        annotated_frame = frame.copy()
        for det in detections:
            # Draw bounding box
            cv2.rectangle(annotated_frame, 
                         (det.x, det.y), 
                         (det.x + det.width, det.y + det.height),
                         det.color, 2)
            
            # Draw label background
            label_text = f"{det.label}: {det.confidence:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            # Get text size
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(annotated_frame,
                         (det.x, det.y - text_height - 10),
                         (det.x + text_width + 10, det.y),
                         det.color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label_text,
                       (det.x + 5, det.y - 5),
                       font, font_scale, (255, 255, 255), thickness)
            
            # Draw center point
            center_x = det.x + det.width // 2
            center_y = det.y + det.height // 2
            cv2.circle(annotated_frame, (center_x, center_y), 5, det.color, -1)
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        
        # Emit signal
        self.detection_complete.emit(detections, fps)
        
        return annotated_frame, detections


class CameraThread(QThread):
    """Camera capture thread"""
    frame_ready = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        
    def run(self):
        """Main capture loop"""
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                logger.error("Failed to capture frame")
            time.sleep(0.033)  # ~30 FPS
    
    def stop(self):
        """Stop capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class CameraWidget(QWidget):
    """Widget to display camera feed with detection boxes"""
    
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.detections = []
        self.fps = 0.0
        self.show_detections = True
        self.detection_stats = {'unripe': 0, 'ripe': 0, 'rotten': 0}
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        
    def update_frame(self, frame: np.ndarray, detections: List[Detection] = None, fps: float = 0):
        """Update displayed frame"""
        self.current_frame = frame
        if detections is not None:
            self.detections = detections
            self.update_stats()
        self.fps = fps
        self.update()
    
    def update_stats(self):
        """Update detection statistics"""
        self.detection_stats = {'unripe': 0, 'ripe': 0, 'rotten': 0}
        for det in self.detections:
            if det.label in self.detection_stats:
                self.detection_stats[det.label] += 1
    
    def paintEvent(self, event):
        """Custom paint event"""
        if self.current_frame is None:
            return
        
        painter = QPainter(self)
        
        # Convert frame to QImage
        height, width, channel = self.current_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.current_frame.data, width, height,
                        bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale to widget size
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        
        # Center image
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        
        # Draw overlay info
        self.draw_overlay(painter)
    
    def draw_overlay(self, painter: QPainter):
        """Draw overlay information"""
        # Setup font
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        # FPS counter
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.drawText(10, 25, f"FPS: {self.fps:.1f}")
        
        # Detection count
        y_pos = 50
        for label, count in self.detection_stats.items():
            if count > 0:
                if label == 'unripe':
                    painter.setPen(QPen(QColor(0, 255, 0), 2))
                elif label == 'ripe':
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                elif label == 'rotten':
                    painter.setPen(QPen(QColor(139, 69, 19), 2))
                
                painter.drawText(10, y_pos, f"{label.capitalize()}: {count}")
                y_pos += 25
        
        # Detection status
        if self.show_detections:
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawText(self.width() - 150, 25, "Detection: ON")
        else:
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawText(self.width() - 150, 25, "Detection: OFF")
        
        # Total detections
        total = sum(self.detection_stats.values())
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawText(self.width() - 150, 50, f"Total: {total}")


class ESP32Controller(QObject):
    """ESP32 Servo Controller"""
    status_update = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.serial_port = None
        self.connected = False
        
    def connect(self, port: str, baudrate: int = 115200):
        """Connect to ESP32"""
        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            time.sleep(2)  # Wait for ESP32 reset
            
            # Send handshake
            self.send_command({"cmd": "HELLO"})
            response = self.read_response()
            
            if response and response.get("status") == "ready":
                self.connected = True
                self.status_update.emit("Connected")
                return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.status_update.emit(f"Error: {e}")
        return False
    
    def send_command(self, command: dict):
        """Send command to ESP32"""
        if self.serial_port:
            json_cmd = json.dumps(command) + '\n'
            self.serial_port.write(json_cmd.encode())
    
    def read_response(self) -> dict:
        """Read response from ESP32"""
        if self.serial_port:
            try:
                response = self.serial_port.readline().decode().strip()
                if response:
                    return json.loads(response)
            except:
                pass
        return None
    
    def move_servo(self, servo_id: int, angle: int):
        """Move single servo"""
        self.send_command({
            "cmd": "SERVO_MOVE",
            "servo": servo_id,
            "angle": angle
        })
    
    def home(self):
        """Home position"""
        self.send_command({"cmd": "HOME"})
    
    def harvest_sequence(self, x_pos: int, y_pos: int):
        """Execute harvest sequence based on detection position"""
        # Calculate servo angles based on position
        # This is simplified - adjust for your setup
        base_angle = int(90 + (x_pos - 320) * 0.2)
        shoulder_angle = int(90 - (y_pos - 240) * 0.2)
        
        # Limit angles
        base_angle = max(0, min(180, base_angle))
        shoulder_angle = max(0, min(180, shoulder_angle))
        
        # Execute sequence
        self.send_command({
            "cmd": "SERVO_MOVE_ALL",
            "angles": [base_angle, shoulder_angle, 135, 60]
        })
        
        time.sleep(1)
        
        # Grip
        self.send_command({"cmd": "GRIPPER", "action": "grip"})
        
        time.sleep(0.5)
        
        # Move to basket
        self.send_command({
            "cmd": "SERVO_MOVE_ALL",
            "angles": [0, 90, 90, 60]
        })
        
        time.sleep(1)
        
        # Release
        self.send_command({"cmd": "GRIPPER", "action": "open"})
        
        # Home
        self.home()


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.detector = TomatoDetector()
        self.esp32 = ESP32Controller()
        self.camera_thread = None
        self.current_frame = None
        self.auto_harvest = False
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("🍅 Tomato Harvesting System - Real-time Detection")
        self.setGeometry(100, 100, 1200, 700)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { background-color: #2b2b2b; color: #ffffff; }
            QPushButton { 
                background-color: #4CAF50; 
                color: white; 
                border: none; 
                padding: 10px; 
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #45a049; }
            QPushButton:checked { background-color: #f44336; }
            QGroupBox { 
                border: 2px solid #4CAF50; 
                border-radius: 5px; 
                margin-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 10px 0 10px; 
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #555;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #4CAF50;
                width: 20px;
                height: 20px;
                border-radius: 10px;
                margin: -6px 0;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Left: Camera view
        left_panel = QVBoxLayout()
        
        self.camera_widget = CameraWidget()
        left_panel.addWidget(self.camera_widget)
        
        # Camera controls
        cam_controls = QHBoxLayout()
        
        self.start_cam_btn = QPushButton("📷 Start Camera")
        self.start_cam_btn.clicked.connect(self.toggle_camera)
        cam_controls.addWidget(self.start_cam_btn)
        
        self.detection_btn = QPushButton("🔍 Detection: ON")
        self.detection_btn.setCheckable(True)
        self.detection_btn.setChecked(True)
        self.detection_btn.clicked.connect(self.toggle_detection)
        cam_controls.addWidget(self.detection_btn)
        
        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        cam_controls.addWidget(self.capture_btn)
        
        left_panel.addLayout(cam_controls)
        layout.addLayout(left_panel, 2)
        
        # Right: Controls
        right_panel = QVBoxLayout()
        
        # ESP32 Connection
        conn_group = QGroupBox("ESP32 Connection")
        conn_layout = QVBoxLayout()
        
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("Port:"))
        
        self.port_combo = QComboBox()
        self.refresh_ports()
        port_layout.addWidget(self.port_combo)
        
        refresh_btn = QPushButton("🔄")
        refresh_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(refresh_btn)
        
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self.connect_esp32)
        
        self.conn_status = QLabel("⚫ Disconnected")
        
        conn_layout.addLayout(port_layout)
        conn_layout.addWidget(self.connect_btn)
        conn_layout.addWidget(self.conn_status)
        
        conn_group.setLayout(conn_layout)
        right_panel.addWidget(conn_group)
        
        # Servo Control
        servo_group = QGroupBox("Servo Control")
        servo_layout = QGridLayout()
        
        self.servo_sliders = []
        servo_names = ["Base", "Shoulder", "Elbow", "Gripper"]
        
        for i, name in enumerate(servo_names):
            label = QLabel(f"{name}:")
            servo_layout.addWidget(label, i, 0)
            
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 180)
            slider.setValue(90)
            slider.valueChanged.connect(lambda v, sid=i: self.move_servo(sid, v))
            servo_layout.addWidget(slider, i, 1)
            
            value_label = QLabel("90°")
            servo_layout.addWidget(value_label, i, 2)
            
            slider.valueChanged.connect(lambda v, lbl=value_label: lbl.setText(f"{v}°"))
            self.servo_sliders.append(slider)
        
        # Preset buttons
        preset_layout = QGridLayout()
        
        home_btn = QPushButton("🏠 Home")
        home_btn.clicked.connect(self.home_position)
        preset_layout.addWidget(home_btn, 0, 0)
        
        pick_btn = QPushButton("🤏 Pick")
        pick_btn.clicked.connect(self.pick_position)
        preset_layout.addWidget(pick_btn, 0, 1)
        
        place_btn = QPushButton("📦 Place")
        place_btn.clicked.connect(self.place_position)
        preset_layout.addWidget(place_btn, 1, 0)
        
        scan_btn = QPushButton("👁️ Scan")
        scan_btn.clicked.connect(self.scan_position)
        preset_layout.addWidget(scan_btn, 1, 1)
        
        servo_layout.addLayout(preset_layout, 4, 0, 1, 3)
        
        servo_group.setLayout(servo_layout)
        right_panel.addWidget(servo_group)
        
        # Model Loading
        model_group = QGroupBox("AI Model")
        model_layout = QVBoxLayout()
        
        load_model_btn = QPushButton("📁 Load Model (.h5)")
        load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(load_model_btn)
        
        load_labels_btn = QPushButton("📝 Load Labels (.txt)")
        load_labels_btn.clicked.connect(self.load_labels)
        model_layout.addWidget(load_labels_btn)
        
        self.model_status = QLabel("Model: Not loaded")
        model_layout.addWidget(self.model_status)
        
        # Threshold slider
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        thresh_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("50%")
        thresh_layout.addWidget(self.threshold_label)
        
        model_layout.addLayout(thresh_layout)
        
        model_group.setLayout(model_layout)
        right_panel.addWidget(model_group)
        
        # Auto Harvest
        auto_group = QGroupBox("Automation")
        auto_layout = QVBoxLayout()
        
        self.auto_harvest_btn = QPushButton("🤖 Auto Harvest: OFF")
        self.auto_harvest_btn.setCheckable(True)
        self.auto_harvest_btn.clicked.connect(self.toggle_auto_harvest)
        auto_layout.addWidget(self.auto_harvest_btn)
        
        self.harvest_stats = QLabel("Harvested: 0")
        auto_layout.addWidget(self.harvest_stats)
        
        auto_group.setLayout(auto_layout)
        right_panel.addWidget(auto_group)
        
        right_panel.addStretch()
        layout.addLayout(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Connect detector
        self.detector.detection_complete.connect(self.handle_detection)
        
        # Auto harvest timer
        self.harvest_timer = QTimer()
        self.harvest_timer.timeout.connect(self.auto_harvest_check)
        self.harvest_count = 0
    
    def refresh_ports(self):
        """Refresh serial ports"""
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
    
    def connect_esp32(self):
        """Connect to ESP32"""
        port = self.port_combo.currentText()
        if port:
            if self.esp32.connect(port):
                self.conn_status.setText("🟢 Connected")
                self.connect_btn.setText("Disconnect")
            else:
                self.conn_status.setText("🔴 Failed")
    
    def toggle_camera(self):
        """Start/stop camera"""
        if self.camera_thread is None:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.process_frame)
            self.camera_thread.start()
            self.start_cam_btn.setText("⏹️ Stop Camera")
            self.statusBar().showMessage("Camera started")
        else:
            self.camera_thread.stop()
            self.camera_thread = None
            self.start_cam_btn.setText("📷 Start Camera")
            self.statusBar().showMessage("Camera stopped")
    
    def process_frame(self, frame: np.ndarray):
        """Process camera frame"""
        self.current_frame = frame
        
        if self.camera_widget.show_detections:
            # Process with detection
            annotated, detections = self.detector.process_frame(frame)
            self.camera_widget.update_frame(annotated, detections, self.detector.fps)
        else:
            # Show raw frame
            self.camera_widget.update_frame(frame)
    
    def handle_detection(self, detections: List[Detection], fps: float):
        """Handle detection results"""
        self.detector.fps = fps
        
        # Auto harvest if enabled
        if self.auto_harvest and detections and self.esp32.connected:
            # Find best detection (highest confidence ripe tomato)
            ripe_detections = [d for d in detections if d.label == 'ripe']
            if ripe_detections:
                best = max(ripe_detections, key=lambda d: d.confidence)
                self.perform_harvest(best)
    
    def perform_harvest(self, detection: Detection):
        """Execute harvest for detected tomato"""
        center_x = detection.x + detection.width // 2
        center_y = detection.y + detection.height // 2
        
        self.esp32.harvest_sequence(center_x, center_y)
        self.harvest_count += 1
        self.harvest_stats.setText(f"Harvested: {self.harvest_count}")
        
        # Pause auto harvest briefly
        self.harvest_timer.stop()
        QTimer.singleShot(3000, lambda: self.harvest_timer.start(1000))
    
    def toggle_detection(self):
        """Toggle detection display"""
        self.camera_widget.show_detections = not self.camera_widget.show_detections
        if self.camera_widget.show_detections:
            self.detection_btn.setText("🔍 Detection: ON")
        else:
            self.detection_btn.setText("🔍 Detection: OFF")
    
    def toggle_auto_harvest(self):
        """Toggle auto harvest"""
        self.auto_harvest = not self.auto_harvest
        if self.auto_harvest:
            self.auto_harvest_btn.setText("🤖 Auto Harvest: ON")
            self.harvest_timer.start(1000)
        else:
            self.auto_harvest_btn.setText("🤖 Auto Harvest: OFF")
            self.harvest_timer.stop()
    
    def auto_harvest_check(self):
        """Check for harvest opportunities"""
        # Handled in handle_detection
        pass
    
    def load_model(self):
        """Load AI model"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Model", "", "Model Files (*.h5 *.keras);;All Files (*)")
        
        if file_path:
            if self.detector.load_model(file_path):
                self.model_status.setText(f"Model: {Path(file_path).name}")
                QMessageBox.information(self, "Success", "Model loaded!")
            else:
                QMessageBox.critical(self, "Error", "Failed to load model!")
    
    def load_labels(self):
        """Load label file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Labels", "", "Text Files (*.txt);;All Files (*)")
        
        if file_path:
            if self.detector.load_labels(file_path):
                QMessageBox.information(self, "Success", "Labels loaded!")
            else:
                QMessageBox.critical(self, "Error", "Failed to load labels!")
    
    def update_threshold(self, value):
        """Update detection threshold"""
        self.threshold_label.setText(f"{value}%")
        self.detector.confidence_threshold = value / 100.0
    
    def move_servo(self, servo_id: int, angle: int):
        """Move servo"""
        if self.esp32.connected:
            self.esp32.move_servo(servo_id, angle)
    
    def home_position(self):
        """Home position"""
        if self.esp32.connected:
            self.esp32.home()
            for slider in self.servo_sliders:
                slider.setValue(90)
    
    def pick_position(self):
        """Pick position"""
        positions = [90, 45, 135, 30]
        for i, angle in enumerate(positions):
            self.servo_sliders[i].setValue(angle)
    
    def place_position(self):
        """Place position"""
        positions = [0, 90, 90, 120]
        for i, angle in enumerate(positions):
            self.servo_sliders[i].setValue(angle)
    
    def scan_position(self):
        """Scan position"""
        positions = [90, 60, 120, 90]
        for i, angle in enumerate(positions):
            self.servo_sliders[i].setValue(angle)
    
    def capture_image(self):
        """Capture current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.statusBar().showMessage(f"Saved: {filename}")
    
    def closeEvent(self, event):
        """Clean shutdown"""
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    
    # Check dependencies
    if not TF_AVAILABLE:
        QMessageBox.warning(None, "Warning", 
                           "TensorFlow not installed.\nAI classification will be limited.\n"
                           "Install with: pip install tensorflow")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
