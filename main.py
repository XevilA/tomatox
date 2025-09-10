#!/usr/bin/env python3
"""
Advanced Tomato Harvesting Robot Control System for Raspberry Pi 4
ESP32 Servo Control Edition with Real-time Detection
Optimized for Pi Camera with Modern UI/UX Design
Requires PyQt6 - Install with: pip install PyQt6 PyQt6-Charts
"""

import sys
import json
import time
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from collections import deque
from dataclasses import dataclass
import numpy as np
import threading

# PyQt6 imports (required)
from PyQt6.QtCore import (Qt, QThread, QTimer, pyqtSignal, QObject,
                         QPropertyAnimation, QEasingCurve, QRect, QPoint,
                         QRectF, QPointF)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QGroupBox, QGridLayout, QStatusBar, QMessageBox,
                             QSlider, QComboBox, QCheckBox, QTabWidget,
                             QProgressBar, QFrame, QGraphicsDropShadowEffect,
                             QFileDialog, QSpinBox, QDial, QLineEdit)
from PyQt6.QtGui import (QImage, QPixmap, QFont, QPalette, QColor,
                        QLinearGradient, QPainter, QPen, QBrush, QIcon,
                        QPainterPath)
from PyQt6.QtCharts import QChart, QChartView, QLineSeries, QValueAxis

# Pi Camera support
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import FileOutput
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: picamera2 not available. Using OpenCV fallback.")

import cv2
import serial
import serial.tools.list_ports

# TensorFlow imports
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

# ESP32 WiFi control support
try:
    import requests
    WIFI_AVAILABLE = True
except ImportError:
    WIFI_AVAILABLE = False
    print("Warning: requests not available. Install with: pip install requests")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    label: str
    confidence: float
    timestamp: float


@dataclass
class HarvestStats:
    """Statistics for harvesting operations"""
    total_detected: int = 0
    unripe_count: int = 0
    ripe_count: int = 0
    rotten_count: int = 0
    harvest_success: int = 0
    harvest_failed: int = 0
    accuracy_history: deque = None
    detections: List[Detection] = None

    def __post_init__(self):
        if self.accuracy_history is None:
            self.accuracy_history = deque(maxlen=100)
        if self.detections is None:
            self.detections = []


class StyleSheet:
    """Modern stylesheet for the application"""
    DARK_THEME = """
    QMainWindow {
        background-color: #1a1a2e;
    }

    QWidget {
        background-color: #16213e;
        color: #eee;
        font-family: 'Segoe UI', Arial, sans-serif;
    }

    QGroupBox {
        background-color: #0f3460;
        border: 2px solid #e94560;
        border-radius: 10px;
        margin-top: 10px;
        padding-top: 10px;
        font-size: 14px;
        font-weight: bold;
    }

    QGroupBox::title {
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 10px 0 10px;
        color: #e94560;
    }

    QPushButton {
        background-color: #e94560;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 14px;
        font-weight: bold;
        min-height: 40px;
    }

    QPushButton:hover {
        background-color: #c13651;
    }

    QPushButton:pressed {
        background-color: #a02c44;
    }

    QPushButton:checked {
        background-color: #4CAF50;
        color: white;
    }

    QPushButton#primaryButton {
        background-color: #4CAF50;
        font-size: 16px;
        min-height: 50px;
    }

    QPushButton#primaryButton:hover {
        background-color: #45a049;
    }

    QPushButton#dangerButton {
        background-color: #f44336;
    }

    QPushButton#dangerButton:hover {
        background-color: #da190b;
    }

    QLabel {
        color: #eee;
        font-size: 13px;
    }

    QLabel#titleLabel {
        font-size: 24px;
        font-weight: bold;
        color: #e94560;
        padding: 10px;
    }

    QLabel#statsLabel {
        font-size: 16px;
        font-weight: bold;
        color: #4CAF50;
        padding: 5px;
    }

    QTextEdit, QLineEdit {
        background-color: #0f3460;
        color: #eee;
        border: 1px solid #e94560;
        border-radius: 5px;
        padding: 5px;
        font-family: 'Consolas', 'Courier New', monospace;
    }

    QProgressBar {
        border: 2px solid #e94560;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }

    QProgressBar::chunk {
        background-color: #4CAF50;
        border-radius: 3px;
    }

    QSlider::groove:horizontal {
        height: 8px;
        background: #0f3460;
        border-radius: 4px;
    }

    QSlider::handle:horizontal {
        background: #e94560;
        width: 20px;
        height: 20px;
        border-radius: 10px;
        margin: -6px 0;
    }

    QSlider::handle:horizontal:hover {
        background: #c13651;
    }

    QDial {
        background-color: #0f3460;
    }

    QStatusBar {
        background-color: #0f3460;
        color: #eee;
        border-top: 2px solid #e94560;
        font-size: 12px;
    }

    QTabWidget::pane {
        border: 2px solid #e94560;
        background-color: #16213e;
        border-radius: 5px;
    }

    QTabBar::tab {
        background-color: #0f3460;
        color: #eee;
        padding: 10px 20px;
        margin-right: 2px;
        border-top-left-radius: 5px;
        border-top-right-radius: 5px;
    }

    QTabBar::tab:selected {
        background-color: #e94560;
        color: white;
    }

    QComboBox {
        background-color: #0f3460;
        color: #eee;
        border: 2px solid #e94560;
        border-radius: 5px;
        padding: 5px;
        min-height: 30px;
    }

    QSpinBox {
        background-color: #0f3460;
        color: #eee;
        border: 2px solid #e94560;
        border-radius: 5px;
        padding: 5px;
    }
    """


class ESP32ServoController(QObject):
    """ESP32-based servo controller with WiFi and Serial support"""
    status_update = pyqtSignal(str, dict)
    position_update = pyqtSignal(int, int)  # servo_id, position

    def __init__(self):
        super().__init__()
        self.connection_type = "serial"  # "serial" or "wifi"
        self.serial_port = None
        self.esp32_ip = "192.168.1.100"  # Default ESP32 IP
        self.esp32_port = 80
        self.connected = False
        self.servo_positions = [90] * 4  # 4 servos, centered at 90 degrees

    def connect_serial(self, port: str, baudrate: int = 115200):
        """Connect to ESP32 via Serial"""
        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=1.0)
            time.sleep(2)  # Wait for ESP32 to reset
            
            # Send handshake
            self.send_command({"cmd": "HELLO"})
            response = self.read_response()
            
            if response and response.get("status") == "ready":
                self.connected = True
                self.connection_type = "serial"
                logger.info(f"ESP32 connected via serial on {port}")
                self.status_update.emit("connected", {"type": "serial", "port": port})
                return True
        except Exception as e:
            logger.error(f"Serial connection failed: {e}")
            self.status_update.emit("error", {"message": str(e)})
        return False

    def connect_wifi(self, ip: str, port: int = 80):
        """Connect to ESP32 via WiFi"""
        if not WIFI_AVAILABLE:
            logger.error("WiFi connection requires 'requests' module")
            return False
            
        try:
            self.esp32_ip = ip
            self.esp32_port = port
            
            # Test connection
            response = requests.get(f"http://{ip}:{port}/status", timeout=2)
            if response.status_code == 200:
                self.connected = True
                self.connection_type = "wifi"
                logger.info(f"ESP32 connected via WiFi at {ip}:{port}")
                self.status_update.emit("connected", {"type": "wifi", "ip": ip})
                return True
        except Exception as e:
            logger.error(f"WiFi connection failed: {e}")
            self.status_update.emit("error", {"message": str(e)})
        return False

    def send_command(self, command: dict) -> bool:
        """Send command to ESP32"""
        if not self.connected:
            return False
            
        try:
            if self.connection_type == "serial":
                json_cmd = json.dumps(command) + '\n'
                self.serial_port.write(json_cmd.encode('utf-8'))
                return True
            elif self.connection_type == "wifi":
                response = requests.post(
                    f"http://{self.esp32_ip}:{self.esp32_port}/command",
                    json=command,
                    timeout=2
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Send command error: {e}")
            return False

    def read_response(self) -> Optional[dict]:
        """Read response from ESP32"""
        try:
            if self.connection_type == "serial" and self.serial_port:
                response = self.serial_port.readline().decode('utf-8').strip()
                if response:
                    return json.loads(response)
        except Exception as e:
            logger.error(f"Read response error: {e}")
        return None

    def move_servo(self, servo_id: int, angle: int, speed: int = 50):
        """Move a specific servo to angle"""
        if not (0 <= servo_id <= 3):  # Changed from 5 to 3 (0-3 for 4 servos)
            return False
        if not (0 <= angle <= 180):
            return False
            
        command = {
            "cmd": "SERVO_MOVE",
            "servo": servo_id,
            "angle": angle,
            "speed": speed
        }
        
        if self.send_command(command):
            self.servo_positions[servo_id] = angle
            self.position_update.emit(servo_id, angle)
            return True
        return False

    def move_all_servos(self, angles: List[int], speed: int = 50):
        """Move all servos simultaneously"""
        if len(angles) != 4:  # Changed from 6 to 4
            return False
            
        command = {
            "cmd": "SERVO_MOVE_ALL",
            "angles": angles,
            "speed": speed
        }
        
        if self.send_command(command):
            self.servo_positions = angles.copy()
            for i, angle in enumerate(angles):
                self.position_update.emit(i, angle)
            return True
        return False

    def gripper_action(self, action: str):
        """Control gripper (open/close/grip)"""
        command = {
            "cmd": "GRIPPER",
            "action": action
        }
        return self.send_command(command)

    def home_position(self):
        """Move all servos to home position"""
        home_angles = [90, 90, 90, 90, 90, 90]
        return self.move_all_servos(home_angles, speed=30)

    def emergency_stop(self):
        """Emergency stop all servos"""
        command = {"cmd": "EMERGENCY_STOP"}
        return self.send_command(command)


class OptimizedPredictor(QObject):
    """Optimized Model Predictor with real-time detection"""
    prediction_complete = pyqtSignal(str, float, np.ndarray, float, list)  # label, confidence, image, fps, detections

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = None
        self.labels_path = None
        self.classes = ['unripe', 'ripe', 'rotten']
        self.input_shape = (224, 224)
        self.last_inference_time = 0
        self.detection_threshold = 0.5

    def load_model(self, model_path: str):
        """Load TensorFlow/Keras model"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
            
        try:
            self.model_path = model_path
            self.model = keras.models.load_model(model_path)
            
            # Get input shape
            input_shape = self.model.input_shape
            self.input_shape = (input_shape[1], input_shape[2])
            
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def load_labels(self, labels_path: str):
        """Load class labels from text file"""
        try:
            with open(labels_path, 'r') as f:
                self.classes = [line.strip() for line in f.readlines()]
            logger.info(f"Loaded {len(self.classes)} classes from {labels_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return False

    def detect_tomatoes(self, image: np.ndarray) -> List[Detection]:
        """Detect tomatoes in image and return bounding boxes"""
        detections = []
        
        # Simple color-based detection for demonstration
        # In production, use YOLO or similar object detection model
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for tomatoes
        color_ranges = [
            ((35, 50, 50), (85, 255, 255)),   # Green (unripe)
            ((0, 50, 50), (10, 255, 255)),    # Red (ripe)
            ((170, 50, 50), (180, 255, 255)), # Red (ripe) wrap-around
        ]
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Extract ROI and classify
                    roi = image[y:y+h, x:x+w]
                    if roi.size > 0:
                        label, confidence = self.classify_roi(roi)
                        if confidence > self.detection_threshold:
                            detection = Detection(
                                bbox=(x, y, w, h),
                                label=label,
                                confidence=confidence,
                                timestamp=time.time()
                            )
                            detections.append(detection)
        
        return detections

    def classify_roi(self, roi: np.ndarray) -> Tuple[str, float]:
        """Classify a region of interest"""
        if self.model is None:
            return "unknown", 0.0
            
        try:
            # Preprocess ROI
            resized = cv2.resize(roi, self.input_shape)
            normalized = resized.astype(np.float32) / 255.0
            batch = np.expand_dims(normalized, axis=0)
            
            # Predict
            predictions = self.model.predict(batch, verbose=0)[0]
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            label = self.classes[class_idx] if class_idx < len(self.classes) else "unknown"
            
            return label, confidence
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return "error", 0.0

    def predict_with_detection(self, image: np.ndarray) -> Tuple[str, float, float, List[Detection]]:
        """Predict with detection boxes"""
        start_time = time.time()
        
        # Detect tomatoes
        detections = self.detect_tomatoes(image)
        
        # Get overall classification
        if detections:
            # Use detection with highest confidence
            best_detection = max(detections, key=lambda d: d.confidence)
            label = best_detection.label
            confidence = best_detection.confidence
        else:
            # Fallback to whole image classification
            label, confidence = self.classify_roi(image)
        
        # Calculate FPS
        inference_time = time.time() - start_time
        fps = 1.0 / inference_time if inference_time > 0 else 0
        
        self.prediction_complete.emit(label, confidence, image, fps, detections)
        
        return label, confidence, fps, detections


class ModernCameraWidget(QWidget):
    """Modern camera display with real-time detection frames"""

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.current_label = "Waiting..."
        self.current_confidence = 0.0
        self.fps = 0.0
        self.detections = []
        self.show_detections = True
        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #0f3460; border-radius: 15px;")

    def update_frame(self, frame: np.ndarray, label: str, confidence: float, fps: float, detections: List[Detection]):
        """Update display with new frame and detection info"""
        self.current_frame = frame
        self.current_label = label
        self.current_confidence = confidence
        self.fps = fps
        self.detections = detections
        self.update()

    def paintEvent(self, event):
        """Custom paint event for overlay graphics"""
        if self.current_frame is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Convert frame to QImage
        height, width, channel = self.current_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.current_frame.data, width, height,
                        bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

        # Scale image to widget size
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)

        # Calculate scaling factors
        scale_x = scaled_pixmap.width() / width
        scale_y = scaled_pixmap.height() / height

        # Center the image
        x_offset = (self.width() - scaled_pixmap.width()) // 2
        y_offset = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x_offset, y_offset, scaled_pixmap)

        # Draw detection boxes
        if self.show_detections:
            self.draw_detections(painter, x_offset, y_offset, scale_x, scale_y)

        # Draw overlay
        self.draw_overlay(painter, x_offset, y_offset, scaled_pixmap.width(), scaled_pixmap.height())

    def draw_detections(self, painter: QPainter, x_offset: int, y_offset: int, scale_x: float, scale_y: float):
        """Draw detection bounding boxes"""
        colors = {
            'unripe': QColor(76, 175, 80),   # Green
            'ripe': QColor(233, 69, 96),     # Red
            'rotten': QColor(121, 85, 72),   # Brown
            'unknown': QColor(158, 158, 158)  # Gray
        }

        for detection in self.detections:
            x, y, w, h = detection.bbox
            
            # Scale coordinates
            x = int(x * scale_x) + x_offset
            y = int(y * scale_y) + y_offset
            w = int(w * scale_x)
            h = int(h * scale_y)
            
            color = colors.get(detection.label, colors['unknown'])
            
            # Draw bounding box
            pen = QPen(color, 3)
            painter.setPen(pen)
            painter.drawRect(x, y, w, h)
            
            # Draw label background
            label_text = f"{detection.label}: {detection.confidence:.1%}"
            font = QFont("Arial", 10, QFont.Weight.Bold)
            painter.setFont(font)
            
            text_rect = painter.fontMetrics().boundingRect(label_text)
            label_bg = QRect(x, y - text_rect.height() - 5, text_rect.width() + 10, text_rect.height() + 5)
            
            painter.fillRect(label_bg, QColor(0, 0, 0, 180))
            
            # Draw label text
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(label_bg, Qt.AlignmentFlag.AlignCenter, label_text)

    def draw_overlay(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw modern overlay graphics"""
        # Semi-transparent overlay for info
        overlay_color = QColor(0, 0, 0, 100)
        painter.fillRect(x, y, w, 80, overlay_color)
        painter.fillRect(x, y + h - 60, w, 60, overlay_color)

        # Top info bar
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        painter.drawText(x + 20, y + 35, f"Mode: {self.current_label.upper()}")
        painter.drawText(x + 20, y + 60, f"Confidence: {self.current_confidence:.1%}")
        painter.drawText(x + w - 100, y + 35, f"FPS: {self.fps:.1f}")
        
        # Detection count
        if self.detections:
            painter.drawText(x + w - 200, y + 60, f"Detections: {len(self.detections)}")

        # Bottom status bar
        painter.setFont(QFont("Arial", 12))
        painter.drawText(x + 20, y + h - 25, datetime.now().strftime("%H:%M:%S"))
        
        if self.show_detections:
            painter.drawText(x + w - 150, y + h - 25, "Detection: ON")
        else:
            painter.drawText(x + w - 150, y + h - 25, "Detection: OFF")

    def toggle_detections(self):
        """Toggle detection box display"""
        self.show_detections = not self.show_detections
        self.update()


class ServoControlWidget(QWidget):
    """Widget for controlling individual servos"""
    
    def __init__(self, esp32_controller):
        super().__init__()
        self.esp32 = esp32_controller
        self.init_ui()
        
    def init_ui(self):
        """Initialize servo control UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Servo Control Panel")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Connection section
        conn_group = QGroupBox("ESP32 Connection")
        conn_layout = QVBoxLayout()
        
        # Connection type selector
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Connection:"))
        
        self.conn_type_combo = QComboBox()
        self.conn_type_combo.addItems(["Serial", "WiFi"])
        type_layout.addWidget(self.conn_type_combo)
        
        conn_layout.addLayout(type_layout)
        
        # Serial settings
        self.serial_widget = QWidget()
        serial_layout = QHBoxLayout(self.serial_widget)
        
        self.port_combo = QComboBox()
        self.refresh_ports()
        serial_layout.addWidget(QLabel("Port:"))
        serial_layout.addWidget(self.port_combo)
        
        refresh_btn = QPushButton("🔄")
        refresh_btn.clicked.connect(self.refresh_ports)
        serial_layout.addWidget(refresh_btn)
        
        conn_layout.addWidget(self.serial_widget)
        
        # WiFi settings
        self.wifi_widget = QWidget()
        wifi_layout = QHBoxLayout(self.wifi_widget)
        
        wifi_layout.addWidget(QLabel("IP:"))
        self.ip_input = QLineEdit("192.168.1.100")
        wifi_layout.addWidget(self.ip_input)
        
        wifi_layout.addWidget(QLabel("Port:"))
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(80)
        wifi_layout.addWidget(self.port_input)
        
        conn_layout.addWidget(self.wifi_widget)
        self.wifi_widget.hide()
        
        # Connect button
        self.connect_btn = QPushButton("Connect ESP32")
        self.connect_btn.clicked.connect(self.connect_esp32)
        conn_layout.addWidget(self.connect_btn)
        
        # Connection status
        self.conn_status = QLabel("⚫ Disconnected")
        self.conn_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        conn_layout.addWidget(self.conn_status)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        # Servo controls
        servo_group = QGroupBox("Servo Positions")
        servo_layout = QGridLayout()
        
        self.servo_controls = []
        servo_names = ["Base", "Shoulder", "Elbow", "Gripper"]
        
        for i in range(4):  # Changed from 6 to 4
            # Servo name
            name_label = QLabel(f"{servo_names[i]}:")
            servo_layout.addWidget(name_label, i, 0)
            
            # Servo slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 180)
            slider.setValue(90)
            slider.valueChanged.connect(lambda v, sid=i: self.move_servo(sid, v))
            servo_layout.addWidget(slider, i, 1)
            
            # Position label
            pos_label = QLabel("90°")
            pos_label.setMinimumWidth(40)
            servo_layout.addWidget(pos_label, i, 2)
            
            # Dial for fine control
            dial = QDial()
            dial.setRange(0, 180)
            dial.setValue(90)
            dial.setMaximumSize(60, 60)
            dial.valueChanged.connect(lambda v, sid=i: self.move_servo(sid, v))
            servo_layout.addWidget(dial, i, 3)
            
            self.servo_controls.append({
                'slider': slider,
                'label': pos_label,
                'dial': dial
            })
        
        servo_group.setLayout(servo_layout)
        layout.addWidget(servo_group)
        
        # Preset positions
        preset_group = QGroupBox("Preset Positions")
        preset_layout = QGridLayout()
        
        self.home_btn = QPushButton("🏠 Home")
        self.home_btn.clicked.connect(self.home_position)
        preset_layout.addWidget(self.home_btn, 0, 0)
        
        self.pick_btn = QPushButton("🤏 Pick Position")
        self.pick_btn.clicked.connect(self.pick_position)
        preset_layout.addWidget(self.pick_btn, 0, 1)
        
        self.place_btn = QPushButton("📦 Place Position")
        self.place_btn.clicked.connect(self.place_position)
        preset_layout.addWidget(self.place_btn, 1, 0)
        
        self.scan_btn = QPushButton("👁️ Scan Position")
        self.scan_btn.clicked.connect(self.scan_position)
        preset_layout.addWidget(self.scan_btn, 1, 1)
        
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)
        
        # Gripper control
        gripper_group = QGroupBox("Gripper Control")
        gripper_layout = QHBoxLayout()
        
        open_btn = QPushButton("Open")
        open_btn.clicked.connect(lambda: self.esp32.gripper_action("open"))
        gripper_layout.addWidget(open_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(lambda: self.esp32.gripper_action("close"))
        gripper_layout.addWidget(close_btn)
        
        grip_btn = QPushButton("Grip")
        grip_btn.clicked.connect(lambda: self.esp32.gripper_action("grip"))
        gripper_layout.addWidget(grip_btn)
        
        gripper_group.setLayout(gripper_layout)
        layout.addWidget(gripper_group)
        
        # Emergency stop
        stop_btn = QPushButton("🛑 EMERGENCY STOP")
        stop_btn.setObjectName("dangerButton")
        stop_btn.clicked.connect(self.emergency_stop)
        layout.addWidget(stop_btn)
        
        layout.addStretch()
        
        # Connect signals
        self.conn_type_combo.currentTextChanged.connect(self.switch_connection_type)
        self.esp32.status_update.connect(self.update_status)
        self.esp32.position_update.connect(self.update_servo_position)
    
    def switch_connection_type(self, conn_type: str):
        """Switch between serial and WiFi settings"""
        if conn_type == "Serial":
            self.serial_widget.show()
            self.wifi_widget.hide()
        else:
            self.serial_widget.hide()
            self.wifi_widget.show()
    
    def refresh_ports(self):
        """Refresh available serial ports"""
        self.port_combo.clear()
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo.addItems(ports)
    
    def connect_esp32(self):
        """Connect to ESP32"""
        if self.conn_type_combo.currentText() == "Serial":
            port = self.port_combo.currentText()
            if port:
                self.esp32.connect_serial(port)
        else:
            ip = self.ip_input.text()
            port = self.port_input.value()
            self.esp32.connect_wifi(ip, port)
    
    def update_status(self, status: str, data: dict):
        """Update connection status"""
        if status == "connected":
            self.conn_status.setText("🟢 Connected")
            self.connect_btn.setText("Disconnect")
        elif status == "error":
            self.conn_status.setText(f"🔴 Error: {data.get('message', 'Unknown')}")
        else:
            self.conn_status.setText("⚫ Disconnected")
            self.connect_btn.setText("Connect ESP32")
    
    def move_servo(self, servo_id: int, angle: int):
        """Move servo to specified angle"""
        if 0 <= servo_id < 4:  # Changed from 6 to 4
            # Update UI
            self.servo_controls[servo_id]['slider'].setValue(angle)
            self.servo_controls[servo_id]['dial'].setValue(angle)
            self.servo_controls[servo_id]['label'].setText(f"{angle}°")
            
            # Send command to ESP32
            self.esp32.move_servo(servo_id, angle)
    
    def update_servo_position(self, servo_id: int, position: int):
        """Update servo position display"""
        if 0 <= servo_id < 4:  # Changed from 6 to 4
            self.servo_controls[servo_id]['slider'].setValue(position)
            self.servo_controls[servo_id]['dial'].setValue(position)
            self.servo_controls[servo_id]['label'].setText(f"{position}°")
    
    def home_position(self):
        """Move to home position"""
        self.esp32.home_position()
        for i in range(4):  # Changed from 6 to 4
            self.update_servo_position(i, 90)
    
    def pick_position(self):
        """Move to pick position"""
        positions = [90, 45, 135, 30]  # Example pick position for 4 servos
        self.esp32.move_all_servos(positions)
        for i, pos in enumerate(positions):
            self.update_servo_position(i, pos)
    
    def place_position(self):
        """Move to place position"""
        positions = [0, 90, 90, 120]  # Example place position for 4 servos
        self.esp32.move_all_servos(positions)
        for i, pos in enumerate(positions):
            self.update_servo_position(i, pos)
    
    def scan_position(self):
        """Move to scan position"""
        positions = [90, 60, 120, 90]  # Example scan position for 4 servos
        self.esp32.move_all_servos(positions)
        for i, pos in enumerate(positions):
            self.update_servo_position(i, pos)
    
    def emergency_stop(self):
        """Emergency stop all servos"""
        self.esp32.emergency_stop()
        logger.warning("Emergency stop activated!")


class ModelBrowserWidget(QWidget):
    """Widget for browsing and loading models and labels"""
    
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.init_ui()
    
    def init_ui(self):
        """Initialize model browser UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Model & Labels Browser")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Model section
        model_group = QGroupBox("Model File (.h5)")
        model_layout = QVBoxLayout()
        
        # Current model
        current_model_layout = QHBoxLayout()
        current_model_layout.addWidget(QLabel("Current:"))
        self.model_path_label = QLabel("No model loaded")
        self.model_path_label.setWordWrap(True)
        current_model_layout.addWidget(self.model_path_label)
        model_layout.addLayout(current_model_layout)
        
        # Browse button
        browse_model_btn = QPushButton("📁 Browse Model (.h5)")
        browse_model_btn.clicked.connect(self.browse_model)
        model_layout.addWidget(browse_model_btn)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Labels section
        labels_group = QGroupBox("Labels File (.txt)")
        labels_layout = QVBoxLayout()
        
        # Current labels
        current_labels_layout = QHBoxLayout()
        current_labels_layout.addWidget(QLabel("Current:"))
        self.labels_path_label = QLabel("Default: unripe, ripe, rotten")
        self.labels_path_label.setWordWrap(True)
        current_labels_layout.addWidget(self.labels_path_label)
        labels_layout.addLayout(current_labels_layout)
        
        # Browse button
        browse_labels_btn = QPushButton("📁 Browse Labels (.txt)")
        browse_labels_btn.clicked.connect(self.browse_labels)
        labels_layout.addWidget(browse_labels_btn)
        
        # Show loaded labels
        self.labels_list = QTextEdit()
        self.labels_list.setReadOnly(True)
        self.labels_list.setMaximumHeight(100)
        self.labels_list.setPlainText("0: unripe\n1: ripe\n2: rotten")
        labels_layout.addWidget(self.labels_list)
        
        labels_group.setLayout(labels_layout)
        layout.addWidget(labels_group)
        
        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout()
        
        self.model_info = QTextEdit()
        self.model_info.setReadOnly(True)
        self.model_info.setMaximumHeight(150)
        self.model_info.setPlainText("No model loaded")
        info_layout.addWidget(self.model_info)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        # Detection settings
        detection_group = QGroupBox("Detection Settings")
        detection_layout = QVBoxLayout()
        
        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Detection Threshold:"))
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("50%")
        threshold_layout.addWidget(self.threshold_label)
        
        detection_layout.addLayout(threshold_layout)
        detection_group.setLayout(detection_layout)
        layout.addWidget(detection_group)
        
        layout.addStretch()
    
    def browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.h5 *.keras *.pb);;All Files (*.*)"
        )
        
        if file_path:
            if self.predictor.load_model(file_path):
                self.model_path_label.setText(Path(file_path).name)
                self.update_model_info()
                QMessageBox.information(self, "Success", "Model loaded successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to load model!")
    
    def browse_labels(self):
        """Browse for labels file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Labels File",
            "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            if self.predictor.load_labels(file_path):
                self.labels_path_label.setText(Path(file_path).name)
                self.update_labels_display()
                QMessageBox.information(self, "Success", "Labels loaded successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to load labels!")
    
    def update_model_info(self):
        """Update model information display"""
        if self.predictor.model is not None:
            info_text = f"Model Path: {self.predictor.model_path}\n"
            info_text += f"Input Shape: {self.predictor.input_shape}\n"
            info_text += f"Classes: {len(self.predictor.classes)}\n"
            
            if hasattr(self.predictor.model, 'summary'):
                # Get model summary
                import io
                stream = io.StringIO()
                self.predictor.model.summary(print_fn=lambda x: stream.write(x + '\n'))
                summary = stream.getvalue()
                # Show only first few lines
                lines = summary.split('\n')[:5]
                info_text += '\n'.join(lines)
            
            self.model_info.setPlainText(info_text)
    
    def update_labels_display(self):
        """Update labels list display"""
        labels_text = ""
        for i, label in enumerate(self.predictor.classes):
            labels_text += f"{i}: {label}\n"
        self.labels_list.setPlainText(labels_text)
    
    def update_threshold(self, value: int):
        """Update detection threshold"""
        self.threshold_label.setText(f"{value}%")
        self.predictor.detection_threshold = value / 100.0


class PiCameraThread(QThread):
    """Optimized thread for Pi Camera capture"""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.use_pi_camera = PICAMERA_AVAILABLE

    def run(self):
        """Main camera capture loop"""
        self.running = True

        if self.use_pi_camera:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"},
                    buffer_count=1
                )
                self.camera.configure(config)
                self.camera.start()

                while self.running:
                    frame = self.camera.capture_array()
                    # Convert RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.frame_ready.emit(frame)
                    time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"Pi Camera error: {e}")
                self.use_pi_camera = False

        if not self.use_pi_camera:
            # Fallback to USB camera
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            while self.running:
                ret, frame = cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    logger.error("Failed to capture frame")
                    time.sleep(0.1)

            cap.release()

    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.camera and self.use_pi_camera:
            self.camera.stop()
        self.wait()


class AdvancedMainWindow(QMainWindow):
    """Advanced main window with ESP32 control and model browser"""

    def __init__(self):
        super().__init__()
        self.predictor = OptimizedPredictor()
        self.esp32_controller = ESP32ServoController()
        self.camera_thread = None
        self.current_frame = None
        self.auto_harvest = False
        self.stats = HarvestStats()

        self.init_ui()
        self.setup_connections()
        self.start_system()

    def init_ui(self):
        """Initialize the modern UI"""
        self.setWindowTitle("🍅 Advanced Tomato Harvesting System - ESP32 Edition")
        self.setGeometry(100, 100, 1600, 900)

        # Apply dark theme
        self.setStyleSheet(StyleSheet.DARK_THEME)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Left panel - Camera and controls
        left_panel = QVBoxLayout()

        # Camera widget
        self.camera_widget = ModernCameraWidget()
        left_panel.addWidget(self.camera_widget, 3)

        # Quick actions
        quick_actions = QHBoxLayout()

        self.auto_harvest_btn = QPushButton("🤖 Auto Harvest: OFF")
        self.auto_harvest_btn.setObjectName("primaryButton")
        self.auto_harvest_btn.setCheckable(True)
        self.auto_harvest_btn.clicked.connect(self.toggle_auto_harvest)

        self.detection_btn = QPushButton("📦 Detection: ON")
        self.detection_btn.setCheckable(True)
        self.detection_btn.setChecked(True)
        self.detection_btn.clicked.connect(self.toggle_detection)

        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.clicked.connect(self.capture_image)

        quick_actions.addWidget(self.auto_harvest_btn)
        quick_actions.addWidget(self.detection_btn)
        quick_actions.addWidget(self.capture_btn)

        left_panel.addLayout(quick_actions)
        main_layout.addLayout(left_panel, 2)

        # Right panel - Tabbed interface
        right_panel = QTabWidget()
        right_panel.setTabPosition(QTabWidget.TabPosition.North)

        # Servo control tab
        self.servo_widget = ServoControlWidget(self.esp32_controller)
        right_panel.addTab(self.servo_widget, "🎮 Servo Control")

        # Model browser tab
        self.model_browser = ModelBrowserWidget(self.predictor)
        right_panel.addTab(self.model_browser, "🧠 Model Browser")

        # Statistics tab
        self.stats_widget = StatisticsWidget()
        right_panel.addTab(self.stats_widget, "📊 Statistics")

        # System log tab
        self.log_widget = self.create_log_widget()
        right_panel.addTab(self.log_widget, "📝 System Log")

        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.create_status_bar()

        # Auto harvest timer
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.auto_harvest_cycle)

    def create_log_widget(self) -> QWidget:
        """Create system log widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(400)

        # Log controls
        controls = QHBoxLayout()

        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)

        export_btn = QPushButton("Export Log")
        export_btn.clicked.connect(self.export_log)

        controls.addWidget(clear_btn)
        controls.addWidget(export_btn)
        controls.addStretch()

        layout.addWidget(self.log_text)
        layout.addLayout(controls)
        layout.addStretch()

        return widget

    def create_status_bar(self):
        """Create enhanced status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # System status
        self.system_status = QLabel("🟢 System Ready")
        self.status_bar.addWidget(self.system_status)

        # Camera status
        self.camera_status = QLabel("📷 Camera: Initializing...")
        self.status_bar.addWidget(self.camera_status)

        # Model status
        self.model_status = QLabel("🤖 Model: Not loaded")
        self.status_bar.addWidget(self.model_status)

        # ESP32 status
        self.esp32_status = QLabel("🔌 ESP32: Disconnected")
        self.status_bar.addPermanentWidget(self.esp32_status)

    def setup_connections(self):
        """Setup all signal/slot connections"""
        # ESP32 controller signals
        self.esp32_controller.status_update.connect(self.handle_esp32_status)

        # Predictor signals
        self.predictor.prediction_complete.connect(self.handle_prediction)

    def start_system(self):
        """Start all system components"""
        # Start camera
        self.camera_thread = PiCameraThread()
        self.camera_thread.frame_ready.connect(self.process_frame)
        self.camera_thread.start()

        # Update status
        self.log_message("System", "Tomato Harvesting System started")
        self.camera_status.setText("📷 Camera: Active")

    def process_frame(self, frame: np.ndarray):
        """Process camera frame"""
        self.current_frame = frame

        # Run prediction with detection
        label, confidence, fps, detections = self.predictor.predict_with_detection(frame)

        # Update camera widget
        self.camera_widget.update_frame(frame, label, confidence, fps, detections)

        # Update statistics
        if label != "unknown" and confidence > 0.5:
            self.stats_widget.update_stats(label, confidence)

    def handle_prediction(self, label: str, confidence: float, image: np.ndarray, fps: float, detections: List[Detection]):
        """Handle prediction results"""
        # Auto harvest if enabled
        if self.auto_harvest and confidence >= self.predictor.detection_threshold:
            if detections:
                # Use best detection for harvesting
                best_detection = max(detections, key=lambda d: d.confidence)
                self.perform_harvest(best_detection)

    def handle_esp32_status(self, status: str, data: dict):
        """Handle ESP32 status updates"""
        if status == "connected":
            self.esp32_status.setText("🔌 ESP32: Connected")
            self.log_message("ESP32", f"Connected via {data.get('type', 'unknown')}")
        elif status == "error":
            self.esp32_status.setText("🔌 ESP32: Error")
            self.log_message("ESP32", f"Error: {data.get('message', 'Unknown')}")
        else:
            self.esp32_status.setText("🔌 ESP32: Disconnected")

    def perform_harvest(self, detection: Detection):
        """Perform harvest action for detected tomato"""
        # Calculate servo positions based on detection location
        x, y, w, h = detection.bbox
        
        # Simple mapping (customize based on your setup)
        base_angle = int(90 + (x - 320) * 0.1)  # Center at 320 pixels
        shoulder_angle = int(90 - (y - 240) * 0.1)  # Center at 240 pixels
        
        # Move to position (4 servos)
        positions = [base_angle, shoulder_angle, 135, 30]
        self.esp32_controller.move_all_servos(positions, speed=30)
        
        # Grip
        time.sleep(1)
        self.esp32_controller.gripper_action("grip")
        
        # Move to place position (4 servos)
        time.sleep(0.5)
        place_positions = [0, 90, 90, 30]
        self.esp32_controller.move_all_servos(place_positions, speed=30)
        
        # Release
        time.sleep(1)
        self.esp32_controller.gripper_action("open")
        
        # Return to home
        self.esp32_controller.home_position()
        
        # Log harvest
        self.log_message("Harvest", f"Harvested {detection.label} tomato at ({x}, {y})")
        self.stats_widget.update_harvest_result(True)

    def toggle_auto_harvest(self):
        """Toggle automatic harvesting"""
        self.auto_harvest = not self.auto_harvest

        if self.auto_harvest:
            self.auto_harvest_btn.setText("🤖 Auto Harvest: ON")
            self.auto_timer.start(2000)  # Check every 2 seconds
            self.log_message("System", "Auto harvest enabled")
        else:
            self.auto_harvest_btn.setText("🤖 Auto Harvest: OFF")
            self.auto_timer.stop()
            self.log_message("System", "Auto harvest disabled")

    def toggle_detection(self):
        """Toggle detection display"""
        self.camera_widget.toggle_detections()
        if self.camera_widget.show_detections:
            self.detection_btn.setText("📦 Detection: ON")
        else:
            self.detection_btn.setText("📦 Detection: OFF")

    def auto_harvest_cycle(self):
        """Automatic harvest cycle"""
        # This is handled in handle_prediction when auto_harvest is True
        pass

    def capture_image(self):
        """Capture current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.log_message("System", f"Image saved: {filename}")

    def export_log(self):
        """Export system log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_log_{timestamp}.txt"

        try:
            with open(filename, 'w') as f:
                f.write(self.log_text.toPlainText())

            QMessageBox.information(self, "Export Success",
                                  f"Log exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                               f"Failed to export log: {e}")

    def log_message(self, source: str, message: str):
        """Add message to system log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {source}: {message}"
        self.log_text.append(log_entry)

        # Auto-scroll
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """Handle application close"""
        # Stop auto harvest
        self.auto_harvest = False
        self.auto_timer.stop()

        # Stop camera
        if self.camera_thread:
            self.camera_thread.stop()

        # Disconnect ESP32
        if self.esp32_controller.connected:
            self.esp32_controller.emergency_stop()

        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style

    # Set application icon
    app.setApplicationName("Tomato Harvesting System - ESP32")
    app.setOrganizationName("Smart Agriculture")

    # Check dependencies
    if not TF_AVAILABLE:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setText("TensorFlow not found!")
        msg.setInformativeText("Install TensorFlow for AI predictions:\npip install tensorflow")
        msg.setWindowTitle("Dependency Warning")
        msg.exec()

    if not PICAMERA_AVAILABLE:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText("Pi Camera library not found")
        msg.setInformativeText("Using USB camera fallback.\nFor Pi Camera: pip install picamera2")
        msg.setWindowTitle("Camera Info")
        msg.exec()

    # Create and show main window
    window = AdvancedMainWindow()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
