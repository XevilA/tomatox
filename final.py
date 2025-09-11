#!/usr/bin/env python3
"""
Complete Advanced Tomato Harvesting Robot Control System for Raspberry Pi 4
Enhanced with Detection Frames, File Browser, and Robot Arm Integration
Requires PyQt6 - Install with: pip install PyQt6 PyQt6-Charts
"""

import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import deque
from dataclasses import dataclass
import numpy as np

# PyQt6 imports (required)
from PyQt6.QtCore import (Qt, QThread, QTimer, pyqtSignal, QObject,
                         QPropertyAnimation, QEasingCurve, QRect, QPoint)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QGroupBox, QGridLayout, QStatusBar, QMessageBox,
                             QSlider, QComboBox, QCheckBox, QTabWidget,
                             QProgressBar, QFrame, QGraphicsDropShadowEffect,
                             QFileDialog, QLineEdit, QSpinBox)
from PyQt6.QtGui import (QImage, QPixmap, QFont, QPalette, QColor,
                        QLinearGradient, QPainter, QPen, QBrush, QIcon)
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Detection result with bounding box and metadata"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    timestamp: datetime
    center_point: Tuple[int, int]


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

    def __post_init__(self):
        if self.accuracy_history is None:
            self.accuracy_history = deque(maxlen=100)


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

    QPushButton#browseButton {
        background-color: #2196F3;
        min-height: 30px;
        padding: 5px 15px;
        font-size: 12px;
    }

    QPushButton#browseButton:hover {
        background-color: #1976D2;
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

    QLabel#detectionLabel {
        font-size: 20px;
        font-weight: bold;
        padding: 12px;
        border-radius: 8px;
        margin: 3px;
        text-align: center;
    }

    QTextEdit {
        background-color: #0f3460;
        color: #eee;
        border: 1px solid #e94560;
        border-radius: 5px;
        padding: 5px;
        font-family: 'Consolas', 'Courier New', monospace;
    }

    QLineEdit {
        background-color: #0f3460;
        color: #eee;
        border: 2px solid #e94560;
        border-radius: 5px;
        padding: 8px;
        font-size: 13px;
    }

    QLineEdit:focus {
        border-color: #4CAF50;
    }

    QProgressBar {
        border: 2px solid #e94560;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        min-height: 20px;
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

    QComboBox::drop-down {
        border: none;
    }

    QComboBox::down-arrow {
        image: none;
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid #e94560;
        margin-right: 5px;
    }

    QSpinBox {
        background-color: #0f3460;
        color: #eee;
        border: 2px solid #e94560;
        border-radius: 5px;
        padding: 5px;
        min-height: 20px;
    }
    """


class OptimizedPredictor(QObject):
    """Enhanced Model Predictor with advanced detection visualization"""
    prediction_complete = pyqtSignal(str, float, np.ndarray, float, tuple)  # label, confidence, image, fps, bbox

    def __init__(self, model_path: str = 'tomato.h5', labels_path: str = 'labels.txt'):
        super().__init__()
        self.model = None
        self.model_path = model_path
        self.labels_path = labels_path
        self.classes = ['unripe', 'ripe', 'rotten']
        self.input_shape = (224, 224)
        self.preprocessing_cache = {}
        self.last_inference_time = 0

        # Enhanced color ranges for better detection (HSV)
        self.color_ranges = {
            'unripe': [(35, 40, 40), (85, 255, 255)],    # Green tomatoes
            'ripe': [(0, 50, 50), (15, 255, 255)],       # Red tomatoes
            'rotten': [(10, 30, 20), (30, 200, 150)]     # Brown/Dark tomatoes
        }

        self.load_labels()
        if TF_AVAILABLE:
            self.load_model()

    def load_labels(self):
        """Load class labels from file"""
        try:
            if Path(self.labels_path).exists():
                with open(self.labels_path, 'r', encoding='utf-8') as f:
                    self.classes = [line.strip() for line in f.readlines() if line.strip()]
                logger.info(f"Loaded {len(self.classes)} classes from {self.labels_path}")
            else:
                logger.warning(f"Labels file not found: {self.labels_path}")
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")

    def set_model_path(self, model_path: str):
        """Set new model path and reload"""
        self.model_path = model_path
        self.load_model()

    def set_labels_path(self, labels_path: str):
        """Set new labels path and reload"""
        self.labels_path = labels_path
        self.load_labels()

    def load_model(self):
        """Load and optimize the Keras model"""
        if not self.model_path or not Path(self.model_path).exists():
            logger.error(f"Model file not found: {self.model_path}")
            return False

        try:
            # Load the original Keras model
            self.model = keras.models.load_model(self.model_path)
            
            # Try to optimize with TFLite if possible
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                tflite_model = converter.convert()
                
                self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
                self.interpreter.allocate_tensors()

                input_details = self.interpreter.get_input_details()
                self.input_shape = input_details[0]['shape'][1:3]
                logger.info("Model optimized with TFLite")
            except Exception as e:
                logger.warning(f"TFLite optimization failed: {e}, using regular model")
                # Fallback to regular model
                input_shape = self.model.input_shape
                self.input_shape = (input_shape[1], input_shape[2])

            logger.info(f"Model successfully loaded from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            return False

    def detect_tomato_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Advanced tomato region detection using color and contour analysis"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for all tomato colors
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for class_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Enhanced morphological operations
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour that looks like a tomato
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Minimum area threshold
                    # Check aspect ratio to filter out non-tomato shapes
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.5 <= aspect_ratio <= 2.0:  # Reasonable aspect ratio for tomatoes
                        valid_contours.append((contour, area))
            
            if valid_contours:
                # Get the largest valid contour
                largest_contour = max(valid_contours, key=lambda x: x[1])[0]
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                return (x, y, w, h)
        
        # Default to center region if no tomato detected
        h, w = image.shape[:2]
        return (w//4, h//4, w//2, h//2)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with multiple techniques"""
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # Enhance contrast using CLAHE in LAB color space
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Resize for model with high-quality interpolation
        resized = cv2.resize(enhanced, self.input_shape, interpolation=cv2.INTER_LANCZOS4)

        # Normalize to [0, 1] range
        normalized = resized.astype(np.float32) / 255.0

        return np.expand_dims(normalized, axis=0)

    def quick_color_analysis(self, image: np.ndarray) -> Dict[str, float]:
        """Advanced color analysis for pre-classification"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        total_pixels = image.shape[0] * image.shape[1]
        
        color_scores = {}
        for class_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = cv2.countNonZero(mask) / total_pixels
            color_scores[class_name] = ratio

        return color_scores

    def predict(self, image: np.ndarray) -> Tuple[str, float, float, Tuple[int, int, int, int]]:
        """Enhanced prediction with comprehensive analysis"""
        start_time = time.time()

        if self.model is None:
            bbox = self.detect_tomato_region(image)
            return "unknown", 0.0, 0.0, bbox

        try:
            # Detect tomato region
            bbox = self.detect_tomato_region(image)
            
            # Extract region of interest
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w] if bbox else image
            
            if roi.size == 0:
                roi = image

            # Color analysis for confidence boosting
            color_scores = self.quick_color_analysis(roi)
            best_color_class = max(color_scores, key=color_scores.get)
            max_color_score = color_scores[best_color_class]

            # Preprocess for model
            preprocessed = self.preprocess_image(roi)

            # Model inference
            if hasattr(self, 'interpreter'):
                # TFLite inference
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()

                self.interpreter.set_tensor(input_details[0]['index'], preprocessed)
                self.interpreter.invoke()

                predictions = self.interpreter.get_tensor(output_details[0]['index'])[0]
            else:
                # Regular Keras model inference
                predictions = self.model.predict(preprocessed, verbose=0)[0]

            # Get prediction results
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            if class_idx < len(self.classes):
                label = self.classes[class_idx]
            else:
                label = "unknown"

            # Enhance confidence using color analysis
            if best_color_class == label and max_color_score > 0.3:
                confidence = min(confidence * (1 + max_color_score * 0.2), 1.0)

            # Calculate FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0

            # Emit signal with results
            self.prediction_complete.emit(label, confidence, image, fps, bbox)

            return label, confidence, fps, bbox

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            bbox = self.detect_tomato_region(image)
            return "error", 0.0, 0.0, bbox


class PiCameraThread(QThread):
    """Enhanced thread for Pi Camera capture with better performance"""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.use_pi_camera = PICAMERA_AVAILABLE
        self.target_fps = 30
        self.resolution = (640, 480)
        self.brightness = 0.5
        self.contrast = 1.0

    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        self.resolution = (width, height)
        if self.camera and self.use_pi_camera:
            self.restart_camera()

    def set_fps(self, fps: int):
        """Set target FPS"""
        self.target_fps = max(10, min(60, fps))

    def restart_camera(self):
        """Restart camera with new settings"""
        if self.camera and self.use_pi_camera:
            try:
                self.camera.stop()
                time.sleep(0.1)
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    buffer_count=2
                )
                self.camera.configure(config)
                self.camera.start()
            except Exception as e:
                logger.error(f"Failed to restart camera: {e}")

    def run(self):
        """Enhanced camera capture loop"""
        self.running = True
        frame_time = 1.0 / self.target_fps

        if self.use_pi_camera:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    buffer_count=2
                )
                self.camera.configure(config)
                
                # Set camera controls
                controls = {
                    "Brightness": self.brightness,
                    "Contrast": self.contrast,
                    "AwbEnable": True,
                    "AeEnable": True
                }
                self.camera.set_controls(controls)
                
                self.camera.start()
                logger.info(f"Pi Camera started at {self.resolution} @ {self.target_fps}fps")

                while self.running:
                    frame_start = time.time()
                    try:
                        frame = self.camera.capture_array()
                        # Convert RGB to BGR for OpenCV
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        self.frame_ready.emit(frame)
                    except Exception as e:
                        logger.error(f"Frame capture error: {e}")
                        continue
                    
                    # Maintain target FPS
                    elapsed = time.time() - frame_start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)

            except Exception as e:
                logger.error(f"Pi Camera initialization error: {e}")
                self.use_pi_camera = False

        if not self.use_pi_camera:
            # Enhanced USB camera fallback
            for camera_id in range(3):  # Try multiple camera indices
                cap = cv2.VideoCapture(camera_id)
                if cap.isOpened():
                    break
            else:
                logger.error("No camera found")
                return

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            logger.info(f"USB Camera started at {self.resolution} @ {self.target_fps}fps")

            while self.running:
                frame_start = time.time()
                ret, frame = cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    logger.error("Failed to capture frame from USB camera")
                    time.sleep(0.1)
                    continue
                
                # Maintain target FPS
                elapsed = time.time() - frame_start
                if elapsed < frame_time:
                    time.sleep(frame_time - elapsed)

            cap.release()

    def stop(self):
        """Stop camera capture"""
        self.running = False
        if self.camera and self.use_pi_camera:
            try:
                self.camera.stop()
            except:
                pass
        self.wait()


class EnhancedRobotController(QObject):
    """Advanced robot arm controller with comprehensive features"""
    status_update = pyqtSignal(int, str, dict)  # arm_id, status, data
    harvest_complete = pyqtSignal(int, str, bool)  # arm_id, tomato_type, success

    def __init__(self):
        super().__init__()
        self.connections = {}
        self.ports = [f'/dev/ttyUSB{i}' for i in range(4)]
        self.alt_ports = [f'/dev/ttyACM{i}' for i in range(4)]
        self.baudrate = 115200
        self.timeout = 2.0
        self.arm_status = {i: "disconnected" for i in range(4)}
        self.harvest_queue = deque()
        self.command_history = deque(maxlen=100)

    def disconnect_all(self):
        """Safely disconnect all arms"""
        for arm_id, connection in list(self.connections.items()):
            try:
                if connection and connection.is_open:
                    connection.close()
            except Exception as e:
                logger.error(f"Error disconnecting arm {arm_id}: {e}")
        
        self.connections.clear()
        for i in range(4):
            self.arm_status[i] = "disconnected"
            self.status_update.emit(i, "disconnected", {})

    def connect_all(self):
        """Enhanced connection with better port detection"""
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
        logger.info(f"Available ports: {available_ports}")

        # Try primary ports first
        for i, port in enumerate(self.ports):
            if port in available_ports:
                self.connect_arm(i, port)
            elif self.alt_ports[i] in available_ports:
                self.connect_arm(i, self.alt_ports[i])

        # Try any remaining ports for unconnected arms
        connected_arms = set(self.connections.keys())
        remaining_ports = [p for p in available_ports if 'tty' in p]
        
        for i in range(4):
            if i not in connected_arms and remaining_ports:
                port = remaining_ports.pop(0)
                self.connect_arm(i, port)

    def connect_arm(self, arm_id: int, port: str):
        """Enhanced connection with retry logic"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Close any existing connection
                if arm_id in self.connections:
                    try:
                        self.connections[arm_id].close()
                    except:
                        pass

                ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
                time.sleep(2)  # Wait for Arduino reset

                # Enhanced handshake
                for _ in range(3):
                    ser.write(b'{"cmd":"PING","arm_id":' + str(arm_id).encode() + b'}\n')
                    time.sleep(0.5)
                    
                    if ser.in_waiting:
                        response = ser.readline().decode('utf-8').strip()
                        if response:
                            try:
                                resp_data = json.loads(response)
                                if resp_data.get("status") == "ready":
                                    self.connections[arm_id] = ser
                                    self.arm_status[arm_id] = "ready"
                                    logger.info(f"Arm {arm_id} connected on {port}")
                                    self.status_update.emit(arm_id, "connected", {"port": port, "attempt": attempt + 1})
                                    return
                            except json.JSONDecodeError:
                                continue

                ser.close()
                raise Exception(f"No valid handshake response after 3 attempts")

            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to connect arm {arm_id} after {max_retries} attempts: {e}")
                    self.arm_status[arm_id] = "error"
                    self.status_update.emit(arm_id, "error", {"message": str(e), "port": port})
                else:
                    time.sleep(1)  # Wait before retry

    def smart_harvest(self, arm_id: int, tomato_class: str, detection_result: DetectionResult):
        """Advanced harvesting with position and class awareness"""
        if arm_id not in self.connections:
            return False

        # Enhanced commands based on tomato type and position
        base_commands = {
            'unripe': {
                "cmd": "SMART_HARVEST",
                "type": "unripe",
                "force": 75,  # Medium-firm grip
                "speed": "medium",
                "approach_angle": "gentle"
            },
            'ripe': {
                "cmd": "SMART_HARVEST",
                "type": "ripe",
                "force": 50,  # Gentle grip
                "speed": "slow",
                "approach_angle": "careful"
            },
            'rotten': {
                "cmd": "SMART_REMOVE",
                "type": "rotten",
                "force": 30,  # Very gentle
                "speed": "very_slow",
                "approach_angle": "minimal"
            }
        }

        if tomato_class not in base_commands:
            return False

        command = base_commands[tomato_class].copy()
        
        # Add position information
        command.update({
            "position": {
                "x": detection_result.center_point[0],
                "y": detection_result.center_point[1],
                "bbox": detection_result.bbox,
                "confidence": detection_result.confidence
            },
            "timestamp": detection_result.timestamp.isoformat(),
            "arm_id": arm_id
        })

        success = self.send_command(arm_id, command)
        self.harvest_complete.emit(arm_id, tomato_class, success)
        return success

    def send_command(self, arm_id: int, command: Dict) -> bool:
        """Enhanced command sending with comprehensive error handling"""
        if arm_id not in self.connections or self.arm_status[arm_id] != "ready":
            logger.warning(f"Arm {arm_id} not ready for commands")
            return False

        try:
            ser = self.connections[arm_id]
            json_cmd = json.dumps(command) + '\n'
            
            # Record command in history
            self.command_history.append({
                "arm_id": arm_id,
                "command": command,
                "timestamp": datetime.now().isoformat()
            })

            # Send command
            ser.write(json_cmd.encode('utf-8'))
            self.arm_status[arm_id] = "busy"
            self.status_update.emit(arm_id, "command_sent", {"command": command["cmd"]})

            # Wait for response with timeout
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                if ser.in_waiting:
                    response = ser.readline().decode('utf-8').strip()
                    if response:
                        try:
                            resp_data = json.loads(response)
                            self.arm_status[arm_id] = "ready"
                            
                            success = resp_data.get("status") == "success"
                            self.status_update.emit(arm_id, "response", resp_data)
                            return success
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON response from arm {arm_id}: {response}")
                            continue
                time.sleep(0.1)

            # Timeout handling
            self.arm_status[arm_id] = "error"
            self.status_update.emit(arm_id, "timeout", {"command": command["cmd"]})
            return False

        except Exception as e:
            logger.error(f"Command error for arm {arm_id}: {e}")
            self.arm_status[arm_id] = "error"
            self.status_update.emit(arm_id, "error", {"message": str(e)})
            return False

    def emergency_stop_all(self):
        """Emergency stop all arms"""
        for arm_id in list(self.connections.keys()):
            try:
                self.send_command(arm_id, {"cmd": "EMERGENCY_STOP", "priority": "high"})
            except:
                pass

    def home_all_arms(self):
        """Home all connected arms"""
        for arm_id in list(self.connections.keys()):
            if self.arm_status[arm_id] == "ready":
                self.send_command(arm_id, {"cmd": "HOME", "speed": "medium"})


class ModernCameraWidget(QWidget):
    """Advanced camera display widget with professional detection visualization"""

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.current_label = "Waiting..."
        self.current_confidence = 0.0
        self.fps = 0.0
        self.detection_bbox = None
        self.detection_history = deque(maxlen=15)
        self.animation_phase = 0.0
        self.pulse_intensity = 0.8
        
        self.init_ui()

        # Enhanced animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 FPS animation

    def init_ui(self):
        """Initialize enhanced UI components"""
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            background-color: #0f3460; 
            border-radius: 15px; 
            border: 3px solid #e94560;
            margin: 5px;
        """)

    def update_frame(self, frame: np.ndarray, label: str, confidence: float, fps: float, bbox: Tuple[int, int, int, int]):
        """Update display with enhanced detection info"""
        self.current_frame = frame
        self.current_label = label
        self.current_confidence = confidence
        self.fps = fps
        self.detection_bbox = bbox
        
        # Enhanced detection history
        detection_data = {
            'label': label,
            'confidence': confidence,
            'bbox': bbox,
            'timestamp': time.time(),
            'center': (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2) if bbox else None
        }
        self.detection_history.append(detection_data)
        
        self.update()

    def update_animation(self):
        """Enhanced animation with multiple effects"""
        self.animation_phase = (self.animation_phase + 0.08) % (2 * np.pi)
        self.pulse_intensity = 0.6 + 0.4 * abs(np.sin(self.animation_phase))
        
        if self.current_frame is not None and self.current_confidence > 0.3:
            self.update()

    def paintEvent(self, event):
        """Professional paint event with advanced visualization"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self.current_frame is None:
            # Professional loading screen
            painter.fillRect(self.rect(), QColor(15, 52, 96))
            painter.setPen(QPen(QColor(233, 69, 96), 3))
            painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "🎥 Initializing Camera System...")
            return

        # Convert and scale image
        height, width, channel = self.current_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.current_frame.data, width, height,
                        bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

        pixmap = QPixmap.fromImage(q_image)
        widget_size = self.size()
        scaled_pixmap = pixmap.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)

        # Center the image
        img_x = (self.width() - scaled_pixmap.width()) // 2
        img_y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(img_x, img_y, scaled_pixmap)

        # Calculate scaling factors
        scale_x = scaled_pixmap.width() / width
        scale_y = scaled_pixmap.height() / height

        # Draw professional overlays
        self.draw_professional_overlay(painter, img_x, img_y, scaled_pixmap.width(), 
                                     scaled_pixmap.height(), scale_x, scale_y)

    def draw_professional_overlay(self, painter: QPainter, img_x: int, img_y: int, 
                                img_w: int, img_h: int, scale_x: float, scale_y: float):
        """Draw professional-grade detection overlay"""
        
        # Enhanced color scheme
        colors = {
            'unripe': QColor(76, 175, 80, 200),    # Green
            'ripe': QColor(233, 69, 96, 200),      # Red  
            'rotten': QColor(121, 85, 72, 200),    # Brown
            'unknown': QColor(158, 158, 158, 100), # Gray
            'error': QColor(255, 152, 0, 200)      # Orange
        }

        current_color = colors.get(self.current_label, colors['unknown'])
        
        # Professional detection frame
        if self.detection_bbox and self.current_confidence > 0.2:
            bbox_x, bbox_y, bbox_w, bbox_h = self.detection_bbox
            
            # Scale to display coordinates
            scaled_x = int(img_x + bbox_x * scale_x)
            scaled_y = int(img_y + bbox_y * scale_y)
            scaled_w = int(bbox_w * scale_x)
            scaled_h = int(bbox_h * scale_y)
            
            # Multi-layer detection visualization
            self.draw_detection_frame(painter, scaled_x, scaled_y, scaled_w, scaled_h, current_color)
            self.draw_detection_info(painter, scaled_x, scaled_y, scaled_w, scaled_h, current_color)

        # Professional HUD overlay
        self.draw_professional_hud(painter, img_x, img_y, img_w, img_h, current_color)
        
        # Detection trail
        self.draw_detection_trail(painter, img_x, img_y, scale_x, scale_y, current_color)

    def draw_detection_frame(self, painter: QPainter, x: int, y: int, w: int, h: int, color: QColor):
        """Draw advanced detection frame with multiple visual elements"""
        
        # Animated main frame
        pulse_alpha = int(255 * self.pulse_intensity)
        animated_color = QColor(color)
        animated_color.setAlpha(pulse_alpha)
        
        # Main detection box with thickness based on confidence
        thickness = max(2, int(4 * self.current_confidence))
        painter.setPen(QPen(animated_color, thickness))
        painter.drawRect(x, y, w, h)
        
        # Corner brackets for professional look
        corner_length = min(25, min(w, h) // 4)
        bracket_color = QColor(color)
        bracket_color.setAlpha(255)
        painter.setPen(QPen(bracket_color, 3))
        
        # Draw corner brackets
        corners = [
            (x, y, x + corner_length, y, x, y + corner_length),  # Top-left
            (x + w - corner_length, y, x + w, y, x + w, y + corner_length),  # Top-right
            (x, y + h - corner_length, x, y + h, x + corner_length, y + h),  # Bottom-left
            (x + w - corner_length, y + h, x + w, y + h, x + w, y + h - corner_length)  # Bottom-right
        ]
        
        for corner in corners:
            painter.drawLine(corner[0], corner[1], corner[2], corner[3])
            painter.drawLine(corner[2], corner[3], corner[4], corner[5])
        
        # Center crosshair
        center_x, center_y = x + w//2, y + h//2
        cross_size = 12
        painter.setPen(QPen(bracket_color, 2))
        painter.drawLine(center_x - cross_size, center_y, center_x + cross_size, center_y)
        painter.drawLine(center_x, center_y - cross_size, center_x, center_y + cross_size)
        
        # Confidence-based glow effect
        if self.current_confidence > 0.7:
            glow_color = QColor(color)
            glow_color.setAlpha(50)
            painter.setPen(QPen(glow_color, 8))
            painter.drawRect(x - 2, y - 2, w + 4, h + 4)

    def draw_detection_info(self, painter: QPainter, x: int, y: int, w: int, h: int, color: QColor):
        """Draw detection information near the bounding box"""
        
        # Info panel positioning
        info_x = x
        info_y = y - 35 if y > 50 else y + h + 5
        
        # Background for text
        info_bg = QColor(0, 0, 0, 150)
        painter.fillRect(info_x, info_y, 200, 30, info_bg)
        
        # Detection text
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        info_text = f"{self.current_label.upper()} {self.current_confidence:.0%}"
        painter.drawText(info_x + 5, info_y + 20, info_text)
        
        # Confidence indicator
        conf_bar_w = 150
        conf_bar_h = 4
        conf_x = info_x + 5
        conf_y = info_y + 25
        
        # Background bar
        painter.fillRect(conf_x, conf_y, conf_bar_w, conf_bar_h, QColor(50, 50, 50))
        
        # Confidence fill
        fill_w = int(conf_bar_w * self.current_confidence)
        painter.fillRect(conf_x, conf_y, fill_w, conf_bar_h, color)

    def draw_professional_hud(self, painter: QPainter, x: int, y: int, w: int, h: int, color: QColor):
        """Draw professional heads-up display"""
        
        # Top status bar with gradient
        top_gradient = QLinearGradient(0, y, 0, y + 80)
        top_gradient.setColorAt(0, QColor(0, 0, 0, 180))
        top_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(x, y, w, 80, QBrush(top_gradient))

        # Bottom status bar
        bottom_gradient = QLinearGradient(0, y + h - 60, 0, y + h)
        bottom_gradient.setColorAt(0, QColor(0, 0, 0, 0))
        bottom_gradient.setColorAt(1, QColor(0, 0, 0, 180))
        painter.fillRect(x, y + h - 60, w, 60, QBrush(bottom_gradient))

        # Main detection result
        painter.setPen(QPen(Qt.GlobalColor.black, 4))  # Shadow
        painter.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        painter.drawText(x + 22, y + 42, f"🍅 {self.current_label.upper()}")
        
        painter.setPen(QPen(color, 2))  # Main text
        painter.drawText(x + 20, y + 40, f"🍅 {self.current_label.upper()}")

        # System metrics
        painter.setFont(QFont("Arial", 14))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        
        # FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        fps_rect = painter.fontMetrics().boundingRect(fps_text)
        painter.drawText(x + w - fps_rect.width() - 20, y + 30, fps_text)
        
        # Confidence percentage
        conf_text = f"Confidence: {self.current_confidence:.1%}"
        painter.drawText(x + 20, y + 65, conf_text)
        
        # Timestamp
        painter.setFont(QFont("Arial", 12))
        timestamp = datetime.now().strftime("%H:%M:%S")
        painter.drawText(x + 20, y + h - 20, f"⏰ {timestamp}")
        
        # Status indicator
        status_indicators = {
            'high': ('🟢', 'EXCELLENT') if self.current_confidence >= 0.8 else None,
            'medium': ('🟡', 'GOOD') if 0.5 <= self.current_confidence < 0.8 else None,
            'low': ('🔴', 'LOW') if self.current_confidence < 0.5 else None
        }
        
        for level, indicator in status_indicators.items():
            if indicator:
                painter.drawText(x + w - 120, y + h - 20, f"{indicator[0]} {indicator[1]}")
                break

    def draw_detection_trail(self, painter: QPainter, img_x: int, img_y: int, 
                           scale_x: float, scale_y: float, color: QColor):
        """Draw trail of recent detections"""
        
        if len(self.detection_history) < 2:
            return
            
        current_time = time.time()
        
        for i, detection in enumerate(self.detection_history[-8:]):  # Last 8 detections
            if not detection['center'] or not detection['bbox']:
                continue
                
            age = current_time - detection['timestamp']
            if age > 3.0:  # Fade out after 3 seconds
                continue
                
            # Calculate fade
            alpha = max(0, 1 - age / 3.0) * 0.4
            trail_color = QColor(color)
            trail_color.setAlphaF(alpha)
            
            # Scale position
            center_x = int(img_x + detection['center'][0] * scale_x)
            center_y = int(img_y + detection['center'][1] * scale_y)
            
            # Draw trail point
            painter.setPen(QPen(trail_color, 2))
            painter.drawEllipse(center_x - 3, center_y - 3, 6, 6)


class HarvestControlPanel(QWidget):
    """Professional control panel for harvesting operations"""

    def __init__(self, robot_controller):
        super().__init__()
        self.robot_controller = robot_controller
        self.selected_arms = set()
        self.harvest_mode = "single"
        self.init_ui()

    def init_ui(self):
        """Initialize professional control panel UI"""
        layout = QVBoxLayout(self)

        # Professional title
        title = QLabel("🎮 Harvest Control Center")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Robot arm selection with enhanced status
        arm_group = QGroupBox("Robot Arm Management")
        arm_layout = QGridLayout()

        self.arm_widgets = []
        for i in range(4):
            widget = self.create_enhanced_arm_widget(i)
            self.arm_widgets.append(widget)
            arm_layout.addWidget(widget, i // 2, i % 2)

        arm_group.setLayout(arm_layout)
        layout.addWidget(arm_group)

        # Enhanced harvest mode selection
        mode_group = QGroupBox("Operation Mode")
        mode_layout = QHBoxLayout()

        self.single_mode_btn = QPushButton("🎯 Single Arm")
        self.single_mode_btn.setCheckable(True)
        self.single_mode_btn.setChecked(True)
        self.single_mode_btn.clicked.connect(lambda: self.set_harvest_mode("single"))

        self.multi_mode_btn = QPushButton("🔄 Multi Arm")
        self.multi_mode_btn.setCheckable(True)
        self.multi_mode_btn.clicked.connect(lambda: self.set_harvest_mode("multi"))

        self.auto_mode_btn = QPushButton("🤖 Full Auto")
        self.auto_mode_btn.setCheckable(True)
        self.auto_mode_btn.setObjectName("primaryButton")
        self.auto_mode_btn.clicked.connect(lambda: self.set_harvest_mode("auto"))

        mode_layout.addWidget(self.single_mode_btn)
        mode_layout.addWidget(self.multi_mode_btn)
        mode_layout.addWidget(self.auto_mode_btn)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Enhanced harvest configuration
        config_group = QGroupBox("Harvest Configuration")
        config_layout = QVBoxLayout()

        # Tomato type selection with icons
        type_layout = QGridLayout()
        
        self.harvest_unripe = QCheckBox("🟢 Harvest Unripe Tomatoes")
        self.harvest_unripe.setChecked(False)
        self.harvest_unripe.setToolTip("Harvest green, unripe tomatoes")
        
        self.harvest_ripe = QCheckBox("🔴 Harvest Ripe Tomatoes")
        self.harvest_ripe.setChecked(True)
        self.harvest_ripe.setToolTip("Harvest red, ripe tomatoes")
        
        self.harvest_rotten = QCheckBox("🟤 Remove Rotten Tomatoes")
        self.harvest_rotten.setChecked(True)
        self.harvest_rotten.setToolTip("Remove brown, rotten tomatoes")

        type_layout.addWidget(self.harvest_unripe, 0, 0)
        type_layout.addWidget(self.harvest_ripe, 0, 1)
        type_layout.addWidget(self.harvest_rotten, 1, 0, 1, 2)

        config_layout.addLayout(type_layout)

        # Enhanced confidence threshold
        threshold_group = QGroupBox("Detection Threshold")
        threshold_layout = QVBoxLayout()

        threshold_control = QHBoxLayout()
        threshold_control.addWidget(QLabel("Minimum Confidence:"))

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 95)
        self.threshold_slider.setValue(75)
        self.threshold_slider.valueChanged.connect(self.update_threshold_display)

        self.threshold_label = QLabel("75%")
        self.threshold_status = QLabel("🟡 Good")

        threshold_control.addWidget(self.threshold_slider)
        threshold_control.addWidget(self.threshold_label)
        threshold_control.addWidget(self.threshold_status)

        threshold_layout.addLayout(threshold_control)
        threshold_group.setLayout(threshold_layout)
        config_layout.addWidget(threshold_group)

        config_group.setLayout(config_layout)
        layout.addWidget(config_group)

        # Enhanced manual control
        control_group = QGroupBox("Manual Robot Control")
        control_layout = QGridLayout()

        self.pick_btn = QPushButton("🤏 Execute Pick")
        self.pick_btn.clicked.connect(lambda: self.manual_action("pick"))
        self.pick_btn.setToolTip("Execute picking operation on selected arms")

        self.place_btn = QPushButton("📦 Place Item")
        self.place_btn.clicked.connect(lambda: self.manual_action("place"))
        self.place_btn.setToolTip("Place harvested item in container")

        self.home_btn = QPushButton("🏠 Home Position")
        self.home_btn.clicked.connect(lambda: self.manual_action("home"))
        self.home_btn.setToolTip("Return arms to home position")

        self.emergency_btn = QPushButton("🛑 EMERGENCY STOP")
        self.emergency_btn.setObjectName("dangerButton")
        self.emergency_btn.clicked.connect(lambda: self.manual_action("emergency"))
        self.emergency_btn.setToolTip("Immediate emergency stop all arms")

        control_layout.addWidget(self.pick_btn, 0, 0)
        control_layout.addWidget(self.place_btn, 0, 1)
        control_layout.addWidget(self.home_btn, 1, 0)
        control_layout.addWidget(self.emergency_btn, 1, 1)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        layout.addStretch()

    def create_enhanced_arm_widget(self, arm_id: int) -> QWidget:
        """Create enhanced arm control widget with detailed status"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        widget.setStyleSheet("border: 2px solid #e94560; border-radius: 8px; margin: 2px;")
        layout = QVBoxLayout(widget)

        # Arm header
        header_layout = QHBoxLayout()
        
        # Arm button
        btn = QPushButton(f"ARM #{arm_id + 1}")
        btn.setCheckable(True)
        btn.clicked.connect(lambda checked: self.toggle_arm(arm_id, checked))
        btn.setMinimumHeight(35)

        # Connection indicator
        conn_indicator = QLabel("⚫")
        conn_indicator.setToolTip("Connection status")

        header_layout.addWidget(btn)
        header_layout.addWidget(conn_indicator)

        # Status display
        status = QLabel("Disconnected")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        status.setStyleSheet("font-size: 11px; color: #ccc;")

        # Enhanced progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setTextVisible(True)
        progress.setMaximumHeight(8)
        progress.setStyleSheet("""
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #4CAF50, stop:1 #45a049);
            }
        """)

        # Last action display
        last_action = QLabel("Ready")
        last_action.setAlignment(Qt.AlignmentFlag.AlignCenter)
        last_action.setStyleSheet("font-size: 10px; color: #999;")

        layout.addLayout(header_layout)
        layout.addWidget(status)
        layout.addWidget(progress)
        layout.addWidget(last_action)

        # Store references
        widget.button = btn
        widget.status = status
        widget.progress = progress
        widget.connection_indicator = conn_indicator
        widget.last_action = last_action
        widget.arm_id = arm_id

        return widget

    def toggle_arm(self, arm_id: int, checked: bool):
        """Enhanced arm selection with mode awareness"""
        if checked:
            self.selected_arms.add(arm_id)
        else:
            self.selected_arms.discard(arm_id)

        # Single mode constraint
        if self.harvest_mode == "single" and checked:
            for i, widget in enumerate(self.arm_widgets):
                if i != arm_id:
                    widget.button.setChecked(False)
                    self.selected_arms.discard(i)

        # Update button styling
        self.update_arm_selection_display()

    def set_harvest_mode(self, mode: str):
        """Enhanced mode setting with validation"""
        self.harvest_mode = mode

        # Update button states
        self.single_mode_btn.setChecked(mode == "single")
        self.multi_mode_btn.setChecked(mode == "multi")
        self.auto_mode_btn.setChecked(mode == "auto")

        # Mode-specific logic
        if mode == "single" and len(self.selected_arms) > 1:
            # Keep only the first selected arm
            first_arm = min(self.selected_arms)
            self.selected_arms = {first_arm}
            for i, widget in enumerate(self.arm_widgets):
                widget.button.setChecked(i == first_arm)

        self.update_arm_selection_display()

    def update_threshold_display(self, value: int):
        """Enhanced threshold display with status indicators"""
        self.threshold_label.setText(f"{value}%")
        
        if value >= 85:
            self.threshold_status.setText("🟢 Excellent")
        elif value >= 70:
            self.threshold_status.setText("🟡 Good")
        elif value >= 50:
            self.threshold_status.setText("🟠 Fair")
        else:
            self.threshold_status.setText("🔴 Low")

    def update_arm_selection_display(self):
        """Update visual indication of selected arms"""
        for i, widget in enumerate(self.arm_widgets):
            if i in self.selected_arms:
                widget.setStyleSheet("border: 2px solid #4CAF50; border-radius: 8px; margin: 2px;")
            else:
                widget.setStyleSheet("border: 2px solid #e94560; border-radius: 8px; margin: 2px;")

    def manual_action(self, action: str):
        """Enhanced manual action execution"""
        if not self.selected_arms:
            QMessageBox.warning(self, "No Arms Selected", "Please select at least one robot arm.")
            return

        command_map = {
            "pick": {"cmd": "MANUAL_PICK", "priority": "normal"},
            "place": {"cmd": "MANUAL_PLACE", "priority": "normal"},
            "home": {"cmd": "HOME_POSITION", "priority": "normal"},
            "emergency": {"cmd": "EMERGENCY_STOP", "priority": "high"}
        }

        if action not in command_map:
            return

        command = command_map[action]
        
        # Execute on selected arms
        for arm_id in self.selected_arms:
            self.robot_controller.send_command(arm_id, command)
            
            # Update UI
            if arm_id < len(self.arm_widgets):
                widget = self.arm_widgets[arm_id]
                widget.last_action.setText(f"Executing {action}...")

        # Special handling for emergency stop
        if action == "emergency":
            self.robot_controller.emergency_stop_all()

    def update_arm_status(self, arm_id: int, status: str, data: dict):
        """Enhanced arm status updates with rich information"""
        if 0 <= arm_id < len(self.arm_widgets):
            widget = self.arm_widgets[arm_id]

            # Status icons and colors
            status_config = {
                "connected": ("🟢", "Connected", "#4CAF50"),
                "ready": ("🟢", "Ready", "#4CAF50"),
                "busy": ("🟡", "Working", "#FFC107"),
                "error": ("🔴", "Error", "#f44336"),
                "disconnected": ("⚫", "Offline", "#666"),
                "timeout": ("🟠", "Timeout", "#FF9800")
            }

            config = status_config.get(status, ("⚫", "Unknown", "#666"))
            icon, text, color = config

            # Update displays
            widget.connection_indicator.setText(icon)
            widget.status.setText(text)
            widget.status.setStyleSheet(f"font-size: 11px; color: {color};")

            # Progress bar updates
            if "progress" in data:
                widget.progress.setValue(int(data["progress"]))
            elif status == "ready":
                widget.progress.setValue(100)
            elif status == "busy":
                widget.progress.setValue(50)
            else:
                widget.progress.setValue(0)

            # Last action updates
            if "command" in data:
                widget.last_action.setText(f"Last: {data['command']}")
            elif status == "ready":
                widget.last_action.setText("Ready for commands")

    def get_harvest_settings(self) -> Dict:
        """Get current harvest configuration"""
        return {
            "harvest_unripe": self.harvest_unripe.isChecked(),
            "harvest_ripe": self.harvest_ripe.isChecked(),
            "harvest_rotten": self.harvest_rotten.isChecked(),
            "confidence_threshold": self.threshold_slider.value() / 100.0,
            "selected_arms": list(self.selected_arms),
            "harvest_mode": self.harvest_mode
        }


class StatisticsWidget(QWidget):
    """Enhanced real-time statistics and analytics widget"""

    def __init__(self):
        super().__init__()
        self.stats = HarvestStats()
        self.session_start_time = datetime.now()
        self.init_ui()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)

    def init_ui(self):
        """Initialize enhanced statistics UI"""
        layout = QVBoxLayout(self)

        # Professional title
        title = QLabel("📊 Harvest Analytics Dashboard")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Session information
        session_group = QGroupBox("Current Session")
        session_layout = QGridLayout()

        # Enhanced stats with icons and colors
        self.total_label = self.create_enhanced_stat_label("🎯 Total Detected:", "0")
        self.unripe_label = self.create_enhanced_stat_label("🟢 Unripe:", "0", QColor(76, 175, 80))
        self.ripe_label = self.create_enhanced_stat_label("🔴 Ripe:", "0", QColor(233, 69, 96))
        self.rotten_label = self.create_enhanced_stat_label("🟤 Rotten:", "0", QColor(121, 85, 72))
        self.success_label = self.create_enhanced_stat_label("✅ Success Rate:", "0%", QColor(76, 175, 80))
        self.accuracy_label = self.create_enhanced_stat_label("🎯 Avg Confidence:", "0%")

        # Session time
        self.session_time_label = self.create_enhanced_stat_label("⏱️ Session Time:", "00:00:00")

        # Layout stats
        stats = [self.total_label, self.unripe_label, self.ripe_label, 
                self.rotten_label, self.success_label, self.accuracy_label, self.session_time_label]
        
        for i, (title, value) in enumerate(stats):
            session_layout.addWidget(title, i // 2, (i % 2) * 2)
            session_layout.addWidget(value, i // 2, (i % 2) * 2 + 1)

        session_group.setLayout(session_layout)
        layout.addWidget(session_group)

        # Enhanced performance chart
        self.create_enhanced_performance_chart()
        layout.addWidget(self.chart_view)

        # Action buttons
        actions_layout = QHBoxLayout()
        
        export_btn = QPushButton("📊 Export Statistics")
        export_btn.clicked.connect(self.export_stats)
        
        reset_btn = QPushButton("🔄 Reset Session")
        reset_btn.clicked.connect(self.reset_session)
        
        actions_layout.addWidget(export_btn)
        actions_layout.addWidget(reset_btn)
        
        layout.addLayout(actions_layout)
        layout.addStretch()

    def create_enhanced_stat_label(self, title: str, value: str, color: QColor = None) -> Tuple[QLabel, QLabel]:
        """Create enhanced statistics label pair"""
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))

        value_label = QLabel(value)
        value_label.setObjectName("statsLabel")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        value_label.setFont(QFont("Arial", 14, QFont.Weight.Bold))

        if color:
            value_label.setStyleSheet(f"color: {color.name()}; font-weight: bold;")

        return title_label, value_label

    def create_enhanced_performance_chart(self):
        """Create enhanced performance trend chart"""
        self.chart = QChart()
        self.chart.setTitle("Real-time Detection Performance")
        self.chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)
        self.chart.setBackgroundBrush(QBrush(QColor(15, 52, 96)))

        # Multiple data series
        self.confidence_series = QLineSeries()
        self.confidence_series.setName("Confidence %")
        self.confidence_series.setColor(QColor(76, 175, 80))

        # Initialize with empty data
        for i in range(30):
            self.confidence_series.append(i, 0)

        self.chart.addSeries(self.confidence_series)

        # Enhanced axes
        axis_x = QValueAxis()
        axis_x.setRange(0, 30)
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("Time (seconds)")
        axis_x.setGridLineColor(QColor(100, 100, 100))

        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        axis_y.setLabelFormat("%d%%")
        axis_y.setTitleText("Confidence Level")
        axis_y.setGridLineColor(QColor(100, 100, 100))

        self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        self.confidence_series.attachAxis(axis_x)
        self.confidence_series.attachAxis(axis_y)

        # Chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setMinimumHeight(250)

    def update_stats(self, label: str, confidence: float):
        """Enhanced statistics updates"""
        self.stats.total_detected += 1

        # Update counts
        if label == "unripe":
            self.stats.unripe_count += 1
        elif label == "ripe":
            self.stats.ripe_count += 1
        elif label == "rotten":
            self.stats.rotten_count += 1

        # Update confidence history
        self.stats.accuracy_history.append(confidence)

        # Update chart
        points = list(self.confidence_series.points())
        
        # Shift existing points
        for i in range(len(points) - 1):
            points[i].setY(points[i + 1].y())

        # Add new point
        if points:
            points[-1].setY(confidence * 100)

        self.confidence_series.replace(points)

    def update_harvest_result(self, success: bool):
        """Update harvest result statistics"""
        if success:
            self.stats.harvest_success += 1
        else:
            self.stats.harvest_failed += 1

    def update_display(self):
        """Enhanced display updates"""
        # Update basic stats
        self.total_label[1].setText(str(self.stats.total_detected))
        self.unripe_label[1].setText(str(self.stats.unripe_count))
        self.ripe_label[1].setText(str(self.stats.ripe_count))
        self.rotten_label[1].setText(str(self.stats.rotten_count))

        # Calculate and display success rate
        total_harvests = self.stats.harvest_success + self.stats.harvest_failed
        if total_harvests > 0:
            success_rate = (self.stats.harvest_success / total_harvests) * 100
            self.success_label[1].setText(f"{success_rate:.1f}%")

        # Calculate and display average confidence
        if self.stats.accuracy_history:
            avg_confidence = sum(self.stats.accuracy_history) / len(self.stats.accuracy_history)
            self.accuracy_label[1].setText(f"{avg_confidence:.1f}%")

        # Update session time
        session_duration = datetime.now() - self.session_start_time
        hours, remainder = divmod(int(session_duration.total_seconds()), 3600)
        minutes, seconds = divmod(remainder, 60)
        self.session_time_label[1].setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    def reset_session(self):
        """Reset session statistics"""
        reply = QMessageBox.question(self, "Reset Session", 
                                    "Are you sure you want to reset all session statistics?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.stats = HarvestStats()
            self.session_start_time = datetime.now()
            
            # Reset chart
            points = list(self.confidence_series.points())
            for point in points:
                point.setY(0)
            self.confidence_series.replace(points)

    def export_stats(self):
        """Enhanced statistics export"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"harvest_analytics_{timestamp}.json"

        # Comprehensive statistics data
        stats_dict = {
            "session_info": {
                "start_time": self.session_start_time.isoformat(),
                "export_time": datetime.now().isoformat(),
                "duration_seconds": (datetime.now() - self.session_start_time).total_seconds()
            },
            "detection_stats": {
                "total_detected": self.stats.total_detected,
                "unripe_count": self.stats.unripe_count,
                "ripe_count": self.stats.ripe_count,
                "rotten_count": self.stats.rotten_count
            },
            "harvest_stats": {
                "harvest_success": self.stats.harvest_success,
                "harvest_failed": self.stats.harvest_failed,
                "success_rate": (self.stats.harvest_success / max(1, self.stats.harvest_success + self.stats.harvest_failed)) * 100
            },
            "performance_data": {
                "accuracy_history": list(self.stats.accuracy_history),
                "average_confidence": sum(self.stats.accuracy_history) / len(self.stats.accuracy_history) if self.stats.accuracy_history else 0
            }
        }

        try:
            with open(filename, 'w') as f:
                json.dump(stats_dict, f, indent=2)

            QMessageBox.information(self, "Export Successful",
                                  f"Analytics exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                               f"Failed to export analytics:\n{e}")


class AdvancedMainWindow(QMainWindow):
    """Complete advanced main window with professional features"""

    def __init__(self):
        super().__init__()
        self.predictor = OptimizedPredictor()
        self.robot_controller = EnhancedRobotController()
        self.camera_thread = None
        self.current_frame = None
        self.auto_harvest = False
        self.detection_results = deque(maxlen=50)

        self.init_ui()
        self.setup_connections()
        self.start_system()

    def init_ui(self):
        """Initialize the complete professional UI"""
        self.setWindowTitle("🍅 Professional Tomato Harvesting System v3.0")
        self.setGeometry(100, 100, 1600, 1000)
        self.setStyleSheet(StyleSheet.DARK_THEME)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Professional main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Left panel - Camera and detection info
        left_panel = QVBoxLayout()

        # Enhanced camera widget
        self.camera_widget = ModernCameraWidget()
        left_panel.addWidget(self.camera_widget, 3)

        # Detection information panel
        detection_panel = self.create_detection_info_panel()
        left_panel.addWidget(detection_panel)

        # Professional action controls
        action_panel = self.create_action_panel()
        left_panel.addWidget(action_panel)

        main_layout.addLayout(left_panel, 2)

        # Right panel - Professional tabbed interface
        right_panel = QTabWidget()
        right_panel.setTabPosition(QTabWidget.TabPosition.North)

        # Enhanced tabs
        self.control_panel = HarvestControlPanel(self.robot_controller)
        right_panel.addTab(self.control_panel, "🎮 Control Center")

        self.stats_widget = StatisticsWidget()
        right_panel.addTab(self.stats_widget, "📊 Analytics")

        self.log_widget = self.create_enhanced_log_widget()
        right_panel.addTab(self.log_widget, "📝 System Log")

        self.settings_widget = self.create_professional_settings_widget()
        right_panel.addTab(self.settings_widget, "⚙️ Configuration")

        main_layout.addWidget(right_panel, 1)

        # Professional status bar
        self.create_professional_status_bar()

        # Auto harvest timer
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.auto_harvest_cycle)

    def create_detection_info_panel(self) -> QWidget:
        """Create professional detection information panel"""
        panel = QGroupBox("🎯 Detection Status")
        layout = QVBoxLayout(panel)

        # Current detection display
        self.detection_display = QLabel("Waiting for detection...")
        self.detection_display.setObjectName("detectionLabel")
        self.detection_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_display.setMinimumHeight(60)
        self.detection_display.setStyleSheet("""
            QLabel#detectionLabel {
                background-color: #0f3460;
                border: 2px solid #e94560;
                border-radius: 10px;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.detection_display)

        # Enhanced metrics
        metrics_layout = QGridLayout()

        # Confidence meter
        conf_layout = QVBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Level:"))
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setTextVisible(True)
        self.confidence_bar.setStyleSheet("""
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                                stop:0 #4CAF50, stop:1 #45a049);
            }
        """)
        conf_layout.addWidget(self.confidence_bar)
        
        # Bounding box info
        bbox_layout = QGridLayout()
        bbox_layout.addWidget(QLabel("X:"), 0, 0)
        bbox_layout.addWidget(QLabel("Y:"), 0, 2)
        bbox_layout.addWidget(QLabel("W:"), 1, 0)
        bbox_layout.addWidget(QLabel("H:"), 1, 2)

        self.bbox_labels = {
            'x': QLabel("0"),
            'y': QLabel("0"),
            'w': QLabel("0"),
            'h': QLabel("0")
        }

        bbox_layout.addWidget(self.bbox_labels['x'], 0, 1)
        bbox_layout.addWidget(self.bbox_labels['y'], 0, 3)
        bbox_layout.addWidget(self.bbox_labels['w'], 1, 1)
        bbox_layout.addWidget(self.bbox_labels['h'], 1, 3)

        metrics_layout.addLayout(conf_layout, 0, 0, 1, 2)
        
        bbox_group = QGroupBox("Bounding Box")
        bbox_group.setLayout(bbox_layout)
        metrics_layout.addWidget(bbox_group, 1, 0, 1, 2)

        layout.addLayout(metrics_layout)
        return panel

    def create_action_panel(self) -> QWidget:
        """Create professional action control panel"""
        panel = QGroupBox("🎮 Quick Actions")
        layout = QGridLayout(panel)

        # Auto harvest control
        self.auto_harvest_btn = QPushButton("🤖 Auto Harvest: OFF")
        self.auto_harvest_btn.setObjectName("primaryButton")
        self.auto_harvest_btn.setCheckable(True)
        self.auto_harvest_btn.clicked.connect(self.toggle_auto_harvest)
        self.auto_harvest_btn.setToolTip("Enable/disable automatic harvesting")

        # Manual harvest trigger
        self.manual_harvest_btn = QPushButton("🎯 Manual Harvest")
        self.manual_harvest_btn.clicked.connect(self.trigger_manual_harvest)
        self.manual_harvest_btn.setToolTip("Trigger manual harvest of current detection")

        # Image capture
        self.capture_btn = QPushButton("📸 Capture Image")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setToolTip("Save current camera frame")

        # Emergency stop
        self.emergency_stop_btn = QPushButton("🛑 Emergency Stop")
        self.emergency_stop_btn.setObjectName("dangerButton")
        self.emergency_stop_btn.clicked.connect(self.emergency_stop_all)
        self.emergency_stop_btn.setToolTip("Emergency stop all robot arms")

        layout.addWidget(self.auto_harvest_btn, 0, 0, 1, 2)
        layout.addWidget(self.manual_harvest_btn, 1, 0)
        layout.addWidget(self.capture_btn, 1, 1)
        layout.addWidget(self.emergency_stop_btn, 2, 0, 1, 2)

        return panel

    def create_enhanced_log_widget(self) -> QWidget:
        """Create enhanced system log widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Log display
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 10))
        self.log_text.setMaximumHeight(400)

        # Enhanced log controls
        controls_layout = QHBoxLayout()

        # Log level filter
        level_combo = QComboBox()
        level_combo.addItems(["All", "INFO", "WARNING", "ERROR"])
        level_combo.currentTextChanged.connect(self.filter_log_level)

        # Clear log
        clear_btn = QPushButton("🗑️ Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)

        # Export log
        export_btn = QPushButton("💾 Export Log")
        export_btn.clicked.connect(self.export_log)

        # Auto-scroll toggle
        self.auto_scroll_cb = QCheckBox("Auto-scroll")
        self.auto_scroll_cb.setChecked(True)

        controls_layout.addWidget(QLabel("Level:"))
        controls_layout.addWidget(level_combo)
        controls_layout.addWidget(clear_btn)
        controls_layout.addWidget(export_btn)
        controls_layout.addWidget(self.auto_scroll_cb)
        controls_layout.addStretch()

        layout.addWidget(self.log_text)
        layout.addLayout(controls_layout)
        
        return widget

    def create_professional_settings_widget(self) -> QWidget:
        """Create comprehensive settings widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # AI Model Configuration
        model_group = QGroupBox("🤖 AI Model Configuration")
        model_layout = QVBoxLayout()

        # Model file selection
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(QLabel("Model File (.h5):"))
        
        self.model_path_edit = QLineEdit("tomato.h5")
        self.model_path_edit.setPlaceholderText("Select .h5 model file")
        model_file_layout.addWidget(self.model_path_edit)
        
        browse_model_btn = QPushButton("📁 Browse...")
        browse_model_btn.setObjectName("browseButton")
        browse_model_btn.clicked.connect(self.browse_model_file)
        model_file_layout.addWidget(browse_model_btn)
        
        model_layout.addLayout(model_file_layout)

        # Labels file selection
        labels_file_layout = QHBoxLayout()
        labels_file_layout.addWidget(QLabel("Labels File (.txt):"))
        
        self.labels_path_edit = QLineEdit("labels.txt")
        self.labels_path_edit.setPlaceholderText("Select labels.txt file")
        labels_file_layout.addWidget(self.labels_path_edit)
        
        browse_labels_btn = QPushButton("📁 Browse...")
        browse_labels_btn.setObjectName("browseButton")
        browse_labels_btn.clicked.connect(self.browse_labels_file)
        labels_file_layout.addWidget(browse_labels_btn)
        
        model_layout.addLayout(labels_file_layout)

        # Load model button
        load_model_btn = QPushButton("🔄 Load Model & Labels")
        load_model_btn.setObjectName("primaryButton")
        load_model_btn.clicked.connect(self.load_model_and_labels)
        model_layout.addWidget(load_model_btn)

        # Model status display
        self.model_info_display = QTextEdit()
        self.model_info_display.setReadOnly(True)
        self.model_info_display.setMaximumHeight(80)
        self.model_info_display.setPlainText("Model: Not loaded\nClick 'Load Model & Labels' to begin")
        model_layout.addWidget(self.model_info_display)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Camera Configuration
        camera_group = QGroupBox("📷 Camera Configuration")
        camera_layout = QVBoxLayout()

        # Resolution settings
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["320x240", "640x480", "800x600", "1280x720", "1920x1080"])
        self.resolution_combo.setCurrentText("640x480")
        self.resolution_combo.currentTextChanged.connect(self.change_resolution)

        res_layout.addWidget(self.resolution_combo)
        camera_layout.addLayout(res_layout)

        # FPS settings
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Target FPS:"))

        self.fps_spinbox = QSpinBox()
        self.fps_spinbox.setRange(10, 60)
        self.fps_spinbox.setValue(30)
        self.fps_spinbox.valueChanged.connect(self.change_fps)

        fps_layout.addWidget(self.fps_spinbox)
        camera_layout.addLayout(fps_layout)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Robot Configuration
        robot_group = QGroupBox("🤖 Robot Configuration")
        robot_layout = QVBoxLayout()

        # Connection settings
        conn_layout = QGridLayout()
        
        conn_layout.addWidget(QLabel("Baud Rate:"), 0, 0)
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "57600", "115200", "230400"])
        self.baudrate_combo.setCurrentText("115200")
        conn_layout.addWidget(self.baudrate_combo, 0, 1)

        conn_layout.addWidget(QLabel("Timeout (s):"), 1, 0)
        self.timeout_spinbox = QSpinBox()
        self.timeout_spinbox.setRange(1, 10)
        self.timeout_spinbox.setValue(2)
        conn_layout.addWidget(self.timeout_spinbox, 1, 1)

        robot_layout.addLayout(conn_layout)

        # Robot control buttons
        robot_controls = QHBoxLayout()
        
        reconnect_btn = QPushButton("🔌 Reconnect All")
        reconnect_btn.clicked.connect(self.reconnect_arms)
        
        home_all_btn = QPushButton("🏠 Home All")
        home_all_btn.clicked.connect(self.home_all_arms)
        
        robot_controls.addWidget(reconnect_btn)
        robot_controls.addWidget(home_all_btn)
        
        robot_layout.addLayout(robot_controls)
        robot_group.setLayout(robot_layout)
        layout.addWidget(robot_group)

        layout.addStretch()
        return widget

    def browse_model_file(self):
        """Browse for AI model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Keras Model File",
            "",
            "Keras Models (*.h5 *.hdf5);;All Files (*)"
        )
        if file_path:
            self.model_path_edit.setText(file_path)

    def browse_labels_file(self):
        """Browse for labels file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Labels File",
            "",
            "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.labels_path_edit.setText(file_path)

    def load_model_and_labels(self):
        """Load AI model and labels from selected files"""
        model_path = self.model_path_edit.text().strip()
        labels_path = self.labels_path_edit.text().strip()

        if not model_path:
            QMessageBox.warning(self, "Missing Model", "Please select a model file (.h5)")
            return

        # Update predictor
        self.predictor.set_model_path(model_path)
        if labels_path:
            self.predictor.set_labels_path(labels_path)

        # Update status display
        if self.predictor.model:
            info_text = f"✅ Model: {Path(model_path).name}\n"
            if labels_path and Path(labels_path).exists():
                info_text += f"✅ Labels: {Path(labels_path).name}\n"
                info_text += f"📋 Classes: {', '.join(self.predictor.classes)}"
            else:
                info_text += f"⚠️ Labels: Using default classes\n"
                info_text += f"📋 Classes: {', '.join(self.predictor.classes)}"
            
            self.model_info_display.setPlainText(info_text)
            self.model_status.setText("🤖 Model: Ready")
            self.log_message("Model", f"Successfully loaded {Path(model_path).name}", "INFO")
        else:
            self.model_info_display.setPlainText("❌ Model: Failed to load\nCheck file path and format")
            self.model_status.setText("🤖 Model: Error")
            self.log_message("Model", f"Failed to load {Path(model_path).name}", "ERROR")

    def create_professional_status_bar(self):
        """Create professional status bar with rich information"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # System status indicators
        self.system_status = QLabel("🟢 System Initializing...")
        self.camera_status = QLabel("📷 Camera: Starting...")
        self.model_status = QLabel("🤖 Model: Loading...")
        self.robot_status = QLabel("🤖 Arms: Connecting...")

        # Add to status bar
        self.status_bar.addWidget(self.system_status)
        self.status_bar.addWidget(self.camera_status)
        self.status_bar.addWidget(self.model_status)
        self.status_bar.addPermanentWidget(self.robot_status)

    def setup_connections(self):
        """Setup all signal/slot connections"""
        # Robot controller signals
        self.robot_controller.status_update.connect(self.handle_arm_status)
        self.robot_controller.harvest_complete.connect(self.handle_harvest_complete)

        # Predictor signals
        self.predictor.prediction_complete.connect(self.handle_prediction)

    def start_system(self):
        """Start all system components"""
        self.log_message("System", "🍅 Professional Tomato Harvesting System v3.0 starting...", "INFO")

        # Start camera
        self.camera_thread = PiCameraThread()
        self.camera_thread.frame_ready.connect(self.process_frame)
        self.camera_thread.start()

        # Connect robot arms
        QTimer.singleShot(2000, self.robot_controller.connect_all)

        # Update initial status
        self.system_status.setText("🟢 System: Running")
        self.camera_status.setText("📷 Camera: Active")

        if self.predictor.model:
            self.model_status.setText("🤖 Model: Ready")
        else:
            self.model_status.setText("🤖 Model: Not loaded")

        self.log_message("System", "All subsystems initialized successfully", "INFO")

    def process_frame(self, frame: np.ndarray):
        """Process camera frame with enhanced detection"""
        self.current_frame = frame

        # Run AI prediction
        label, confidence, fps, bbox = self.predictor.predict(frame)

        # Update camera display
        self.camera_widget.update_frame(frame, label, confidence, fps, bbox)

        # Update detection info panel
        self.update_detection_info(label, confidence, bbox)

        # Update statistics
        if label != "unknown" and confidence > 0.4:
            self.stats_widget.update_stats(label, confidence)
            
            # Store detection result
            detection = DetectionResult(
                label=label,
                confidence=confidence,
                bbox=bbox,
                timestamp=datetime.now(),
                center_point=(bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2) if bbox else (0, 0)
            )
            self.detection_results.append(detection)

    def update_detection_info(self, label: str, confidence: float, bbox: Tuple[int, int, int, int]):
        """Update detection information panel"""
        # Color-coded detection display
        colors = {
            'unripe': '#4CAF50',
            'ripe': '#e94560', 
            'rotten': '#795548',
            'unknown': '#9E9E9E',
            'error': '#FF9800'
        }
        
        color = colors.get(label, colors['unknown'])
        self.detection_display.setText(f"🍅 {label.upper()}")
        self.detection_display.setStyleSheet(f"""
            QLabel#detectionLabel {{
                background-color: {color};
                color: white;
                border: 2px solid {color};
                border-radius: 10px;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }}
        """)

        # Update confidence bar
        self.confidence_bar.setValue(int(confidence * 100))

        # Update bounding box info
        if bbox:
            x, y, w, h = bbox
            self.bbox_labels['x'].setText(str(x))
            self.bbox_labels['y'].setText(str(y))
            self.bbox_labels['w'].setText(str(w))
            self.bbox_labels['h'].setText(str(h))

    def handle_prediction(self, label: str, confidence: float, image: np.ndarray, fps: float, bbox: Tuple[int, int, int, int]):
        """Handle AI prediction results"""
        # Auto harvest logic
        if self.auto_harvest and self.should_harvest(label, confidence):
            detection = DetectionResult(
                label=label,
                confidence=confidence,
                bbox=bbox,
                timestamp=datetime.now(),
                center_point=(bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2) if bbox else (0, 0)
            )
            
            # Execute harvest on selected arms
            settings = self.control_panel.get_harvest_settings()
            for arm_id in settings['selected_arms']:
                success = self.robot_controller.smart_harvest(arm_id, label, detection)
                if success:
                    self.log_message("Harvest", f"Auto harvest initiated on Arm {arm_id} for {label}", "INFO")

    def should_harvest(self, label: str, confidence: float) -> bool:
        """Determine if tomato should be harvested"""
        settings = self.control_panel.get_harvest_settings()
        
        # Check confidence threshold
        if confidence < settings['confidence_threshold']:
            return False
            
        # Check if type should be harvested
        if label == "unripe" and not settings['harvest_unripe']:
            return False
        elif label == "ripe" and not settings['harvest_ripe']:
            return False
        elif label == "rotten" and not settings['harvest_rotten']:
            return False
            
        return True

    def handle_arm_status(self, arm_id: int, status: str, data: dict):
        """Handle robot arm status updates"""
        self.control_panel.update_arm_status(arm_id, status, data)

        # Update overall robot status
        connected_count = sum(1 for s in self.robot_controller.arm_status.values()
                            if s in ["ready", "connected"])
        self.robot_status.setText(f"🤖 Arms: {connected_count}/4 Ready")

        # Log significant status changes
        if status in ["connected", "error", "timeout"]:
            level = "INFO" if status == "connected" else "ERROR"
            self.log_message(f"Arm{arm_id}", f"Status: {status} - {data}", level)

    def handle_harvest_complete(self, arm_id: int, tomato_type: str, success: bool):
        """Handle harvest completion"""
        self.stats_widget.update_harvest_result(success)
        
        result = "✅ Success" if success else "❌ Failed"
        self.log_message(f"Arm{arm_id}", f"Harvest {tomato_type}: {result}", "INFO")

    def toggle_auto_harvest(self):
        """Toggle automatic harvesting mode"""
        self.auto_harvest = not self.auto_harvest

        if self.auto_harvest:
            # Check if any arms are selected
            settings = self.control_panel.get_harvest_settings()
            if not settings['selected_arms']:
                QMessageBox.warning(self, "No Arms Selected", 
                                  "Please select at least one robot arm before enabling auto harvest.")
                self.auto_harvest = False
                self.auto_harvest_btn.setChecked(False)
                return

            self.auto_harvest_btn.setText("🤖 Auto Harvest: ON")
            self.auto_timer.start(500)  # Check every 500ms
            self.log_message("System", "🤖 Auto harvest mode enabled", "INFO")
        else:
            self.auto_harvest_btn.setText("🤖 Auto Harvest: OFF")
            self.auto_timer.stop()
            self.log_message("System", "🤖 Auto harvest mode disabled", "INFO")

    def trigger_manual_harvest(self):
        """Trigger manual harvest of current detection"""
        if not self.detection_results:
            QMessageBox.information(self, "No Detection", "No valid detection to harvest.")
            return

        latest_detection = self.detection_results[-1]
        settings = self.control_panel.get_harvest_settings()
        
        if not settings['selected_arms']:
            QMessageBox.warning(self, "No Arms Selected", "Please select at least one robot arm.")
            return

        # Execute manual harvest
        for arm_id in settings['selected_arms']:
            success = self.robot_controller.smart_harvest(arm_id, latest_detection.label, latest_detection)
            if success:
                self.log_message("Harvest", f"Manual harvest initiated on Arm {arm_id}", "INFO")

    def capture_image(self):
        """Capture and save current camera frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tomato_capture_{timestamp}.jpg"
            
            try:
                cv2.imwrite(filename, self.current_frame)
                self.log_message("Camera", f"📸 Image saved: {filename}", "INFO")
                QMessageBox.information(self, "Capture Successful", f"Image saved as:\n{filename}")
            except Exception as e:
                self.log_message("Camera", f"❌ Failed to save image: {e}", "ERROR")
                QMessageBox.critical(self, "Capture Failed", f"Failed to save image:\n{e}")

    def emergency_stop_all(self):
        """Emergency stop all robot arms"""
        reply = QMessageBox.question(self, "Emergency Stop", 
                                    "Emergency stop all robot arms?",
                                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            self.robot_controller.emergency_stop_all()
            self.auto_harvest = False
            self.auto_harvest_btn.setChecked(False)
            self.auto_harvest_btn.setText("🤖 Auto Harvest: OFF")
            self.auto_timer.stop()
            self.log_message("System", "🛑 EMERGENCY STOP activated", "WARNING")

    def change_resolution(self, resolution: str):
        """Change camera resolution"""
        if self.camera_thread:
            try:
                width, height = map(int, resolution.split('x'))
                self.camera_thread.set_resolution(width, height)
                self.log_message("Camera", f"📷 Resolution changed to {resolution}", "INFO")
            except Exception as e:
                self.log_message("Camera", f"❌ Failed to change resolution: {e}", "ERROR")

    def change_fps(self, fps: int):
        """Change camera FPS"""
        if self.camera_thread:
            self.camera_thread.set_fps(fps)
            self.log_message("Camera", f"📷 FPS changed to {fps}", "INFO")

    def reconnect_arms(self):
        """Reconnect all robot arms"""
        self.robot_controller.disconnect_all()
        QTimer.singleShot(1000, self.robot_controller.connect_all)
        self.log_message("System", "🔌 Reconnecting all robot arms...", "INFO")

    def home_all_arms(self):
        """Send all arms to home position"""
        self.robot_controller.home_all_arms()
        self.log_message("System", "🏠 Sending all arms to home position", "INFO")

    def auto_harvest_cycle(self):
        """Automatic harvest cycle (handled in handle_prediction)"""
        pass

    def filter_log_level(self, level: str):
        """Filter log display by level"""
        # This would implement log level filtering
        # For now, just log the filter change
        self.log_message("System", f"Log level filter set to: {level}", "INFO")

    def export_log(self):
        """Export system log to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_log_{timestamp}.txt"

        try:
            with open(filename, 'w') as f:
                f.write("# Tomato Harvesting System Log\n")
                f.write(f"# Generated: {datetime.now().isoformat()}\n\n")
                f.write(self.log_text.toPlainText())

            QMessageBox.information(self, "Export Successful",
                                  f"System log exported to:\n{filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                               f"Failed to export log:\n{e}")

    def log_message(self, source: str, message: str, level: str = "INFO"):
        """Add enhanced message to system log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding for different levels
        colors = {
            "INFO": "#4CAF50",
            "WARNING": "#FF9800", 
            "ERROR": "#f44336"
        }
        
        color = colors.get(level, "#eee")
        
        # Format log entry with HTML for styling
        log_entry = f'<span style="color: #ccc;">[{timestamp}]</span> ' \
                   f'<span style="color: {color}; font-weight: bold;">[{level}]</span> ' \
                   f'<span style="color: #e94560; font-weight: bold;">{source}:</span> ' \
                   f'<span style="color: #eee;">{message}</span>'
        
        self.log_text.append(log_entry)

        # Auto-scroll if enabled
        if self.auto_scroll_cb.isChecked():
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """Handle application shutdown"""
        self.log_message("System", "🔄 Shutting down system...", "INFO")
        
        # Stop auto harvest
        self.auto_harvest = False
        self.auto_timer.stop()

        # Stop camera
        if self.camera_thread:
            self.camera_thread.stop()

        # Disconnect arms
        self.robot_controller.disconnect_all()

        self.log_message("System", "✅ System shutdown complete", "INFO")
        event.accept()


def main():
    """Main application entry point with enhanced error handling"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # Application metadata
    app.setApplicationName("Professional Tomato Harvesting System")
    app.setOrganizationName("Advanced Agriculture Technologies")
    app.setApplicationVersion("3.0")

    # Dependency checks with user-friendly messages
    missing_deps = []
    
    if not TF_AVAILABLE:
        missing_deps.append("TensorFlow (pip install tensorflow)")
    
    if not PICAMERA_AVAILABLE:
        logger.info("Pi Camera not available, using USB camera fallback")

    if missing_deps:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Missing Dependencies")
        msg.setText("Some optional dependencies are missing:")
        msg.setInformativeText("\n".join(missing_deps))
        msg.setDetailedText("The system will run with reduced functionality. Install missing dependencies for full features.")
        msg.exec()

    # Log system information
    try:
        from PyQt6 import QtCore
        qt_version = f"PyQt6 {QtCore.PYQT_VERSION_STR}"
    except:
        qt_version = "PyQt6 version unknown"

    logger.info(f"Starting Professional Tomato Harvesting System v3.0")
    logger.info(f"Platform: {sys.platform}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Qt: {qt_version}")
    logger.info(f"TensorFlow: {'Available' if TF_AVAILABLE else 'Not available'}")
    logger.info(f"Pi Camera: {'Available' if PICAMERA_AVAILABLE else 'Using USB fallback'}")

    # Create and configure main window
    try:
        window = AdvancedMainWindow()
        window.showMaximized()
        
        # Show startup message
        QTimer.singleShot(1000, lambda: window.log_message("System", 
            "🚀 Professional Tomato Harvesting System v3.0 ready for operation!", "INFO"))
        
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle("Startup Error")
        msg.setText("Failed to start the application")
        msg.setInformativeText(str(e))
        msg.exec()
        sys.exit(1)


if __name__ == '__main__':
    main()
