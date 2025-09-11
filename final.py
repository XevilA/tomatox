#!/usr/bin/env python3
"""
Advanced Tomato Harvesting Robot Control System for Raspberry Pi 4
Enhanced with improved detection display and file browser
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
    """Detection result with bounding box"""
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    timestamp: datetime


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
        font-size: 18px;
        font-weight: bold;
        padding: 8px;
        border-radius: 5px;
        margin: 2px;
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
    """Optimized Model Predictor with enhanced detection visualization"""
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

        # Color ranges for pre-filtering (HSV)
        self.color_ranges = {
            'unripe': [(35, 50, 50), (85, 255, 255)],    # Green
            'ripe': [(0, 50, 50), (10, 255, 255)],       # Red
            'rotten': [(10, 50, 20), (25, 255, 150)]     # Brown/Dark
        }

        self.load_labels()
        if TF_AVAILABLE:
            self.load_model()

    def load_labels(self):
        """Load class labels from file"""
        try:
            if Path(self.labels_path).exists():
                with open(self.labels_path, 'r') as f:
                    self.classes = [line.strip() for line in f.readlines()]
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
            return

        try:
            self.model = keras.models.load_model(self.model_path)
            
            # Try to optimize with TFLite if possible
            try:
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                tflite_model = converter.convert()
                self.interpreter = tf.lite.Interpreter(model_content=tflite_model)
                self.interpreter.allocate_tensors()

                input_details = self.interpreter.get_input_details()
                self.input_shape = input_details[0]['shape'][1:3]
                logger.info("Model optimized with TFLite")
            except:
                # Fallback to regular model
                input_shape = self.model.input_shape
                self.input_shape = (input_shape[1], input_shape[2])
                logger.info("Using regular Keras model")

            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def detect_tomato_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Detect tomato region using color-based segmentation"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create combined mask for all tomato colors
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for class_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x, y, w, h)
        
        # Default to center region if no tomato detected
        h, w = image.shape[:2]
        return (w//4, h//4, w//2, h//2)

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with color analysis"""
        # Apply slight denoising
        denoised = cv2.fastNlDenoising(image)

        # Enhance contrast
        lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # Resize for model
        resized = cv2.resize(enhanced, self.input_shape, interpolation=cv2.INTER_AREA)

        # Normalize
        normalized = resized.astype(np.float32) / 255.0

        return np.expand_dims(normalized, axis=0)

    def quick_color_check(self, image: np.ndarray) -> Optional[str]:
        """Quick color-based pre-classification"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        max_ratio = 0
        best_class = None

        for class_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])

            if ratio > max_ratio:
                max_ratio = ratio
                best_class = class_name

        return best_class if max_ratio > 0.2 else None

    def predict(self, image: np.ndarray) -> Tuple[str, float, float, Tuple[int, int, int, int]]:
        """Predict with performance metrics and bounding box"""
        start_time = time.time()

        if self.model is None:
            bbox = self.detect_tomato_region(image)
            return "unknown", 0.0, 0.0, bbox

        try:
            # Detect tomato region
            bbox = self.detect_tomato_region(image)
            
            # Extract region of interest
            x, y, w, h = bbox
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                roi = image

            # Quick color check first
            quick_class = self.quick_color_check(roi)

            # Preprocess
            preprocessed = self.preprocess_image(roi)

            # Inference
            if hasattr(self, 'interpreter'):
                # TFLite inference
                input_details = self.interpreter.get_input_details()
                output_details = self.interpreter.get_output_details()

                self.interpreter.set_tensor(input_details[0]['index'], preprocessed)
                self.interpreter.invoke()

                predictions = self.interpreter.get_tensor(output_details[0]['index'])[0]
            else:
                # Regular model inference
                predictions = self.model.predict(preprocessed, verbose=0)[0]

            # Get result
            class_idx = np.argmax(predictions)
            confidence = float(predictions[class_idx])
            
            if class_idx < len(self.classes):
                label = self.classes[class_idx]
            else:
                label = "unknown"

            # Boost confidence if color check matches
            if quick_class == label:
                confidence = min(confidence * 1.1, 1.0)

            # Calculate FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0

            self.prediction_complete.emit(label, confidence, image, fps, bbox)

            return label, confidence, fps, bbox

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            bbox = self.detect_tomato_region(image)
            return "error", 0.0, 0.0, bbox


class PiCameraThread(QThread):
    """Optimized thread for Pi Camera capture"""
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.running = False
        self.camera = None
        self.use_pi_camera = PICAMERA_AVAILABLE
        self.target_fps = 30
        self.resolution = (640, 480)

    def set_resolution(self, width: int, height: int):
        """Set camera resolution"""
        self.resolution = (width, height)

    def set_fps(self, fps: int):
        """Set target FPS"""
        self.target_fps = fps

    def run(self):
        """Main camera capture loop"""
        self.running = True
        frame_time = 1.0 / self.target_fps

        if self.use_pi_camera:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    buffer_count=1
                )
                self.camera.configure(config)
                self.camera.start()

                while self.running:
                    frame_start = time.time()
                    frame = self.camera.capture_array()
                    # Convert RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.frame_ready.emit(frame)
                    
                    # Maintain target FPS
                    elapsed = time.time() - frame_start
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)

            except Exception as e:
                logger.error(f"Pi Camera error: {e}")
                self.use_pi_camera = False

        if not self.use_pi_camera:
            # Fallback to USB camera
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            while self.running:
                frame_start = time.time()
                ret, frame = cap.read()
                if ret:
                    self.frame_ready.emit(frame)
                else:
                    logger.error("Failed to capture frame")
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
            self.camera.stop()
        self.wait()


class EnhancedRobotController(QObject):
    """Enhanced robot arm controller with position feedback"""
    status_update = pyqtSignal(int, str, dict)  # arm_id, status, data
    harvest_complete = pyqtSignal(int, str, bool)  # arm_id, tomato_type, success

    def __init__(self):
        super().__init__()
        self.connections = {}
        self.ports = [f'/dev/ttyUSB{i}' for i in range(4)]
        self.baudrate = 115200
        self.timeout = 1.0
        self.arm_status = {i: "disconnected" for i in range(4)}
        self.harvest_queue = deque()

    def disconnect_all(self):
        """Disconnect all arms"""
        for arm_id, connection in self.connections.items():
            try:
                connection.close()
            except:
                pass
        self.connections.clear()
        for i in range(4):
            self.arm_status[i] = "disconnected"

    def connect_all(self):
        """Connect to all robot arms with auto-detection"""
        available_ports = [p.device for p in serial.tools.list_ports.comports()]

        for i, port in enumerate(self.ports):
            if port in available_ports:
                self.connect_arm(i, port)
            else:
                # Try alternative naming
                alt_port = f'/dev/ttyACM{i}'
                if alt_port in available_ports:
                    self.connect_arm(i, alt_port)

    def connect_arm(self, arm_id: int, port: str):
        """Connect with handshake verification"""
        try:
            ser = serial.Serial(port, self.baudrate, timeout=self.timeout)
            time.sleep(2)  # Wait for Arduino reset

            # Send handshake
            ser.write(b'{"cmd":"PING"}\n')
            response = ser.readline().decode('utf-8').strip()

            if response:
                self.connections[arm_id] = ser
                self.arm_status[arm_id] = "ready"
                logger.info(f"Arm {arm_id} connected on {port}")
                self.status_update.emit(arm_id, "connected", {"port": port})
            else:
                raise Exception("No handshake response")

        except Exception as e:
            logger.error(f"Failed to connect arm {arm_id}: {e}")
            self.arm_status[arm_id] = "error"
            self.status_update.emit(arm_id, "error", {"message": str(e)})

    def smart_harvest(self, arm_id: int, tomato_class: str, position: Dict):
        """Smart harvesting with position awareness"""
        commands = {
            'unripe': {
                "cmd": "SMART_PICK",
                "type": "unripe",
                "force": 80,  # Firmer grip
                "speed": "normal",
                "position": position
            },
            'ripe': {
                "cmd": "SMART_PICK",
                "type": "ripe",
                "force": 60,  # Gentle grip
                "speed": "slow",
                "position": position
            },
            'rotten': {
                "cmd": "SMART_DISCARD",
                "type": "rotten",
                "force": 40,  # Very gentle
                "speed": "very_slow",
                "position": position
            }
        }

        if tomato_class in commands:
            success = self.send_command(arm_id, commands[tomato_class])
            self.harvest_complete.emit(arm_id, tomato_class, success)
            return success
        return False

    def send_command(self, arm_id: int, command: Dict) -> bool:
        """Enhanced command sending with response handling"""
        if arm_id not in self.connections or self.arm_status[arm_id] != "ready":
            return False

        try:
            ser = self.connections[arm_id]
            json_cmd = json.dumps(command) + '\n'
            ser.write(json_cmd.encode('utf-8'))

            # Wait for response
            response = ser.readline().decode('utf-8').strip()
            if response:
                resp_data = json.loads(response)
                self.status_update.emit(arm_id, "response", resp_data)
                return resp_data.get("status") == "success"

            return False

        except Exception as e:
            logger.error(f"Command error for arm {arm_id}: {e}")
            self.arm_status[arm_id] = "error"
            return False


class ModernCameraWidget(QWidget):
    """Enhanced camera display widget with beautiful detection visualization"""

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.current_label = "Waiting..."
        self.current_confidence = 0.0
        self.fps = 0.0
        self.detection_bbox = None
        self.detection_history = deque(maxlen=10)
        self.animation_alpha = 0.0
        self.init_ui()

        # Animation timer for smooth effects
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 FPS animation

    def init_ui(self):
        """Initialize UI components"""
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #0f3460; border-radius: 15px; border: 2px solid #e94560;")

    def update_frame(self, frame: np.ndarray, label: str, confidence: float, fps: float, bbox: Tuple[int, int, int, int]):
        """Update display with new frame and detection info"""
        self.current_frame = frame
        self.current_label = label
        self.current_confidence = confidence
        self.fps = fps
        self.detection_bbox = bbox
        
        # Add to detection history for effects
        self.detection_history.append({
            'label': label,
            'confidence': confidence,
            'bbox': bbox,
            'timestamp': time.time()
        })
        
        self.update()

    def update_animation(self):
        """Update animation effects"""
        self.animation_alpha = (self.animation_alpha + 0.1) % (2 * np.pi)
        if self.current_frame is not None:
            self.update()

    def paintEvent(self, event):
        """Enhanced paint event with beautiful detection visualization"""
        if self.current_frame is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(15, 52, 96))
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.setFont(QFont("Arial", 16))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Initializing Camera...")
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Convert frame to QImage
        height, width, channel = self.current_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(self.current_frame.data, width, height,
                        bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()

        # Scale image to widget size while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        widget_size = self.size()
        scaled_pixmap = pixmap.scaled(widget_size, Qt.AspectRatioMode.KeepAspectRatio,
                                     Qt.TransformationMode.SmoothTransformation)

        # Center the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)

        # Calculate scaling factors for bbox
        scale_x = scaled_pixmap.width() / width
        scale_y = scaled_pixmap.height() / height

        # Draw enhanced overlay
        self.draw_enhanced_overlay(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height(), scale_x, scale_y)

    def draw_enhanced_overlay(self, painter: QPainter, img_x: int, img_y: int, img_w: int, img_h: int, scale_x: float, scale_y: float):
        """Draw beautiful enhanced overlay with detection frames"""
        
        # Color mapping for different classes
        colors = {
            'unripe': QColor(76, 175, 80),   # Green
            'ripe': QColor(233, 69, 96),     # Red
            'rotten': QColor(121, 85, 72),   # Brown
            'unknown': QColor(158, 158, 158), # Gray
            'error': QColor(255, 152, 0)     # Orange
        }

        # Get current color
        current_color = colors.get(self.current_label, colors['unknown'])
        
        # Draw detection bounding box with animation
        if self.detection_bbox and self.current_confidence > 0.3:
            bbox_x, bbox_y, bbox_w, bbox_h = self.detection_bbox
            
            # Scale bbox to display coordinates
            scaled_x = int(img_x + bbox_x * scale_x)
            scaled_y = int(img_y + bbox_y * scale_y)
            scaled_w = int(bbox_w * scale_x)
            scaled_h = int(bbox_h * scale_y)
            
            # Animated detection box
            pulse = abs(np.sin(self.animation_alpha)) * 0.3 + 0.7
            animated_color = QColor(current_color)
            animated_color.setAlphaF(pulse)
            
            # Draw main detection box
            pen_width = max(2, int(3 * pulse))
            painter.setPen(QPen(animated_color, pen_width))
            painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
            
            # Draw corner indicators
            corner_size = 20
            corner_pen = QPen(current_color, 4)
            painter.setPen(corner_pen)
            
            # Top-left corner
            painter.drawLine(scaled_x, scaled_y + corner_size, scaled_x, scaled_y)
            painter.drawLine(scaled_x, scaled_y, scaled_x + corner_size, scaled_y)
            
            # Top-right corner
            painter.drawLine(scaled_x + scaled_w - corner_size, scaled_y, scaled_x + scaled_w, scaled_y)
            painter.drawLine(scaled_x + scaled_w, scaled_y, scaled_x + scaled_w, scaled_y + corner_size)
            
            # Bottom-left corner
            painter.drawLine(scaled_x, scaled_y + scaled_h - corner_size, scaled_x, scaled_y + scaled_h)
            painter.drawLine(scaled_x, scaled_y + scaled_h, scaled_x + corner_size, scaled_y + scaled_h)
            
            # Bottom-right corner
            painter.drawLine(scaled_x + scaled_w - corner_size, scaled_y + scaled_h, scaled_x + scaled_w, scaled_y + scaled_h)
            painter.drawLine(scaled_x + scaled_w, scaled_y + scaled_h, scaled_x + scaled_w, scaled_y + scaled_h - corner_size)
            
            # Draw crosshair at center
            center_x = scaled_x + scaled_w // 2
            center_y = scaled_y + scaled_h // 2
            cross_size = 10
            
            painter.setPen(QPen(current_color, 2))
            painter.drawLine(center_x - cross_size, center_y, center_x + cross_size, center_y)
            painter.drawLine(center_x, center_y - cross_size, center_x, center_y + cross_size)

        # Draw modern info overlay
        self.draw_info_overlay(painter, img_x, img_y, img_w, img_h, current_color)

    def draw_info_overlay(self, painter: QPainter, x: int, y: int, w: int, h: int, color: QColor):
        """Draw modern information overlay"""
        
        # Semi-transparent background for top info
        top_gradient = QLinearGradient(0, y, 0, y + 100)
        top_gradient.setColorAt(0, QColor(0, 0, 0, 150))
        top_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        painter.fillRect(x, y, w, 100, QBrush(top_gradient))

        # Semi-transparent background for bottom info
        bottom_gradient = QLinearGradient(0, y + h - 80, 0, y + h)
        bottom_gradient.setColorAt(0, QColor(0, 0, 0, 0))
        bottom_gradient.setColorAt(1, QColor(0, 0, 0, 150))
        painter.fillRect(x, y + h - 80, w, 80, QBrush(bottom_gradient))

        # Detection result with glow effect
        painter.setPen(QPen(Qt.GlobalColor.black, 3))  # Shadow
        painter.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        painter.drawText(x + 22, y + 42, f"{self.current_label.upper()}")
        
        painter.setPen(QPen(color, 2))  # Main text
        painter.drawText(x + 20, y + 40, f"{self.current_label.upper()}")

        # Confidence with progress bar style
        painter.setFont(QFont("Arial", 16))
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        confidence_text = f"Confidence: {self.current_confidence:.1%}"
        painter.drawText(x + 20, y + 70, confidence_text)

        # Confidence bar
        bar_x = x + 20
        bar_y = y + 75
        bar_width = min(200, int(w * 0.3))
        bar_height = 8
        
        # Background bar
        painter.fillRect(bar_x, bar_y, bar_width, bar_height, QColor(50, 50, 50, 150))
        
        # Confidence fill
        fill_width = int(bar_width * self.current_confidence)
        painter.fillRect(bar_x, bar_y, fill_width, bar_height, color)

        # FPS counter (top right)
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.setFont(QFont("Arial", 14))
        fps_text = f"FPS: {self.fps:.1f}"
        fps_rect = painter.fontMetrics().boundingRect(fps_text)
        painter.drawText(x + w - fps_rect.width() - 20, y + 35, fps_text)

        # Timestamp (bottom left)
        painter.setFont(QFont("Arial", 12))
        timestamp = datetime.now().strftime("%H:%M:%S")
        painter.drawText(x + 20, y + h - 25, timestamp)

        # System status indicator (bottom right)
        status_colors = {
            'high': QColor(76, 175, 80),    # Green
            'medium': QColor(255, 193, 7),  # Yellow
            'low': QColor(244, 67, 54)      # Red
        }
        
        if self.current_confidence >= 0.8:
            status_color = status_colors['high']
            status_text = "HIGH"
        elif self.current_confidence >= 0.5:
            status_color = status_colors['medium']
            status_text = "MED"
        else:
            status_color = status_colors['low']
            status_text = "LOW"
        
        painter.setPen(QPen(status_color, 2))
        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        status_rect = painter.fontMetrics().boundingRect(status_text)
        painter.drawText(x + w - status_rect.width() - 20, y + h - 25, status_text)

        # Draw recent detection trail
        if len(self.detection_history) > 1:
            painter.setPen(QPen(color, 1))
            for i, detection in enumerate(self.detection_history[-5:]):  # Last 5 detections
                alpha = (i + 1) / 5 * 0.3  # Fade effect
                trail_color = QColor(color)
                trail_color.setAlphaF(alpha)
                painter.setPen(QPen(trail_color, 1))
                
                if detection['bbox']:
                    bbox_x, bbox_y, bbox_w, bbox_h = detection['bbox']
                    # Scale and draw small trail boxes
                    trail_rect = QRect(
                        int(x + bbox_x * w / self.current_frame.shape[1]),
                        int(y + bbox_y * h / self.current_frame.shape[0]),
                        int(bbox_w * w / self.current_frame.shape[1]),
                        int(bbox_h * h / self.current_frame.shape[0])
                    )
                    painter.drawRect(trail_rect)


class HarvestControlPanel(QWidget):
    """Advanced control panel for harvesting operations"""

    def __init__(self, robot_controller):
        super().__init__()
        self.robot_controller = robot_controller
        self.selected_arms = set()
        self.harvest_mode = "single"  # single, multi, auto
        self.init_ui()

    def init_ui(self):
        """Initialize control panel UI"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Harvest Control Center")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Arm selection
        arm_group = QGroupBox("Robot Arm Selection")
        arm_layout = QGridLayout()

        self.arm_widgets = []
        for i in range(4):
            widget = self.create_arm_widget(i)
            self.arm_widgets.append(widget)
            arm_layout.addWidget(widget, i // 2, i % 2)

        arm_group.setLayout(arm_layout)
        layout.addWidget(arm_group)

        # Harvest mode selection
        mode_group = QGroupBox("Harvest Mode")
        mode_layout = QHBoxLayout()

        self.single_mode_btn = QPushButton("Single Arm")
        self.single_mode_btn.setCheckable(True)
        self.single_mode_btn.setChecked(True)
        self.single_mode_btn.clicked.connect(lambda: self.set_harvest_mode("single"))

        self.multi_mode_btn = QPushButton("Multi Arm")
        self.multi_mode_btn.setCheckable(True)
        self.multi_mode_btn.clicked.connect(lambda: self.set_harvest_mode("multi"))

        self.auto_mode_btn = QPushButton("Full Auto")
        self.auto_mode_btn.setCheckable(True)
        self.auto_mode_btn.setObjectName("primaryButton")
        self.auto_mode_btn.clicked.connect(lambda: self.set_harvest_mode("auto"))

        mode_layout.addWidget(self.single_mode_btn)
        mode_layout.addWidget(self.multi_mode_btn)
        mode_layout.addWidget(self.auto_mode_btn)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Harvest type selection
        type_group = QGroupBox("Harvest Type Selection")
        type_layout = QVBoxLayout()

        # Checkboxes for harvest types
        self.harvest_unripe = QCheckBox("Harvest Unripe (Green)")
        self.harvest_unripe.setChecked(False)

        self.harvest_ripe = QCheckBox("Harvest Ripe (Red)")
        self.harvest_ripe.setChecked(True)

        self.harvest_rotten = QCheckBox("Remove Rotten (Brown)")
        self.harvest_rotten.setChecked(True)

        type_layout.addWidget(self.harvest_unripe)
        type_layout.addWidget(self.harvest_ripe)
        type_layout.addWidget(self.harvest_rotten)

        # Confidence threshold
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(QLabel("Confidence Threshold:"))

        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(80)
        self.threshold_slider.valueChanged.connect(self.update_threshold_label)

        self.threshold_label = QLabel("80%")

        threshold_layout.addWidget(self.threshold_slider)
        threshold_layout.addWidget(self.threshold_label)

        type_layout.addLayout(threshold_layout)
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)

        # Manual control buttons
        control_group = QGroupBox("Manual Control")
        control_layout = QGridLayout()

        self.pick_btn = QPushButton("🤏 Pick")
        self.pick_btn.clicked.connect(lambda: self.manual_action("pick"))

        self.place_btn = QPushButton("📦 Place")
        self.place_btn.clicked.connect(lambda: self.manual_action("place"))

        self.home_btn = QPushButton("🏠 Home")
        self.home_btn.clicked.connect(lambda: self.manual_action("home"))

        self.emergency_btn = QPushButton("🛑 STOP")
        self.emergency_btn.setObjectName("dangerButton")
        self.emergency_btn.clicked.connect(lambda: self.manual_action("stop"))

        control_layout.addWidget(self.pick_btn, 0, 0)
        control_layout.addWidget(self.place_btn, 0, 1)
        control_layout.addWidget(self.home_btn, 1, 0)
        control_layout.addWidget(self.emergency_btn, 1, 1)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        layout.addStretch()

    def create_arm_widget(self, arm_id: int) -> QWidget:
        """Create individual arm control widget"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(widget)

        # Arm button
        btn = QPushButton(f"ARM {arm_id + 1}")
        btn.setCheckable(True)
        btn.clicked.connect(lambda checked: self.toggle_arm(arm_id, checked))

        # Status indicator
        status = QLabel("⚫ Disconnected")
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setTextVisible(False)
        progress.setMaximumHeight(5)

        layout.addWidget(btn)
        layout.addWidget(status)
        layout.addWidget(progress)

        # Store references
        widget.button = btn
        widget.status = status
        widget.progress = progress
        widget.arm_id = arm_id

        return widget

    def toggle_arm(self, arm_id: int, checked: bool):
        """Toggle arm selection"""
        if checked:
            self.selected_arms.add(arm_id)
        else:
            self.selected_arms.discard(arm_id)

        # In single mode, deselect others
        if self.harvest_mode == "single" and checked:
            for i, widget in enumerate(self.arm_widgets):
                if i != arm_id:
                    widget.button.setChecked(False)
                    self.selected_arms.discard(i)

    def set_harvest_mode(self, mode: str):
        """Set harvesting mode"""
        self.harvest_mode = mode

        # Update button states
        self.single_mode_btn.setChecked(mode == "single")
        self.multi_mode_btn.setChecked(mode == "multi")
        self.auto_mode_btn.setChecked(mode == "auto")

        # Clear selection in single mode
        if mode == "single" and len(self.selected_arms) > 1:
            self.selected_arms.clear()
            for widget in self.arm_widgets:
                widget.button.setChecked(False)

    def update_threshold_label(self, value: int):
        """Update threshold label"""
        self.threshold_label.setText(f"{value}%")

    def manual_action(self, action: str):
        """Execute manual action on selected arms"""
        for arm_id in self.selected_arms:
            if action == "stop":
                self.robot_controller.send_command(arm_id, {"cmd": "EMERGENCY_STOP"})
            elif action == "home":
                self.robot_controller.send_command(arm_id, {"cmd": "HOME"})
            elif action == "pick":
                self.robot_controller.send_command(arm_id, {"cmd": "PICK"})
            elif action == "place":
                self.robot_controller.send_command(arm_id, {"cmd": "PLACE"})

    def update_arm_status(self, arm_id: int, status: str, data: dict):
        """Update arm widget status"""
        if 0 <= arm_id < len(self.arm_widgets):
            widget = self.arm_widgets[arm_id]

            status_icons = {
                "connected": "🟢",
                "ready": "🟢",
                "busy": "🟡",
                "error": "🔴",
                "disconnected": "⚫"
            }

            icon = status_icons.get(status, "⚫")
            widget.status.setText(f"{icon} {status.capitalize()}")

            # Update progress if in data
            if "progress" in data:
                widget.progress.setValue(int(data["progress"]))


class StatisticsWidget(QWidget):
    """Real-time statistics and analytics widget"""

    def __init__(self):
        super().__init__()
        self.stats = HarvestStats()
        self.init_ui()

        # Setup update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(1000)  # Update every second

    def init_ui(self):
        """Initialize statistics UI"""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Harvest Statistics")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Stats grid
        stats_group = QGroupBox("Current Session")
        stats_layout = QGridLayout()

        # Create stat labels
        self.total_label = self.create_stat_label("Total Detected:", "0")
        self.unripe_label = self.create_stat_label("Unripe:", "0", QColor(76, 175, 80))
        self.ripe_label = self.create_stat_label("Ripe:", "0", QColor(233, 69, 96))
        self.rotten_label = self.create_stat_label("Rotten:", "0", QColor(121, 85, 72))
        self.success_label = self.create_stat_label("Success Rate:", "0%", QColor(76, 175, 80))
        self.accuracy_label = self.create_stat_label("Avg Confidence:", "0%")

        stats_layout.addWidget(self.total_label[0], 0, 0)
        stats_layout.addWidget(self.total_label[1], 0, 1)
        stats_layout.addWidget(self.unripe_label[0], 1, 0)
        stats_layout.addWidget(self.unripe_label[1], 1, 1)
        stats_layout.addWidget(self.ripe_label[0], 2, 0)
        stats_layout.addWidget(self.ripe_label[1], 2, 1)
        stats_layout.addWidget(self.rotten_label[0], 3, 0)
        stats_layout.addWidget(self.rotten_label[1], 3, 1)
        stats_layout.addWidget(self.success_label[0], 4, 0)
        stats_layout.addWidget(self.success_label[1], 4, 1)
        stats_layout.addWidget(self.accuracy_label[0], 5, 0)
        stats_layout.addWidget(self.accuracy_label[1], 5, 1)

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Performance chart
        chart_group = QGroupBox("Performance Trend")
        chart_layout = QVBoxLayout()

        # Create chart
        self.create_performance_chart()
        chart_layout.addWidget(self.chart_view)

        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group)

        # Export button
        export_btn = QPushButton("📊 Export Statistics")
        export_btn.clicked.connect(self.export_stats)
        layout.addWidget(export_btn)

        layout.addStretch()

    def create_stat_label(self, title: str, value: str, color: QColor = None) -> Tuple[QLabel, QLabel]:
        """Create a statistics label pair"""
        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 12))

        value_label = QLabel(value)
        value_label.setObjectName("statsLabel")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        if color:
            value_label.setStyleSheet(f"color: {color.name()};")

        return title_label, value_label

    def create_performance_chart(self):
        """Create performance trend chart"""
        self.chart = QChart()
        self.chart.setTitle("Detection Confidence Trend")
        self.chart.setAnimationOptions(QChart.AnimationOption.SeriesAnimations)

        # Create series
        self.confidence_series = QLineSeries()
        self.confidence_series.setName("Confidence")

        # Add initial data
        for i in range(20):
            self.confidence_series.append(i, 0)

        self.chart.addSeries(self.confidence_series)

        # Create axes
        axis_x = QValueAxis()
        axis_x.setRange(0, 20)
        axis_x.setLabelFormat("%d")
        axis_x.setTitleText("Time")

        axis_y = QValueAxis()
        axis_y.setRange(0, 100)
        axis_y.setLabelFormat("%d%%")
        axis_y.setTitleText("Confidence")

        self.chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        self.chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        self.confidence_series.attachAxis(axis_x)
        self.confidence_series.attachAxis(axis_y)

        # Create chart view
        self.chart_view = QChartView(self.chart)
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.chart_view.setMinimumHeight(200)

    def update_stats(self, label: str, confidence: float):
        """Update statistics with new detection"""
        self.stats.total_detected += 1

        if label == "unripe":
            self.stats.unripe_count += 1
        elif label == "ripe":
            self.stats.ripe_count += 1
        elif label == "rotten":
            self.stats.rotten_count += 1

        self.stats.accuracy_history.append(confidence)

        # Update chart
        points = self.confidence_series.points()
        for i in range(len(points) - 1):
            points[i].setY(points[i + 1].y())

        if points:
            points[-1].setY(confidence * 100)

        self.confidence_series.replace(points)

    def update_harvest_result(self, success: bool):
        """Update harvest statistics"""
        if success:
            self.stats.harvest_success += 1
        else:
            self.stats.harvest_failed += 1

    def update_display(self):
        """Update statistics display"""
        self.total_label[1].setText(str(self.stats.total_detected))
        self.unripe_label[1].setText(str(self.stats.unripe_count))
        self.ripe_label[1].setText(str(self.stats.ripe_count))
        self.rotten_label[1].setText(str(self.stats.rotten_count))

        # Calculate success rate
        total_harvests = self.stats.harvest_success + self.stats.harvest_failed
        if total_harvests > 0:
            success_rate = (self.stats.harvest_success / total_harvests) * 100
            self.success_label[1].setText(f"{success_rate:.1f}%")

        # Calculate average confidence
        if self.stats.accuracy_history:
            avg_confidence = sum(self.stats.accuracy_history) / len(self.stats.accuracy_history)
            self.accuracy_label[1].setText(f"{avg_confidence:.1f}%")

    def export_stats(self):
        """Export statistics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"harvest_stats_{timestamp}.json"

        stats_dict = {
            "timestamp": timestamp,
            "total_detected": self.stats.total_detected,
            "unripe_count": self.stats.unripe_count,
            "ripe_count": self.stats.ripe_count,
            "rotten_count": self.stats.rotten_count,
            "harvest_success": self.stats.harvest_success,
            "harvest_failed": self.stats.harvest_failed,
            "accuracy_history": list(self.stats.accuracy_history)
        }

        try:
            with open(filename, 'w') as f:
                json.dump(stats_dict, f, indent=2)

            QMessageBox.information(self, "Export Success",
                                  f"Statistics exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                               f"Failed to export statistics: {e}")


class AdvancedMainWindow(QMainWindow):
    """Advanced main window with enhanced UI/UX and file browser"""

    def __init__(self):
        super().__init__()
        self.predictor = OptimizedPredictor()
        self.robot_controller = EnhancedRobotController()
        self.camera_thread = None
        self.current_frame = None
        self.auto_harvest = False

        self.init_ui()
        self.setup_connections()
        self.start_system()

    def init_ui(self):
        """Initialize the modern UI"""
        self.setWindowTitle("🍅 Advanced Tomato Harvesting System v2.0")
        self.setGeometry(100, 100, 1600, 1000)

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

        # Detection info panel
        detection_panel = self.create_detection_panel()
        left_panel.addWidget(detection_panel)

        # Quick actions
        quick_actions = QHBoxLayout()

        self.auto_harvest_btn = QPushButton("🤖 Auto Harvest: OFF")
        self.auto_harvest_btn.setObjectName("primaryButton")
        self.auto_harvest_btn.setCheckable(True)
        self.auto_harvest_btn.clicked.connect(self.toggle_auto_harvest)

        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.clicked.connect(self.capture_image)

        quick_actions.addWidget(self.auto_harvest_btn)
        quick_actions.addWidget(self.capture_btn)

        left_panel.addLayout(quick_actions)
        main_layout.addLayout(left_panel, 2)

        # Right panel - Tabbed interface
        right_panel = QTabWidget()
        right_panel.setTabPosition(QTabWidget.TabPosition.North)

        # Control tab
        self.control_panel = HarvestControlPanel(self.robot_controller)
        right_panel.addTab(self.control_panel, "🎮 Control")

        # Statistics tab
        self.stats_widget = StatisticsWidget()
        right_panel.addTab(self.stats_widget, "📊 Statistics")

        # System log tab
        self.log_widget = self.create_log_widget()
        right_panel.addTab(self.log_widget, "📝 System Log")

        # Settings tab
        self.settings_widget = self.create_enhanced_settings_widget()
        right_panel.addTab(self.settings_widget, "⚙️ Settings")

        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.create_status_bar()

        # Auto harvest timer
        self.auto_timer = QTimer()
        self.auto_timer.timeout.connect(self.auto_harvest_cycle)

    def create_detection_panel(self) -> QWidget:
        """Create detection information panel"""
        panel = QGroupBox("Current Detection")
        layout = QVBoxLayout(panel)

        # Detection result display
        self.detection_result_label = QLabel("Waiting for detection...")
        self.detection_result_label.setObjectName("detectionLabel")
        self.detection_result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_result_label.setStyleSheet("""
            QLabel#detectionLabel {
                background-color: #0f3460;
                border: 2px solid #e94560;
                border-radius: 10px;
                padding: 15px;
                font-size: 18px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.detection_result_label)

        # Confidence meter
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence:"))
        
        self.confidence_progress = QProgressBar()
        self.confidence_progress.setRange(0, 100)
        self.confidence_progress.setTextVisible(True)
        
        confidence_layout.addWidget(self.confidence_progress)
        layout.addLayout(confidence_layout)

        # Bounding box info
        bbox_layout = QGridLayout()
        bbox_layout.addWidget(QLabel("X:"), 0, 0)
        bbox_layout.addWidget(QLabel("Y:"), 0, 2)
        bbox_layout.addWidget(QLabel("W:"), 1, 0)
        bbox_layout.addWidget(QLabel("H:"), 1, 2)

        self.bbox_x_label = QLabel("0")
        self.bbox_y_label = QLabel("0")
        self.bbox_w_label = QLabel("0")
        self.bbox_h_label = QLabel("0")

        bbox_layout.addWidget(self.bbox_x_label, 0, 1)
        bbox_layout.addWidget(self.bbox_y_label, 0, 3)
        bbox_layout.addWidget(self.bbox_w_label, 1, 1)
        bbox_layout.addWidget(self.bbox_h_label, 1, 3)

        bbox_group = QGroupBox("Bounding Box")
        bbox_group.setLayout(bbox_layout)
        layout.addWidget(bbox_group)

        return panel

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

    def create_enhanced_settings_widget(self) -> QWidget:
        """Create enhanced settings widget with file browser"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # AI Model Settings
        model_group = QGroupBox("AI Model Configuration")
        model_layout = QVBoxLayout()

        # Model file browser
        model_file_layout = QHBoxLayout()
        model_file_layout.addWidget(QLabel("Model File (.h5):"))
        
        self.model_path_edit = QLineEdit("tomato.h5")
        self.model_path_edit.setPlaceholderText("Path to .h5 model file")
        model_file_layout.addWidget(self.model_path_edit)
        
        browse_model_btn = QPushButton("Browse...")
        browse_model_btn.setObjectName("browseButton")
        browse_model_btn.clicked.connect(self.browse_model_file)
        model_file_layout.addWidget(browse_model_btn)
        
        model_layout.addLayout(model_file_layout)

        # Labels file browser
        labels_file_layout = QHBoxLayout()
        labels_file_layout.addWidget(QLabel("Labels File (.txt):"))
        
        self.labels_path_edit = QLineEdit("labels.txt")
        self.labels_path_edit.setPlaceholderText("Path to labels.txt file")
        labels_file_layout.addWidget(self.labels_path_edit)
        
        browse_labels_btn = QPushButton("Browse...")
        browse_labels_btn.setObjectName("browseButton")
        browse_labels_btn.clicked.connect(self.browse_labels_file)
        labels_file_layout.addWidget(browse_labels_btn)
        
        model_layout.addLayout(labels_file_layout)

        # Load model button
        load_model_btn = QPushButton("🔄 Load Model & Labels")
        load_model_btn.setObjectName("primaryButton")
        load_model_btn.clicked.connect(self.load_model_and_labels)
        model_layout.addWidget(load_model_btn)

        # Model info display
        self.model_info_label = QLabel("Model: Not loaded")
        self.model_info_label.setWordWrap(True)
        model_layout.addWidget(self.model_info_label)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Camera Settings
        camera_group = QGroupBox("Camera Settings")
        camera_layout = QVBoxLayout()

        # Resolution selector
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Resolution:"))

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["640x480", "800x600", "1280x720", "1920x1080"])
        self.resolution_combo.currentTextChanged.connect(self.change_resolution)

        res_layout.addWidget(self.resolution_combo)
        camera_layout.addLayout(res_layout)

        # FPS selector
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

        # Robot Settings
        robot_group = QGroupBox("Robot Settings")
        robot_layout = QVBoxLayout()

        # Connection settings
        connection_layout = QHBoxLayout()
        connection_layout.addWidget(QLabel("Baud Rate:"))
        
        self.baudrate_combo = QComboBox()
        self.baudrate_combo.addItems(["9600", "57600", "115200", "230400"])
        self.baudrate_combo.setCurrentText("115200")
        connection_layout.addWidget(self.baudrate_combo)

        robot_layout.addLayout(connection_layout)

        # Reconnect button
        reconnect_btn = QPushButton("🔌 Reconnect All Arms")
        reconnect_btn.clicked.connect(self.reconnect_arms)
        robot_layout.addWidget(reconnect_btn)

        robot_group.setLayout(robot_layout)
        layout.addWidget(robot_group)

        layout.addStretch()
        return widget

    def browse_model_file(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Keras Model Files (*.h5);;All Files (*)"
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
        """Load model and labels from specified paths"""
        model_path = self.model_path_edit.text().strip()
        labels_path = self.labels_path_edit.text().strip()

        if not model_path or not labels_path:
            QMessageBox.warning(self, "Missing Files", "Please specify both model and labels files.")
            return

        # Update predictor paths
        self.predictor.set_model_path(model_path)
        self.predictor.set_labels_path(labels_path)

        # Update info display
        if self.predictor.model:
            self.model_info_label.setText(f"Model: {Path(model_path).name}\nLabels: {Path(labels_path).name}\nClasses: {', '.join(self.predictor.classes)}")
            self.model_status.setText("🤖 Model: Ready")
            self.log_message("Model", f"Successfully loaded {Path(model_path).name}")
        else:
            self.model_info_label.setText("Model: Failed to load")
            self.model_status.setText("🤖 Model: Error")
            self.log_message("Model", f"Failed to load {Path(model_path).name}")

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
        self.model_status = QLabel("🤖 Model: Loading...")
        self.status_bar.addWidget(self.model_status)

        # Connection status
        self.connection_status = QLabel("🔌 Arms: 0/4")
        self.status_bar.addPermanentWidget(self.connection_status)

    def setup_connections(self):
        """Setup all signal/slot connections"""
        # Robot controller signals
        self.robot_controller.status_update.connect(self.handle_arm_status)
        self.robot_controller.harvest_complete.connect(self.handle_harvest_complete)

        # Predictor signals
        self.predictor.prediction_complete.connect(self.handle_prediction)

    def start_system(self):
        """Start all system components"""
        # Start camera
        self.camera_thread = PiCameraThread()
        self.camera_thread.frame_ready.connect(self.process_frame)
        self.camera_thread.start()

        # Connect robot arms
        QTimer.singleShot(1000, self.robot_controller.connect_all)

        # Update status
        self.log_message("System", "Tomato Harvesting System v2.0 started")
        self.camera_status.setText("📷 Camera: Active")

        if self.predictor.model:
            self.model_status.setText("🤖 Model: Ready")
        else:
            self.model_status.setText("🤖 Model: Not loaded")

    def process_frame(self, frame: np.ndarray):
        """Process camera frame"""
        self.current_frame = frame

        # Run prediction
        label, confidence, fps, bbox = self.predictor.predict(frame)

        # Update camera widget
        self.camera_widget.update_frame(frame, label, confidence, fps, bbox)

        # Update detection panel
        self.update_detection_panel(label, confidence, bbox)

        # Update statistics
        if label != "unknown" and confidence > 0.5:
            self.stats_widget.update_stats(label, confidence)

    def update_detection_panel(self, label: str, confidence: float, bbox: Tuple[int, int, int, int]):
        """Update detection information panel"""
        # Update result label with color coding
        colors = {
            'unripe': '#4CAF50',   # Green
            'ripe': '#e94560',     # Red
            'rotten': '#795548',   # Brown
            'unknown': '#9E9E9E',  # Gray
            'error': '#FF9800'     # Orange
        }
        
        color = colors.get(label, colors['unknown'])
        self.detection_result_label.setText(f"{label.upper()}")
        self.detection_result_label.setStyleSheet(f"""
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

        # Update confidence meter
        self.confidence_progress.setValue(int(confidence * 100))

        # Update bounding box info
        if bbox:
            x, y, w, h = bbox
            self.bbox_x_label.setText(str(x))
            self.bbox_y_label.setText(str(y))
            self.bbox_w_label.setText(str(w))
            self.bbox_h_label.setText(str(h))

    def handle_prediction(self, label: str, confidence: float, image: np.ndarray, fps: float, bbox: Tuple[int, int, int, int]):
        """Handle prediction results"""
        # Auto harvest if enabled
        if self.auto_harvest and confidence >= self.get_confidence_threshold():
            if self.should_harvest(label):
                position = self.calculate_tomato_position(bbox)
                for arm_id in self.control_panel.selected_arms:
                    self.robot_controller.smart_harvest(arm_id, label, position)

    def handle_arm_status(self, arm_id: int, status: str, data: dict):
        """Handle arm status updates"""
        self.control_panel.update_arm_status(arm_id, status, data)

        # Update connection count
        connected = sum(1 for s in self.robot_controller.arm_status.values()
                       if s in ["ready", "connected"])
        self.connection_status.setText(f"🔌 Arms: {connected}/4")

        # Log
        self.log_message(f"Arm {arm_id}", f"{status}: {data}")

    def handle_harvest_complete(self, arm_id: int, tomato_type: str, success: bool):
        """Handle harvest completion"""
        self.stats_widget.update_harvest_result(success)

        status = "Success" if success else "Failed"
        self.log_message(f"Arm {arm_id}", f"Harvest {tomato_type}: {status}")

    def toggle_auto_harvest(self):
        """Toggle automatic harvesting"""
        self.auto_harvest = not self.auto_harvest

        if self.auto_harvest:
            self.auto_harvest_btn.setText("🤖 Auto Harvest: ON")
            self.auto_timer.start(1000)  # Check every second
            self.log_message("System", "Auto harvest enabled")
        else:
            self.auto_harvest_btn.setText("🤖 Auto Harvest: OFF")
            self.auto_timer.stop()
            self.log_message("System", "Auto harvest disabled")

    def auto_harvest_cycle(self):
        """Automatic harvest cycle"""
        # This is handled in handle_prediction when auto_harvest is True
        pass

    def should_harvest(self, label: str) -> bool:
        """Check if tomato type should be harvested"""
        if label == "unripe":
            return self.control_panel.harvest_unripe.isChecked()
        elif label == "ripe":
            return self.control_panel.harvest_ripe.isChecked()
        elif label == "rotten":
            return self.control_panel.harvest_rotten.isChecked()
        return False

    def get_confidence_threshold(self) -> float:
        """Get confidence threshold from control panel"""
        return self.control_panel.threshold_slider.value() / 100.0

    def calculate_tomato_position(self, bbox: Tuple[int, int, int, int]) -> Dict:
        """Calculate tomato position from bounding box"""
        if bbox:
            x, y, w, h = bbox
            center_x = x + w // 2
            center_y = y + h // 2
            return {
                "x": center_x,
                "y": center_y,
                "width": w,
                "height": h,
                "bbox": bbox
            }
        else:
            # Default to image center
            if self.current_frame is not None:
                height, width = self.current_frame.shape[:2]
                return {
                    "x": width // 2,
                    "y": height // 2,
                    "width": width // 4,
                    "height": height // 4,
                    "bbox": (width//4, height//4, width//2, height//2)
                }
            return {"x": 320, "y": 240, "width": 160, "height": 120, "bbox": (160, 120, 160, 120)}

    def capture_image(self):
        """Capture current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.log_message("System", f"Image saved: {filename}")

    def change_resolution(self, resolution: str):
        """Change camera resolution"""
        if self.camera_thread:
            try:
                width, height = resolution.split('x')
                self.camera_thread.set_resolution(int(width), int(height))
                self.log_message("Camera", f"Resolution changed to {resolution}")
            except:
                self.log_message("Camera", f"Failed to change resolution to {resolution}")

    def change_fps(self, fps: int):
        """Change camera FPS"""
        if self.camera_thread:
            self.camera_thread.set_fps(fps)
            self.log_message("Camera", f"FPS changed to {fps}")

    def reconnect_arms(self):
        """Reconnect all robot arms"""
        self.robot_controller.disconnect_all()
        QTimer.singleShot(500, self.robot_controller.connect_all)
        self.log_message("System", "Reconnecting all arms...")

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

        # Disconnect arms
        self.robot_controller.disconnect_all()

        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style

    # Set application icon
    app.setApplicationName("Advanced Tomato Harvesting System")
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

    # Check PyQt6
    try:
        from PyQt6 import QtCore
        qt_version = f"PyQt6 {QtCore.PYQT_VERSION_STR}"
    except:
        qt_version = "PyQt6 not properly installed"

    logger.info(f"Starting with {qt_version}")

    # Create and show main window
    window = AdvancedMainWindow()
    window.showMaximized()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
