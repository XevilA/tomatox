#!/usr/bin/env python3
"""
Advanced Tomato Harvesting Robot Control System for Raspberry Pi 4
Optimized for Pi Camera with Modern UI/UX Design
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
                             QProgressBar, QFrame, QGraphicsDropShadowEffect)
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

    QTextEdit {
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
    """


class OptimizedPredictor(QObject):
    """Optimized Model Predictor with preprocessing for Pi Camera"""
    prediction_complete = pyqtSignal(str, float, np.ndarray, float)  # label, confidence, image, fps

    def __init__(self, model_path: str = 'tomato.h5'):
        super().__init__()
        self.model = None
        self.model_path = model_path
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

        if TF_AVAILABLE:
            self.load_model()

    def load_model(self):
        """Load and optimize the Keras model"""
        try:
            self.model = keras.models.load_model(self.model_path)
            # Optimize for inference
            self.model = tf.lite.TFLiteConverter.from_keras_model(self.model).convert()
            self.interpreter = tf.lite.Interpreter(model_content=self.model)
            self.interpreter.allocate_tensors()

            input_details = self.interpreter.get_input_details()
            self.input_shape = input_details[0]['shape'][1:3]

            logger.info(f"Model optimized and loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback to regular model
            try:
                self.model = keras.models.load_model(self.model_path)
                input_shape = self.model.input_shape
                self.input_shape = (input_shape[1], input_shape[2])
            except:
                self.model = None

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

        for class_name, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            ratio = cv2.countNonZero(mask) / (image.shape[0] * image.shape[1])

            if ratio > 0.3:  # If more than 30% matches the color range
                return class_name

        return None

    def predict(self, image: np.ndarray) -> Tuple[str, float, float]:
        """Predict with performance metrics"""
        start_time = time.time()

        if self.model is None:
            return "unknown", 0.0, 0.0

        try:
            # Quick color check first
            quick_class = self.quick_color_check(image)

            # Preprocess
            preprocessed = self.preprocess_image(image)

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
            label = self.classes[class_idx]

            # Boost confidence if color check matches
            if quick_class == label:
                confidence = min(confidence * 1.1, 1.0)

            # Calculate FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0

            self.prediction_complete.emit(label, confidence, image, fps)

            return label, confidence, fps

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "error", 0.0, 0.0


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
    """Modern camera display widget with overlay graphics"""

    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.current_label = "Waiting..."
        self.current_confidence = 0.0
        self.fps = 0.0
        self.detection_boxes = []
        self.init_ui()

    def init_ui(self):
        """Initialize UI components"""
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #0f3460; border-radius: 15px;")

    def update_frame(self, frame: np.ndarray, label: str, confidence: float, fps: float):
        """Update display with new frame and detection info"""
        self.current_frame = frame
        self.current_label = label
        self.current_confidence = confidence
        self.fps = fps
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

        # Center the image
        x = (self.width() - scaled_pixmap.width()) // 2
        y = (self.height() - scaled_pixmap.height()) // 2
        painter.drawPixmap(x, y, scaled_pixmap)

        # Draw modern overlay
        self.draw_overlay(painter, x, y, scaled_pixmap.width(), scaled_pixmap.height())

    def draw_overlay(self, painter: QPainter, x: int, y: int, w: int, h: int):
        """Draw modern overlay graphics"""
        # Semi-transparent overlay for info
        overlay_color = QColor(0, 0, 0, 100)
        painter.fillRect(x, y, w, 80, overlay_color)
        painter.fillRect(x, y + h - 60, w, 60, overlay_color)

        # Classification result with color coding
        colors = {
            'unripe': QColor(76, 175, 80),   # Green
            'ripe': QColor(233, 69, 96),     # Red
            'rotten': QColor(121, 85, 72),   # Brown
            'unknown': QColor(158, 158, 158)  # Gray
        }

        color = colors.get(self.current_label, colors['unknown'])

        # Top info bar
        painter.setPen(QPen(color, 3))
        painter.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        painter.drawText(x + 20, y + 35, f"{self.current_label.upper()}")

        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.setFont(QFont("Arial", 14))
        painter.drawText(x + 20, y + 60, f"Confidence: {self.current_confidence:.1%}")

        # FPS counter
        painter.drawText(x + w - 100, y + 35, f"FPS: {self.fps:.1f}")

        # Bottom status bar
        painter.setPen(QPen(Qt.GlobalColor.white, 2))
        painter.setFont(QFont("Arial", 12))
        painter.drawText(x + 20, y + h - 25, datetime.now().strftime("%H:%M:%S"))

        # Detection box with animation
        if self.current_confidence > 0.5:
            pen = QPen(color, 4)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(x + 50, y + 100, w - 100, h - 200)

            # Confidence bar
            bar_width = int((w - 100) * self.current_confidence)
            painter.fillRect(x + 50, y + h - 90, bar_width, 20, color)


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
    """Advanced main window with modern UI/UX"""

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
        self.setWindowTitle("🍅 Advanced Tomato Harvesting System")
        self.setGeometry(100, 100, 1400, 900)

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
        self.settings_widget = self.create_settings_widget()
        right_panel.addTab(self.settings_widget, "⚙️ Settings")

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

    def create_settings_widget(self) -> QWidget:
        """Create settings widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Camera settings
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

        self.fps_slider = QSlider(Qt.Orientation.Horizontal)
        self.fps_slider.setRange(10, 60)
        self.fps_slider.setValue(30)

        self.fps_label = QLabel("30")
        self.fps_slider.valueChanged.connect(
            lambda v: self.fps_label.setText(str(v))
        )

        fps_layout.addWidget(self.fps_slider)
        fps_layout.addWidget(self.fps_label)
        camera_layout.addLayout(fps_layout)

        camera_group.setLayout(camera_layout)
        layout.addWidget(camera_group)

        # Model settings
        model_group = QGroupBox("AI Model Settings")
        model_layout = QVBoxLayout()

        # Model path
        model_path_layout = QHBoxLayout()
        model_path_layout.addWidget(QLabel("Model:"))
        self.model_path_label = QLabel("tomato.h5")
        model_path_layout.addWidget(self.model_path_label)

        reload_model_btn = QPushButton("Reload Model")
        reload_model_btn.clicked.connect(self.reload_model)
        model_path_layout.addWidget(reload_model_btn)

        model_layout.addLayout(model_path_layout)
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Robot settings
        robot_group = QGroupBox("Robot Settings")
        robot_layout = QVBoxLayout()

        # Reconnect button
        reconnect_btn = QPushButton("Reconnect All Arms")
        reconnect_btn.clicked.connect(self.reconnect_arms)
        robot_layout.addWidget(reconnect_btn)

        robot_group.setLayout(robot_layout)
        layout.addWidget(robot_group)

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
        self.log_message("System", "Tomato Harvesting System started")
        self.camera_status.setText("📷 Camera: Active")

        if self.predictor.model:
            self.model_status.setText("🤖 Model: Ready")
        else:
            self.model_status.setText("🤖 Model: Not loaded")

    def process_frame(self, frame: np.ndarray):
        """Process camera frame"""
        self.current_frame = frame

        # Run prediction
        label, confidence, fps = self.predictor.predict(frame)

        # Update camera widget
        self.camera_widget.update_frame(frame, label, confidence, fps)

        # Update statistics
        if label != "unknown" and confidence > 0.5:
            self.stats_widget.update_stats(label, confidence)

    def handle_prediction(self, label: str, confidence: float, image: np.ndarray, fps: float):
        """Handle prediction results"""
        # Auto harvest if enabled
        if self.auto_harvest and confidence >= self.get_confidence_threshold():
            if self.should_harvest(label):
                position = self.calculate_tomato_position(image)
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

    def calculate_tomato_position(self, image: np.ndarray) -> Dict:
        """Calculate tomato position in image (simplified)"""
        height, width = image.shape[:2]
        return {
            "x": width // 2,
            "y": height // 2,
            "width": width,
            "height": height
        }

    def capture_image(self):
        """Capture current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.current_frame)
            self.log_message("System", f"Image saved: {filename}")

    def change_resolution(self, resolution: str):
        """Change camera resolution"""
        # Implementation depends on camera type
        self.log_message("Camera", f"Resolution changed to {resolution}")

    def reload_model(self):
        """Reload AI model"""
        self.predictor.load_model()
        if self.predictor.model:
            self.model_status.setText("🤖 Model: Ready")
            self.log_message("Model", "Model reloaded successfully")
        else:
            self.model_status.setText("🤖 Model: Failed")
            self.log_message("Model", "Failed to reload model")

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
    app.setApplicationName("Tomato Harvesting System")
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
