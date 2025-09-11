#!/usr/bin/env python3
"""
Advanced Tomato Harvesting System with Detection Frames
Enhanced with file browsing for models and labels
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

# PyQt6 imports
from PyQt6.QtCore import (Qt, QThread, QTimer, pyqtSignal, QObject, QPointF)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QTextEdit,
                             QGroupBox, QGridLayout, QStatusBar, QMessageBox,
                             QSlider, QComboBox, QCheckBox, QTabWidget,
                             QProgressBar, QFrame, QFileDialog, QSpinBox)
from PyQt6.QtGui import (QImage, QPixmap, QFont, QPalette, QColor,
                        QPainter, QPen, QBrush)

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result"""
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    label: str
    confidence: float
    center: Tuple[int, int]
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
    detections_history: deque = None

    def __post_init__(self):
        if self.detections_history is None:
            self.detections_history = deque(maxlen=100)


class TomatoDetector(QObject):
    """Enhanced detector with real-time detection frames"""
    detection_complete = pyqtSignal(list, float)  # detections, fps
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.model_path = None
        self.labels_path = None
        self.labels = ['unripe', 'ripe', 'rotten']
        self.confidence_threshold = 0.5
        self.min_area = 500
        
        # Color ranges for detection (HSV)
        self.color_ranges = {
            'unripe': [(35, 50, 50), (85, 255, 255)],    # Green
            'ripe': [(0, 50, 50), (10, 255, 255)],       # Red
            'ripe2': [(170, 50, 50), (180, 255, 255)],   # Red wrap
            'rotten': [(10, 50, 20), (25, 255, 150)]     # Brown
        }
        
    def load_model(self, model_path: str) -> bool:
        """Load TensorFlow/Keras model from file"""
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
            
        try:
            self.model_path = model_path
            self.model = keras.models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def load_labels(self, labels_path: str) -> bool:
        """Load labels from text file"""
        try:
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
            self.labels_path = labels_path
            logger.info(f"Loaded {len(self.labels)} labels from {labels_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load labels: {e}")
            return False
    
    def detect_and_classify(self, frame: np.ndarray) -> List[Detection]:
        """Detect tomatoes and return bounding boxes with classification"""
        detections = []
        height, width = frame.shape[:2]
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Process each color range
        for label, (lower, upper) in self.color_ranges.items():
            if label == 'ripe2':
                label = 'ripe'  # Merge red ranges
                
            # Create mask
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate confidence based on shape
                    confidence = self.calculate_shape_confidence(contour, w, h)
                    
                    # If model available, use it for classification
                    if self.model and confidence > 0.3:
                        roi = frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            predicted_label, model_conf = self.classify_roi(roi)
                            if model_conf > confidence:
                                label = predicted_label
                                confidence = model_conf
                    
                    if confidence > self.confidence_threshold:
                        detection = Detection(
                            bbox=(x, y, w, h),
                            label=label,
                            confidence=confidence,
                            center=(x + w//2, y + h//2),
                            timestamp=time.time()
                        )
                        detections.append(detection)
        
        # Remove overlapping detections
        detections = self.non_max_suppression(detections)
        return detections
    
    def calculate_shape_confidence(self, contour, w, h) -> float:
        """Calculate confidence based on shape properties"""
        # Aspect ratio (tomatoes are roughly circular)
        aspect_ratio = w / h if h > 0 else 0
        ar_score = 1.0 - abs(1.0 - aspect_ratio) * 0.5
        
        # Circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        # Combine scores
        confidence = (circularity * 0.7 + ar_score * 0.3)
        return min(max(confidence, 0.0), 1.0)
    
    def classify_roi(self, roi: np.ndarray) -> Tuple[str, float]:
        """Classify ROI using loaded model"""
        if not self.model:
            return "unknown", 0.0
            
        try:
            # Preprocess ROI
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
            logger.error(f"ROI classification error: {e}")
            return "unknown", 0.0
    
    def non_max_suppression(self, detections: List[Detection], iou_threshold: float = 0.5) -> List[Detection]:
        """Remove overlapping detections"""
        if len(detections) <= 1:
            return detections
        
        # Sort by confidence
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping
            remaining = []
            for det in detections:
                if self.calculate_iou(best.bbox, det.bbox) < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep
    
    def calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0


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
        
        # Try to open camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            # Generate test frames if no camera
            self.run_test_mode()
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            time.sleep(0.033)  # ~30 FPS
        
        self.cap.release()
    
    def run_test_mode(self):
        """Generate test frames when no camera available"""
        frame_count = 0
        while self.running:
            # Create test frame
            frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
            
            # Add text
            cv2.putText(frame, "TEST MODE - No Camera", (150, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Add moving circles (simulated tomatoes)
            # Unripe (green)
            x1 = int(200 + 50 * np.sin(frame_count * 0.05))
            y1 = 200
            cv2.circle(frame, (x1, y1), 30, (0, 255, 0), -1)
            
            # Ripe (red)
            x2 = int(400 + 50 * np.cos(frame_count * 0.03))
            y2 = 250
            cv2.circle(frame, (x2, y2), 35, (0, 0, 255), -1)
            
            # Rotten (brown)
            x3 = 320
            y3 = int(350 + 30 * np.sin(frame_count * 0.04))
            cv2.circle(frame, (x3, y3), 25, (42, 42, 165), -1)
            
            frame_count += 1
            self.frame_ready.emit(frame)
            time.sleep(0.033)
    
    def stop(self):
        """Stop capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait()


class CameraWidget(QWidget):
    """Camera display widget with detection overlay"""
    
    def __init__(self):
        super().__init__()
        self.current_frame = None
        self.detections = []
        self.fps = 0
        self.show_detections = True
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a1a; border: 2px solid #333;")
        
    def update_frame(self, frame: np.ndarray, detections: List[Detection] = None, fps: float = 0):
        """Update displayed frame with detections"""
        self.current_frame = frame
        if detections is not None:
            self.detections = detections
        self.fps = fps
        self.update()
    
    def paintEvent(self, event):
        """Paint event with detection overlay"""
        if self.current_frame is None:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw frame
        height, width, channel = self.current_frame.shape
        bytes_per_line = 3 * width
        
        # Create QImage with detection boxes drawn
        display_frame = self.current_frame.copy()
        
        if self.show_detections:
            # Draw detection boxes on frame
            for det in self.detections:
                self.draw_detection(display_frame, det)
        
        # Convert to QImage
        q_image = QImage(display_frame.data, width, height, bytes_per_line,
                        QImage.Format.Format_RGB888).rgbSwapped()
        
        # Scale to widget size
        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        
        # Center image
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)
        
        # Draw overlay info
        self.draw_overlay_info(painter)
    
    def draw_detection(self, frame: np.ndarray, detection: Detection):
        """Draw detection box on frame"""
        x, y, w, h = detection.bbox
        
        # Choose color based on label
        colors = {
            'unripe': (0, 255, 0),     # Green
            'ripe': (0, 0, 255),       # Red  
            'rotten': (139, 69, 19),   # Brown
            'unknown': (128, 128, 128)  # Gray
        }
        color = colors.get(detection.label, colors['unknown'])
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw label background
        label_text = f"{detection.label}: {detection.confidence:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        
        # Background rectangle for text
        cv2.rectangle(frame, (x, y - text_height - 10), 
                     (x + text_width + 10, y), color, -1)
        
        # Draw label text
        cv2.putText(frame, label_text, (x + 5, y - 5),
                   font, font_scale, (255, 255, 255), thickness)
        
        # Draw center point
        cv2.circle(frame, detection.center, 5, color, -1)
        cv2.circle(frame, detection.center, 7, (255, 255, 255), 1)
    
    def draw_overlay_info(self, painter: QPainter):
        """Draw overlay information"""
        # Setup font
        font = QFont("Arial", 12, QFont.Weight.Bold)
        painter.setFont(font)
        
        # Background for info
        info_bg = QColor(0, 0, 0, 180)
        painter.fillRect(0, 0, self.width(), 40, info_bg)
        
        # FPS
        painter.setPen(QPen(QColor(0, 255, 0)))
        painter.drawText(10, 25, f"FPS: {self.fps:.1f}")
        
        # Detection count by type
        counts = {'unripe': 0, 'ripe': 0, 'rotten': 0}
        for det in self.detections:
            if det.label in counts:
                counts[det.label] += 1
        
        # Draw counts
        x_pos = 150
        colors = {
            'unripe': QColor(0, 255, 0),
            'ripe': QColor(255, 0, 0),
            'rotten': QColor(139, 69, 19)
        }
        
        for label, count in counts.items():
            if count > 0:
                painter.setPen(QPen(colors[label]))
                painter.drawText(x_pos, 25, f"{label}: {count}")
                x_pos += 100
        
        # Total detections
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(self.width() - 100, 25, f"Total: {len(self.detections)}")


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        self.detector = TomatoDetector()
        self.camera_thread = None
        self.stats = HarvestStats()
        self.current_frame = None
        self.init_ui()
        
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("🍅 Tomato Harvesting System - Detection & Analysis")
        self.setGeometry(100, 100, 1200, 700)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QWidget { background-color: #2b2b2b; color: white; }
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
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px;
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
            QLabel { color: white; }
            QTextEdit {
                background-color: #1a1a1a;
                color: white;
                border: 1px solid #555;
            }
        """)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        main_layout = QHBoxLayout(central)
        
        # Left panel - Camera
        left_panel = QVBoxLayout()
        
        # Camera widget
        self.camera_widget = CameraWidget()
        left_panel.addWidget(self.camera_widget)
        
        # Camera controls
        cam_controls = QHBoxLayout()
        
        self.start_btn = QPushButton("📷 Start Camera")
        self.start_btn.clicked.connect(self.toggle_camera)
        cam_controls.addWidget(self.start_btn)
        
        self.detection_btn = QPushButton("🔍 Detection: ON")
        self.detection_btn.setCheckable(True)
        self.detection_btn.setChecked(True)
        self.detection_btn.clicked.connect(self.toggle_detection)
        cam_controls.addWidget(self.detection_btn)
        
        self.capture_btn = QPushButton("📸 Capture")
        self.capture_btn.clicked.connect(self.capture_image)
        cam_controls.addWidget(self.capture_btn)
        
        left_panel.addLayout(cam_controls)
        main_layout.addLayout(left_panel, 2)
        
        # Right panel - Controls
        right_panel = QVBoxLayout()
        
        # Model loading section
        model_group = QGroupBox("AI Model Configuration")
        model_layout = QVBoxLayout()
        
        # Model file browser
        model_file_layout = QHBoxLayout()
        self.model_label = QLabel("Model: Not loaded")
        model_file_layout.addWidget(self.model_label)
        
        browse_model_btn = QPushButton("📁 Browse Model (.h5)")
        browse_model_btn.clicked.connect(self.browse_model)
        model_file_layout.addWidget(browse_model_btn)
        
        model_layout.addLayout(model_file_layout)
        
        # Labels file browser
        labels_file_layout = QHBoxLayout()
        self.labels_label = QLabel("Labels: Default (3 classes)")
        labels_file_layout.addWidget(self.labels_label)
        
        browse_labels_btn = QPushButton("📁 Browse Labels (.txt)")
        browse_labels_btn.clicked.connect(self.browse_labels)
        labels_file_layout.addWidget(browse_labels_btn)
        
        model_layout.addLayout(labels_file_layout)
        
        # Loaded classes display
        self.classes_text = QTextEdit()
        self.classes_text.setMaximumHeight(80)
        self.classes_text.setReadOnly(True)
        self.classes_text.setPlainText("0: unripe\n1: ripe\n2: rotten")
        model_layout.addWidget(QLabel("Loaded Classes:"))
        model_layout.addWidget(self.classes_text)
        
        model_group.setLayout(model_layout)
        right_panel.addWidget(model_group)
        
        # Detection settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        # Confidence threshold
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("Confidence:"))
        
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        thresh_layout.addWidget(self.threshold_slider)
        
        self.threshold_label = QLabel("50%")
        thresh_layout.addWidget(self.threshold_label)
        
        settings_layout.addLayout(thresh_layout)
        
        # Min area
        area_layout = QHBoxLayout()
        area_layout.addWidget(QLabel("Min Area:"))
        
        self.area_spin = QSpinBox()
        self.area_spin.setRange(100, 5000)
        self.area_spin.setValue(500)
        self.area_spin.setSuffix(" px")
        self.area_spin.valueChanged.connect(self.update_min_area)
        area_layout.addWidget(self.area_spin)
        
        settings_layout.addLayout(area_layout)
        
        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)
        
        # Results display
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMaximumHeight(200)
        results_layout.addWidget(self.results_text)
        
        # Statistics
        stats_layout = QGridLayout()
        
        self.total_label = QLabel("Total: 0")
        self.unripe_label = QLabel("Unripe: 0")
        self.ripe_label = QLabel("Ripe: 0")
        self.rotten_label = QLabel("Rotten: 0")
        
        stats_layout.addWidget(self.total_label, 0, 0)
        stats_layout.addWidget(self.unripe_label, 0, 1)
        stats_layout.addWidget(self.ripe_label, 1, 0)
        stats_layout.addWidget(self.rotten_label, 1, 1)
        
        results_layout.addLayout(stats_layout)
        
        # Clear results button
        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_results)
        results_layout.addWidget(clear_btn)
        
        results_group.setLayout(results_layout)
        right_panel.addWidget(results_group)
        
        right_panel.addStretch()
        main_layout.addLayout(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Detection timer
        self.detection_timer = QTimer()
        self.detection_timer.timeout.connect(self.process_detection)
        
    def browse_model(self):
        """Browse for model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "",
            "Model Files (*.h5 *.keras *.pb);;All Files (*.*)"
        )
        
        if file_path:
            if self.detector.load_model(file_path):
                self.model_label.setText(f"Model: {Path(file_path).name}")
                self.statusBar().showMessage(f"Model loaded: {Path(file_path).name}")
                QMessageBox.information(self, "Success", "Model loaded successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to load model!")
    
    def browse_labels(self):
        """Browse for labels file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Labels File", "",
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if file_path:
            if self.detector.load_labels(file_path):
                self.labels_label.setText(f"Labels: {Path(file_path).name}")
                
                # Update classes display
                classes_text = ""
                for i, label in enumerate(self.detector.labels):
                    classes_text += f"{i}: {label}\n"
                self.classes_text.setPlainText(classes_text)
                
                self.statusBar().showMessage(f"Labels loaded: {len(self.detector.labels)} classes")
                QMessageBox.information(self, "Success", "Labels loaded successfully!")
            else:
                QMessageBox.critical(self, "Error", "Failed to load labels!")
    
    def toggle_camera(self):
        """Start/stop camera"""
        if self.camera_thread is None:
            self.camera_thread = CameraThread()
            self.camera_thread.frame_ready.connect(self.process_frame)
            self.camera_thread.start()
            self.start_btn.setText("⏹️ Stop Camera")
            self.detection_timer.start(100)  # Process every 100ms
            self.statusBar().showMessage("Camera started")
        else:
            self.detection_timer.stop()
            self.camera_thread.stop()
            self.camera_thread = None
            self.start_btn.setText("📷 Start Camera")
            self.statusBar().showMessage("Camera stopped")
    
    def process_frame(self, frame: np.ndarray):
        """Store current frame"""
        self.current_frame = frame
    
    def process_detection(self):
        """Process detection on current frame"""
        if self.current_frame is None:
            return
        
        start_time = time.time()
        
        # Detect tomatoes
        detections = self.detector.detect_and_classify(self.current_frame)
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        
        # Update display
        self.camera_widget.update_frame(self.current_frame, detections, fps)
        
        # Update statistics
        self.update_statistics(detections)
        
        # Update results text
        self.update_results(detections)
    
    def update_statistics(self, detections: List[Detection]):
        """Update detection statistics"""
        # Count by type
        counts = {'unripe': 0, 'ripe': 0, 'rotten': 0}
        for det in detections:
            if det.label in counts:
                counts[det.label] += 1
                
        # Update labels
        self.total_label.setText(f"Total: {len(detections)}")
        self.unripe_label.setText(f"Unripe: {counts['unripe']}")
        self.ripe_label.setText(f"Ripe: {counts['ripe']}")
        self.rotten_label.setText(f"Rotten: {counts['rotten']}")
        
        # Update history
        self.stats.total_detected = len(detections)
        self.stats.unripe_count = counts['unripe']
        self.stats.ripe_count = counts['ripe']
        self.stats.rotten_count = counts['rotten']
    
    def update_results(self, detections: List[Detection]):
        """Update results text display"""
        if not detections:
            return
        
        results = f"=== Detection Results ({datetime.now().strftime('%H:%M:%S')}) ===\n"
        results += f"Found {len(detections)} tomato(es)\n\n"
        
        for i, det in enumerate(detections, 1):
            results += f"Tomato #{i}:\n"
            results += f"  Type: {det.label.upper()}\n"
            results += f"  Confidence: {det.confidence:.1%}\n"
            results += f"  Position: ({det.center[0]}, {det.center[1]})\n"
            results += f"  Size: {det.bbox[2]}x{det.bbox[3]} px\n\n"
        
        self.results_text.setPlainText(results)
    
    def toggle_detection(self):
        """Toggle detection display"""
        self.camera_widget.show_detections = not self.camera_widget.show_detections
        if self.camera_widget.show_detections:
            self.detection_btn.setText("🔍 Detection: ON")
        else:
            self.detection_btn.setText("🔍 Detection: OFF")
    
    def update_threshold(self, value):
        """Update confidence threshold"""
        self.threshold_label.setText(f"{value}%")
        self.detector.confidence_threshold = value / 100.0
    
    def update_min_area(self, value):
        """Update minimum detection area"""
        self.detector.min_area = value
    
    def clear_results(self):
        """Clear results display"""
        self.results_text.clear()
        self.stats = HarvestStats()
        self.update_statistics([])
    
    def capture_image(self):
        """Capture current frame with detections"""
        if self.current_frame is not None:
            # Create annotated frame
            annotated = self.current_frame.copy()
            
            # Get current detections
            detections = self.detector.detect_and_classify(self.current_frame)
            
            # Draw detections
            for det in detections:
                x, y, w, h = det.bbox
                color = {'unripe': (0, 255, 0), 'ripe': (0, 0, 255), 
                        'rotten': (139, 69, 19)}.get(det.label, (128, 128, 128))
                
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                label = f"{det.label}: {det.confidence:.1%}"
                cv2.putText(annotated, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_{timestamp}.jpg"
            cv2.imwrite(filename, annotated)
            
            self.statusBar().showMessage(f"Saved: {filename}")
            QMessageBox.information(self, "Success", f"Image saved as {filename}")
    
    def closeEvent(self, event):
        """Clean shutdown"""
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
