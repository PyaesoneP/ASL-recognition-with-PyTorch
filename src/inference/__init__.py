"""
ASL Real-Time Recognition with PyTorch, MediaPipe & Sentence Builder
=====================================================================
ET0732 Machine Learning & Artificial Intelligence Mini Project

Desktop webcam application — NOT for server use.
For the web API, see api/services/predictor.py (ONNX Runtime).

Requirements:
pip install torch torchvision opencv-python mediapipe numpy
"""

import cv2
import numpy as np
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from collections import deque, Counter
import time
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from src.config.settings import (
        BASE_DIR, DEFAULT_MODEL_PATH, PREDICTION_DEFAULTS, CLASS_NAMES, NUM_CLASSES,
    )
except ImportError:
    # Fallback for direct execution without proper package structure
    BASE_DIR = Path(__file__).resolve().parents[2]
    DEFAULT_MODEL_PATH = "outputs/models/best_mobilenet_v2.pth"
    PREDICTION_DEFAULTS = {
        "CONFIDENCE_THRESHOLD": 0.5,
        "STABILITY_FRAMES": 6,
        "COOLDOWN_FRAMES": 18,
        "SMOOTHING_WINDOW": 3,
        "IMG_SIZE": 224,
    }
    CLASS_NAMES = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'del', 'nothing', 'space',
    ]
    NUM_CLASSES = 29

# CustomCNN is shared with the API via src/models.py (single source of truth,
# train/serve parity). ROOT_DIR is already on sys.path (see top of file).
from src.models import CustomCNN

# ============================================================
# CONFIGURATION - UPDATE THESE!
# ============================================================

MODEL_PATH = str(BASE_DIR / DEFAULT_MODEL_PATH)
MODEL_TYPE = "mobilenet_v2"

IMG_SIZE = PREDICTION_DEFAULTS["IMG_SIZE"]

CONFIDENCE_THRESHOLD = PREDICTION_DEFAULTS["CONFIDENCE_THRESHOLD"]
STABILITY_FRAMES = PREDICTION_DEFAULTS["STABILITY_FRAMES"]
COOLDOWN_FRAMES = PREDICTION_DEFAULTS["COOLDOWN_FRAMES"]
SMOOTHING_WINDOW = PREDICTION_DEFAULTS["SMOOTHING_WINDOW"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


inference_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model(model_path, model_type, num_classes=29):
    """Load trained PyTorch model."""
    print(f"Loading {model_type} model from: {model_path}")
    
    if model_type == 'mobilenet_v2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        
    elif model_type == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif model_type == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif model_type == 'custom_cnn':
        model = CustomCNN(num_classes)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    print(f"Model loaded successfully on {DEVICE}!")
    return model


class HandDetector:
    """MediaPipe-based hand detection and tracking."""
    
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
    
    def find_hands(self, frame, draw=True):
        """Detect hands in frame and optionally draw landmarks."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)
        
        hand_boxes = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )
                
                h, w, _ = frame.shape
                x_coords = [lm.x * w for lm in hand_landmarks.landmark]
                y_coords = [lm.y * h for lm in hand_landmarks.landmark]
                
                x_min, x_max = int(min(x_coords)), int(max(x_coords))
                y_min, y_max = int(min(y_coords)), int(max(y_coords))
                
                padding_x = int((x_max - x_min) * 0.25)
                padding_y = int((y_max - y_min) * 0.25)
                
                x_min = max(0, x_min - padding_x)
                y_min = max(0, y_min - padding_y)
                x_max = min(w, x_max + padding_x)
                y_max = min(h, y_max + padding_y)
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                box_size = max(box_w, box_h)
                
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                
                x_min = max(0, center_x - box_size // 2)
                y_min = max(0, center_y - box_size // 2)
                x_max = min(w, x_min + box_size)
                y_max = min(h, y_min + box_size)
                
                hand_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return frame, hand_boxes
    
    def release(self):
        self.hands.close()


class ASLPredictor:
    """ASL alphabet prediction using trained PyTorch model."""
    
    def __init__(self, model):
        self.model = model
        self.prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    
    def preprocess(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = inference_transforms(rgb)
        batch = tensor.unsqueeze(0).to(DEVICE)
        return batch
    
    @torch.no_grad()
    def predict(self, image):
        preprocessed = self.preprocess(image)
        outputs = self.model(preprocessed)
        probabilities = F.softmax(outputs, dim=1)[0]
        
        confidence, pred_idx = torch.max(probabilities, dim=0)
        predicted_class = CLASS_NAMES[pred_idx.item()]
        
        self.prediction_history.append(pred_idx.item())
        
        if len(self.prediction_history) >= 3:
            most_common_idx = Counter(self.prediction_history).most_common(1)[0][0]
            smoothed_class = CLASS_NAMES[most_common_idx]
            smoothed_confidence = probabilities[most_common_idx].item()
        else:
            smoothed_class = predicted_class
            smoothed_confidence = confidence.item()
        
        return smoothed_class, smoothed_confidence, probabilities.cpu().numpy()
    
    def reset_history(self):
        self.prediction_history.clear()


class SentenceBuilder:
    """Builds sentences from predicted ASL letters with stability filtering."""
    
    def __init__(self, stability_frames=STABILITY_FRAMES, cooldown_frames=COOLDOWN_FRAMES):
        self.sentence = ""
        self.current_letter = None
        self.letter_count = 0
        self.stability_threshold = stability_frames
        self.cooldown_threshold = cooldown_frames
        self.cooldown_counter = 0
        self.last_added = None
    
    def update(self, predicted_letter, confidence):
        letter_added = None
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None, self.sentence
        
        if confidence < CONFIDENCE_THRESHOLD or predicted_letter.lower() == 'nothing':
            self.current_letter = None
            self.letter_count = 0
            return None, self.sentence
        
        if predicted_letter == self.current_letter:
            self.letter_count += 1
            
            if self.letter_count >= self.stability_threshold:
                letter_added = self._add_letter(predicted_letter)
                self.letter_count = 0
                self.cooldown_counter = self.cooldown_threshold
        else:
            self.current_letter = predicted_letter
            self.letter_count = 1
        
        return letter_added, self.sentence
    
    def _add_letter(self, letter):
        if letter.lower() == 'space':
            self.sentence += ' '
            self.last_added = '[SPACE]'
            return '[SPACE]'
        elif letter.lower() == 'del':
            if len(self.sentence) > 0:
                self.sentence = self.sentence[:-1]
            self.last_added = '[DEL]'
            return '[DEL]'
        else:
            self.sentence += letter.upper()
            self.last_added = letter.upper()
            return letter.upper()
    
    def add_space(self):
        self.sentence += ' '
        self.last_added = '[SPACE]'
    
    def delete_last(self):
        if len(self.sentence) > 0:
            self.sentence = self.sentence[:-1]
            self.last_added = '[DEL]'
    
    def clear(self):
        self.sentence = ""
        self.current_letter = None
        self.letter_count = 0
        self.last_added = None
    
    def get_progress(self):
        if self.current_letter and self.cooldown_counter == 0:
            return min(1.0, self.letter_count / self.stability_threshold)
        return 0.0


class UIRenderer:
    """Renders the user interface overlays."""
    
    @staticmethod
    def draw_hand_box(frame, box, color=(0, 255, 0), thickness=3):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    @staticmethod
    def draw_prediction(frame, prediction, confidence, x, y):
        text = f"{prediction}: {confidence*100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        
        cv2.rectangle(frame, (x-5, y-text_h-10), (x+text_w+5, y+5), (0, 0, 0), -1)
        
        if confidence >= 0.8:
            color = (0, 255, 0)
        elif confidence >= 0.6:
            color = (0, 255, 255)
        else:
            color = (0, 165, 255)
        
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    @staticmethod
    def draw_sentence_box(frame, sentence, y_position):
        h, w = frame.shape[:2]
        
        box_height = 80
        cv2.rectangle(frame, (10, y_position), (w-10, y_position + box_height), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, y_position), (w-10, y_position + box_height), (100, 100, 100), 2)
        
        cv2.putText(frame, "Sentence:", (20, y_position + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        display_sentence = sentence + "_"
        
        max_chars = 50
        if len(display_sentence) > max_chars:
            display_sentence = "..." + display_sentence[-(max_chars-3):]
        
        cv2.putText(frame, display_sentence, (20, y_position + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    @staticmethod
    def draw_progress_bar(frame, progress, x, y, width=200, height=12):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (50, 50, 50), -1)
        
        if progress > 0:
            progress_width = int(width * progress)
            if progress < 0.5:
                color = (0, 255, 255)
            elif progress < 0.8:
                color = (0, 200, 100)
            else:
                color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + progress_width, y + height), color, -1)
        
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
    
    @staticmethod
    def draw_instructions(frame):
        instructions = "Q:Quit | C:Clear | SPACE:Space | BACKSPACE:Del | S:Screenshot | R:Reset"
        cv2.putText(frame, instructions, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    @staticmethod
    def draw_status(frame, status_text, color=(0, 255, 0)):
        h, w = frame.shape[:2]
        cv2.putText(frame, status_text, (w - 250, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    @staticmethod
    def draw_feedback(frame, letter):
        h, w = frame.shape[:2]
        text = f"+ {letter}"
        cv2.putText(frame, text, (w//2 - 60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    
    @staticmethod
    def draw_top_predictions(frame, probabilities, x, y, top_n=5):
        top_indices = np.argsort(probabilities)[-top_n:][::-1]
        
        bar_width = 120
        bar_height = 16
        spacing = 22
        
        cv2.putText(frame, "Top Predictions:", (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, idx in enumerate(top_indices):
            class_name = CLASS_NAMES[idx]
            conf = probabilities[idx]
            
            bar_y = y + i * spacing
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
            
            fill_width = int(bar_width * conf)
            color = (0, 255, 0) if i == 0 else (100, 150, 100)
            cv2.rectangle(frame, (x, bar_y), (x + fill_width, bar_y + bar_height), color, -1)
            
            label = f"{class_name}: {conf*100:.0f}%"
            cv2.putText(frame, label, (x + bar_width + 10, bar_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    """Main application entry point."""
    
    print("="*60)
    print("ASL Real-Time Recognition (PyTorch)")
    print("="*60)
    print(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\nModel not found: {MODEL_PATH}")
        print("\nPlease update MODEL_PATH to your saved model location.")
        
        pth_files = [f for f in os.listdir('.') if f.endswith('.pth')]
        if pth_files:
            print(f"\nAvailable .pth files:")
            for f in pth_files:
                print(f"  * {f}")
        return
    
    print("\nInitializing components...")
    model = load_model(MODEL_PATH, MODEL_TYPE, NUM_CLASSES)
    hand_detector = HandDetector(max_hands=1, detection_confidence=0.7)
    predictor = ASLPredictor(model)
    sentence_builder = SentenceBuilder()
    ui = UIRenderer()
    
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("\nIf you're using WSL2, run this script on Windows directly!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("\nReady! Show ASL gestures to the camera.")
    print("\nControls:")
    print("  Q         - Quit")
    print("  C         - Clear sentence")
    print("  SPACE     - Add space")
    print("  BACKSPACE - Delete last character")
    print("  S         - Save screenshot")
    print("  R         - Reset prediction history")
    print("-"*60)
    
    feedback_letter = None
    feedback_frames = 0
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read from webcam")
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        frame, hand_boxes = hand_detector.find_hands(frame, draw=True)
        
        current_prediction = None
        current_confidence = 0
        all_probabilities = None
        
        if hand_boxes:
            x, y, bw, bh = hand_boxes[0]
            ui.draw_hand_box(frame, (x, y, bw, bh))
            
            hand_img = frame[y:y+bh, x:x+bw]
            
            if hand_img.size > 0 and hand_img.shape[0] > 10 and hand_img.shape[1] > 10:
                current_prediction, current_confidence, all_probabilities = predictor.predict(hand_img)
                letter_added, _ = sentence_builder.update(current_prediction, current_confidence)
                
                if letter_added:
                    feedback_letter = letter_added
                    feedback_frames = 25
                
                ui.draw_prediction(frame, current_prediction, current_confidence, x, y - 50)
                progress = sentence_builder.get_progress()
                ui.draw_progress_bar(frame, progress, x, y - 18, width=bw, height=10)
                ui.draw_top_predictions(frame, all_probabilities, w - 220, 60)
        else:
            ui.draw_status(frame, "No hand detected", (0, 165, 255))
            predictor.reset_history()
        
        if feedback_frames > 0 and feedback_letter:
            ui.draw_feedback(frame, feedback_letter)
            feedback_frames -= 1
        
        ui.draw_sentence_box(frame, sentence_builder.sentence, h - 100)
        ui.draw_instructions(frame)
        
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            current_fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        cv2.putText(frame, f"FPS: {current_fps}", (10, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(frame, f"Device: {DEVICE}", (w - 150, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        cv2.imshow('ASL Recognition (PyTorch) - Press Q to quit', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            sentence_builder.clear()
            predictor.reset_history()
            print("Sentence cleared")
        elif key == ord(' '):
            sentence_builder.add_space()
            print("Space added")
        elif key == 8:
            sentence_builder.delete_last()
            print("Character deleted")
        elif key == ord('s') or key == ord('S'):
            filename = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('r') or key == ord('R'):
            predictor.reset_history()
            print("Prediction history reset")
    
    cap.release()
    cv2.destroyAllWindows()
    hand_detector.release()
    
    print("\n" + "="*60)
    print("Final Sentence:")
    print(f'   "{sentence_builder.sentence}"')
    print("="*60)


def main_simple():
    """Simple version with manual ROI box (no MediaPipe required)."""
    
    print("="*60)
    print("ASL Recognition - Simple Mode")
    print("="*60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        return
    
    model = load_model(MODEL_PATH, MODEL_TYPE, NUM_CLASSES)
    predictor = ASLPredictor(model)
    sentence_builder = SentenceBuilder()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    roi_size = 300
    roi_x, roi_y = 50, 100
    
    print("\nPlace your hand in the green box!")
    
    feedback_letter = None
    feedback_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_size, roi_y + roi_size), (0, 255, 0), 3)
        cv2.putText(frame, "Place hand here", (roi_x, roi_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        roi = frame[roi_y:roi_y+roi_size, roi_x:roi_x+roi_size]
        pred_class, confidence, probs = predictor.predict(roi)
        
        added, _ = sentence_builder.update(pred_class, confidence)
        if added:
            feedback_letter = added
            feedback_frames = 25
        
        color = (0, 255, 0) if confidence >= 0.7 else (0, 255, 255)
        cv2.putText(frame, f"{pred_class}: {confidence*100:.1f}%", 
                   (roi_x, roi_y + roi_size + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        progress = sentence_builder.get_progress()
        bar_y = roi_y + roi_size + 50
        cv2.rectangle(frame, (roi_x, bar_y), (roi_x + roi_size, bar_y + 10), (50, 50, 50), -1)
        if progress > 0:
            cv2.rectangle(frame, (roi_x, bar_y), (roi_x + int(roi_size * progress), bar_y + 10), (0, 255, 0), -1)
        
        if feedback_frames > 0:
            cv2.putText(frame, f"+ {feedback_letter}", (w//2 - 50, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            feedback_frames -= 1
        
        cv2.rectangle(frame, (10, h - 80), (w - 10, h - 10), (40, 40, 40), -1)
        cv2.putText(frame, "Sentence:", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        display = sentence_builder.sentence + "_"
        if len(display) > 50:
            display = "..." + display[-47:]
        cv2.putText(frame, display, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.putText(frame, "Q:Quit | C:Clear | SPACE:Space | BACKSPACE:Del",
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('ASL Recognition (Simple Mode)', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            sentence_builder.clear()
        elif key == ord(' '):
            sentence_builder.add_space()
        elif key == 8:
            sentence_builder.delete_last()
    
    cap.release()
    cv2.destroyAllWindows()
    print(f'\nFinal sentence: "{sentence_builder.sentence}"')


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--simple':
        main_simple()
    else:
        try:
            main()
        except ImportError as e:
            if 'mediapipe' in str(e):
                print(f"\nMediaPipe not installed: {e}")
                print("Running simple mode instead...")
                main_simple()
            else:
                raise
        except Exception as e:
            print(f"\nError: {e}")
            print("Try running with --simple flag: python asl_pytorch_inference.py --simple")
