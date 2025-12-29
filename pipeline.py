#@title Test
import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import json

# ==================== CONFIGURATION ====================
CONFIG = {
    # Directories
    'comparison_images_dir': '/content/drive/MyDrive/Night2Day/Input/vehicle_comparisons/vehicle_comparisons',
    'labels_dir': '/content/drive/MyDrive/Night2Day/Input/labels_fixed',
    'output_dir': '/content/drive/MyDrive/Night2Day/Output/seatbelt_detection_results',

    # Models
    'person_detection_model': 'yolov8n.pt',
    'seatbelt_model_path': 'yolov8n.pt',

    # Detection parameters
    'person_confidence': 0.30,
    'seatbelt_confidence': 0.20,
    'iou_threshold': 0.45,
    'person_class_id': 0,

    # Label mapping
    'label_classes': {
        0: 'seatbelt',
        1: 'no_seatbelt',
        2: 'person'
    },

    # for evaluation/mAP
    'iou_match_threshold': 0.3,

    # Processing
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_visualization': True,
}
# ======================================================

def parse_yolo_label(label_path, img_width, img_height, x_offset=0):
    """Parse YOLO format label file and adjust coordinates for image splits."""
    annotations = []

    if not os.path.exists(label_path):
        return annotations

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])

                x_center_norm = float(parts[1])
                y_center_norm = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])

                # Convert normalized coords to full image
                full_width = img_width * 2
                x_center_full = x_center_norm * full_width
                y_center_full = y_center_norm * img_height
                box_width_full = width_norm * full_width
                box_height_full = height_norm * img_height

                # Adjust for x offset
                x_center_adjusted = x_center_full - x_offset

                x1 = x_center_adjusted - box_width_full / 2
                x2 = x_center_adjusted + box_width_full / 2
                y1 = y_center_full - box_height_full / 2
                y2 = y_center_full + box_height_full / 2

                # Only include if box intersects this half
                if x1 < img_width and x2 > 0:
                    x1 = max(0, int(x1))
                    y1 = max(0, int(y1))
                    x2 = min(img_width, int(x2))
                    y2 = min(img_height, int(y2))

                    if x2 > x1 and y2 > y1:
                        annotations.append({
                            'class_id': class_id,
                            'box': [x1, y1, x2, y2],
                            'class_name': CONFIG['label_classes'].get(class_id, 'unknown')
                        })

    return annotations

def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

def match_detections_to_ground_truth(detections, ground_truth, iou_threshold=0.3):
    """Match detected boxes to ground truth boxes using IoU."""
    matched_gt = set()
    matched_det = set()
    matches = []

    for det_idx, det in enumerate(detections):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(det['box'], gt['box'])
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            matched_det.add(det_idx)
            matches.append({
                'detection': det,
                'ground_truth': ground_truth[best_gt_idx],
                'iou': best_iou
            })

    return matches, matched_det, matched_gt

class SeatbeltDetectionPipeline:
    """Pipeline for detecting persons and seatbelts, comparing night vs day performance."""

    def __init__(self, config):
        self.config = config
        self.device = config['device']

        # Create output directories
        self.output_dirs = {
            'visualizations': os.path.join(config['output_dir'], 'visualizations'),
            'results': os.path.join(config['output_dir'], 'results'),
        }
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Load models
        self.load_models()

        # Initialize statistics & tracking for mAP
        self.reset_statistics()

    def load_models(self):
        print("="*60)
        print("Loading Models...")
        print("="*60)
        self.person_detector = YOLO(self.config['person_detection_model'])
        print(f"✓ Person detection model loaded: {self.config['person_detection_model']}")
        self.seatbelt_detector = YOLO(self.config['seatbelt_model_path'])
        print(f"✓ Seatbelt detection model loaded: {self.config['seatbelt_model_path']}")
        print("="*60 + "\n")

    def reset_statistics(self):
        """Reset statistics for a new evaluation."""
        self.stats = {
            'night': {
                'person_detection': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
                'seatbelt_detection': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
            },
            'day': {
                'person_detection': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
                'seatbelt_detection': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
            },
            'total_images': 0,
            'images_processed': 0
        }

        # For mAP: store all detections per class, per image type
        self.all_detections = {
            'night': {'person': [], 'seatbelt': [], 'no_seatbelt': []},
            'day':   {'person': [], 'seatbelt': [], 'no_seatbelt': []}
        }
        # For mAP: count of ground truths per class
        self.all_ground_truths = {
            'night': {'person': 0, 'seatbelt': 0, 'no_seatbelt': 0},
            'day':   {'person': 0, 'seatbelt': 0, 'no_seatbelt': 0}
        }
        # For mAP: store ground truth boxes per class
        self.gt_boxes = {
            'night': {'person': [], 'seatbelt': [], 'no_seatbelt': []},
            'day':   {'person': [], 'seatbelt': [], 'no_seatbelt': []}
        }

        self.detailed_results = []

    def detect_persons(self, image, image_type='night'):
        """Detect persons in the image."""
        img_array = np.array(image)
        img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = self.person_detector(
            img_cv,
            conf=self.config['person_confidence'],
            iou=self.config['iou_threshold'],
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == self.config['person_class_id']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(box.conf[0]),
                    'class_id': cls,
                    'class_name': 'person'
                })

        return detections

    def detect_seatbelts_in_crop(self, image, person_box):
        """Detect seatbelt/no‑seatbelt in a cropped person region."""
        x1, y1, x2, y2 = person_box
        padding = 10
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)

        crop = image.crop((x1, y1, x2, y2))
        crop_array = np.array(crop)
        crop_cv = cv2.cvtColor(crop_array, cv2.COLOR_RGB2BGR)

        results = self.seatbelt_detector(
            crop_cv,
            conf=self.config['seatbelt_confidence'],
            iou=self.config['iou_threshold'],
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in [0, 1]:  # 0=seatbelt, 1=no_seatbelt
                bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                full_x1 = int(bx1 + x1)
                full_y1 = int(by1 + y1)
                full_x2 = int(bx2 + x1)
                full_y2 = int(by2 + y1)
                detections.append({
                    'box': [full_x1, full_y1, full_x2, full_y2],
                    'confidence': float(box.conf[0]),
                    'class_id': cls,
                    'class_name': self.config['label_classes'][cls]
                })

        return detections

    def evaluate_person_detection(self, detections, ground_truth, image_type):
        """Evaluate person detection performance (TP/FP/FN)."""
        gt_persons = [gt for gt in ground_truth if gt['class_name'] == 'person']
        matches, matched_det, matched_gt = match_detections_to_ground_truth(
            detections, gt_persons, self.config['iou_match_threshold']
        )
        TP = len(matches)
        FP = len(detections) - len(matched_det)
        FN = len(gt_persons) - len(matched_gt)

        self.stats[image_type]['person_detection']['TP'] += TP
        self.stats[image_type]['person_detection']['FP'] += FP
        self.stats[image_type]['person_detection']['FN'] += FN

        return {'TP': TP, 'FP': FP, 'FN': FN, 'matches': matches}

    def evaluate_seatbelt_detection(self, seatbelt_detections, ground_truth, person_detections, image_type):
        """evaluate only for detected persons"""
        gt_seatbelts = [gt for gt in ground_truth if gt['class_name'] in ['seatbelt', 'no_seatbelt']]

        filtered_seatbelt_detections = []
        person_has_seatbelt = {}

        for person_idx, person_det in enumerate(person_detections):
            person_has_seatbelt[person_idx] = []
            for sb_det in seatbelt_detections:
                sb_box = sb_det['box']
                sb_center_x = (sb_box[0] + sb_box[2]) / 2
                sb_center_y = (sb_box[1] + sb_box[3]) / 2

                p_box = person_det['box']
                if (p_box[0] <= sb_center_x <= p_box[2] and
                    p_box[1] <= sb_center_y <= p_box[3]):
                    filtered_seatbelt_detections.append(sb_det)
                    person_has_seatbelt[person_idx].append(sb_det)
                    break

        implicit_no_seatbelt = []
        for person_idx, person_det in enumerate(person_detections):
            if len(person_has_seatbelt[person_idx]) == 0:
                p_box = person_det['box']
                implicit_detection = {
                    'box': p_box,
                    'confidence': person_det['confidence'],
                    'class_id': 1,
                    'class_name': 'no_seatbelt',
                    'implicit': True
                }
                filtered_seatbelt_detections.append(implicit_detection)
                implicit_no_seatbelt.append(implicit_detection)

        TP = 0
        FP = 0
        FN = 0

        matches, matched_det, matched_gt = match_detections_to_ground_truth(
            filtered_seatbelt_detections, gt_seatbelts, self.config['iou_match_threshold']
        )

        for match in matches:
            det_class = match['detection']['class_name']
            gt_class = match['ground_truth']['class_name']
            if det_class == gt_class:
                TP += 1
            else:
                FP += 1
                FN += 1

        # Unmatched detections are false positives
        FP += len(filtered_seatbelt_detections) - len(matched_det)
        # Unmatched ground truths are false negatives
        FN += len(gt_seatbelts) - len(matched_gt)

        self.stats[image_type]['seatbelt_detection']['TP'] += TP
        self.stats[image_type]['seatbelt_detection']['FP'] += FP
        self.stats[image_type]['seatbelt_detection']['FN'] += FN

        return {
            'TP': TP,
            'FP': FP,
            'FN': FN,
            'matches': matches,
            'filtered_detections': filtered_seatbelt_detections,
            'implicit_no_seatbelt': implicit_no_seatbelt
        }

    def visualize_detections(self, night_img, day_img, night_results, day_results, night_gt, day_gt, img_name):
        """Create 2x2 grid: top = detections, bottom = GT, plus legend & stats."""
        # Load font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        except Exception:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()

        # ========== Draw night detections ==========
        night_det_img = night_img.copy()
        draw_night_det = ImageDraw.Draw(night_det_img)
        for det in night_results['person_detections']:
            box = det['box']
            draw_night_det.rectangle(box, outline='yellow', width=3)
            draw_night_det.text((box[0], box[1] - 30), f"Person {det['confidence']:.2f}", fill='yellow', font=font_small)
        for det in night_results['seatbelt_detections']:
            box = det['box']
            color = 'lime' if det['class_name'] == 'seatbelt' else 'red'
            draw_night_det.rectangle(box, outline=color, width=2)
            label = det['class_name'] + (" (implicit)" if det.get('implicit', False) else "")
            draw_night_det.text((box[0], box[1] - 15), label, fill=color, font=font_small)

        # ========== Draw day detections ==========
        day_det_img = day_img.copy()
        draw_day_det = ImageDraw.Draw(day_det_img)
        for det in day_results['person_detections']:
            box = det['box']
            draw_day_det.rectangle(box, outline='yellow', width=3)
            draw_day_det.text((box[0], box[1] - 30), f"Person {det['confidence']:.2f}", fill='yellow', font=font_small)
        for det in day_results['seatbelt_detections']:
            box = det['box']
            color = 'lime' if det['class_name'] == 'seatbelt' else 'red'
            draw_day_det.rectangle(box, outline=color, width=2)
            label = det['class_name'] + (" (implicit)" if det.get('implicit', False) else "")
            draw_day_det.text((box[0], box[1] - 15), label, fill=color, font=font_small)

        # ========== Draw ground truth (night) ==========
        night_gt_img = night_img.copy()
        draw_night_gt = ImageDraw.Draw(night_gt_img)
        for gt in night_gt:
            box = gt['box']
            if gt['class_name'] == 'person':
                draw_night_gt.rectangle(box, outline='blue', width=3)
                draw_night_gt.text((box[0], box[1] - 30), "GT: Person", fill='blue', font=font_small)
            elif gt['class_name'] == 'seatbelt':
                draw_night_gt.rectangle(box, outline='cyan', width=2)
                draw_night_gt.text((box[0], box[1] - 15), "GT: Seatbelt", fill='cyan', font=font_small)
            elif gt['class_name'] == 'no_seatbelt':
                draw_night_gt.rectangle(box, outline='orange', width=2)
                draw_night_gt.text((box[0], box[1] - 15), "GT: No‑Seatbelt", fill='orange', font=font_small)

        # ========== Draw ground truth (day) ==========
        day_gt_img = day_img.copy()
        draw_day_gt = ImageDraw.Draw(day_gt_img)
        for gt in day_gt:
            box = gt['box']
            if gt['class_name'] == 'person':
                draw_day_gt.rectangle(box, outline='blue', width=3)
                draw_day_gt.text((box[0], box[1] - 30), "GT: Person", fill='blue', font=font_small)
            elif gt['class_name'] == 'seatbelt':
                draw_day_gt.rectangle(box, outline='cyan', width=2)
                draw_day_gt.text((box[0], box[1] - 15), "GT: Seatbelt", fill='cyan', font=font_small)
            elif gt['class_name'] == 'no_seatbelt':
                draw_day_gt.rectangle(box, outline='orange', width=2)
                draw_day_gt.text((box[0], box[1] - 15), "GT: No‑Seatbelt", fill='orange', font=font_small)

        # ========== Create canvas ==========
        img_width, img_height = night_img.size
        padding = 20
        label_height = 40
        legend_height = 140
        total_width = img_width * 2 + padding * 3
        total_height = (img_height + label_height) * 2 + padding * 3 + legend_height

        canvas = Image.new('RGB', (total_width, total_height), color='white')
        draw_canvas = ImageDraw.Draw(canvas)
        y_top = padding + label_height

        # Paste detections
        canvas.paste(night_det_img, (padding, y_top))
        draw_canvas.text((padding + img_width // 2 - 80, padding + 5), "NIGHT - Model Detections", fill='black', font=font)
        canvas.paste(day_det_img, (padding * 2 + img_width, y_top))
        draw_canvas.text((padding * 2 + img_width + img_width // 2 - 80, padding + 5), "DAY - Model Detections", fill='black', font=font)

        # Paste ground truth
        y_bottom = y_top + img_height + padding + label_height
        canvas.paste(night_gt_img, (padding, y_bottom))
        draw_canvas.text((padding + img_width // 2 - 80, y_bottom - label_height + 5), "NIGHT - Ground Truth", fill='black', font=font)
        canvas.paste(day_gt_img, (padding * 2 + img_width, y_bottom))
        draw_canvas.text((padding * 2 + img_width + img_width // 2 - 80, y_bottom - label_height + 5), "DAY - Ground Truth", fill='black', font=font)

        # Stats text
        stats_y = y_bottom + img_height + padding
        night_persons_det = len(night_results['person_detections'])
        night_seatbelts_det = len([d for d in night_results['seatbelt_detections'] if d['class_name'] == 'seatbelt'])
        night_no_seatbelts_det = len([d for d in night_results['seatbelt_detections'] if d['class_name'] == 'no_seatbelt'])
        night_persons_gt = len([g for g in night_gt if g['class_name'] == 'person'])
        night_seatbelts_gt = len([g for g in night_gt if g['class_name'] == 'seatbelt'])
        day_persons_det = len(day_results['person_detections'])
        day_seatbelts_det = len([d for d in day_results['seatbelt_detections'] if d['class_name'] == 'seatbelt'])
        day_no_seatbelts_det = len([d for d in day_results['seatbelt_detections'] if d['class_name'] == 'no_seatbelt'])
        day_persons_gt = len([g for g in day_gt if g['class_name'] == 'person'])
        day_seatbelts_gt = len([g for g in day_gt if g['class_name'] == 'seatbelt'])

        night_stats = (f"Night: Detected {night_persons_det}/{night_persons_gt} persons, "
                       f"{night_seatbelts_det}/{night_seatbelts_gt} seatbelts, {night_no_seatbelts_det} no‑seatbelt")
        draw_canvas.text((padding, stats_y), night_stats, fill='blue', font=font_small)
        day_stats = (f"Day:   Detected {day_persons_det}/{day_persons_gt} persons, "
                     f"{day_seatbelts_det}/{day_seatbelts_gt} seatbelts, {day_no_seatbelts_det} no‑seatbelt")
        draw_canvas.text((padding, stats_y + 20), day_stats, fill='blue', font=font_small)

        # Legend
        legend_y = stats_y + 50
        draw_canvas.text((padding, legend_y), "LEGEND:", fill='black', font=font)
        draw_canvas.text((padding, legend_y + 22), "Model Detections:", fill='black', font=font_small)

        # Person detection legend
        draw_canvas.rectangle([padding + 10, legend_y + 38, padding + 25, legend_y + 48],
                              outline='yellow', width=3)
        draw_canvas.text((padding + 30, legend_y + 36), "Detected Person", fill='black', font=font_small)
        # Seatbelt detection legend
        draw_canvas.rectangle([padding + 10, legend_y + 53, padding + 25, legend_y + 63],
                              outline='lime', width=2)
        draw_canvas.text((padding + 30, legend_y + 51), "Detected Seatbelt", fill='black', font=font_small)
        # No-seatbelt legend
        draw_canvas.rectangle([padding + 10, legend_y + 68, padding + 25, legend_y + 78],
                              outline='red', width=2)
        draw_canvas.text((padding + 30, legend_y + 66), "Detected No‑Seatbelt", fill='black', font=font_small)

        # Ground truth legend
        legend_x2 = padding + img_width
        draw_canvas.text((legend_x2, legend_y + 22), "Ground Truth:", fill='black', font=font_small)

        draw_canvas.rectangle([legend_x2 + 10, legend_y + 38, legend_x2 + 25, legend_y + 48],
                              outline='blue', width=3)
        draw_canvas.text((legend_x2 + 30, legend_y + 36), "GT Person", fill='black', font=font_small)
        draw_canvas.rectangle([legend_x2 + 10, legend_x2 + 53, legend_x2 + 25, legend_y + 63],
                              outline='cyan', width=2)
        draw_canvas.text((legend_x2 + 30, legend_y + 51), "GT Seatbelt", fill='black', font=font_small)
        draw_canvas.rectangle([legend_x2 + 10, legend_y + 68, legend_x2 + 25, legend_y + 78],
                              outline='orange', width=2)
        draw_canvas.text((legend_x2 + 30, legend_y + 66), "GT No‑Seatbelt", fill='black', font=font_small)

        save_path = os.path.join(self.output_dirs['visualizations'], f"detection_{img_name}")
        canvas.save(save_path)
        return save_path

    def process_image(self, comp_img_path, label_path):
        """Process a single comparison image (night/day) along with ground truth."""
        img_name = os.path.basename(comp_img_path)
        base_name = img_name.replace('comparison_', '').replace('.png', '').replace('.jpg', '')

        comp_img = Image.open(comp_img_path).convert('RGB')
        night_img, day_img, mid_x = self.split_comparison_image(comp_img)

        # Parse ground truth
        night_gt = parse_yolo_label(label_path, night_img.width, night_img.height, x_offset=0)
        day_gt   = parse_yolo_label(label_path, day_img.width,  day_img.height, x_offset=mid_x)

        if len(night_gt) == 0 and len(day_gt) == 0:
            print(f"  ⚠ No labels found for {img_name}")
            return None

        print(f"  Ground Truth: Night={len(night_gt)} annotations, Day={len(day_gt)} annotations")

        # Record GT boxes and GT count for mAP
        for cls in ['person', 'seatbelt', 'no_seatbelt']:
            # Night
            cls_gt_n = [gt for gt in night_gt if gt['class_name'] == cls]
            self.all_ground_truths['night'][cls] += len(cls_gt_n)
            self.gt_boxes['night'][cls].extend([gt['box'] for gt in cls_gt_n])
            # Day
            cls_gt_d = [gt for gt in day_gt if gt['class_name'] == cls]
            self.all_ground_truths['day'][cls] += len(cls_gt_d)
            self.gt_boxes['day'][cls].extend([gt['box'] for gt in cls_gt_d])

        # NIGHT PROCESSING
        night_person_det = self.detect_persons(night_img, 'night')
        night_person_eval = self.evaluate_person_detection(night_person_det, night_gt, 'night')

        night_seatbelt_det = []
        for person_det in night_person_det:
            sb = self.detect_seatbelts_in_crop(night_img, person_det['box'])
            night_seatbelt_det.extend(sb)
        night_seatbelt_eval = self.evaluate_seatbelt_detection(night_seatbelt_det, night_gt, night_person_det, 'night')

        # Record detected boxes with confidence for mAP (night)
        for det in night_person_det:
            self.all_detections['night']['person'].append({
                'confidence': det['confidence'],
                'box': det['box'],
                'class_name': 'person'
            })
        for det in night_seatbelt_eval['filtered_detections']:
            cls_name = det['class_name']
            self.all_detections['night'][cls_name].append({
                'confidence': det['confidence'],
                'box': det['box'],
                'class_name': cls_name,
                'implicit': det.get('implicit', False)
            })

        # DAY PROCESSING
        day_person_det = self.detect_persons(day_img, 'day')
        day_person_eval = self.evaluate_person_detection(day_person_det, day_gt, 'day')

        day_seatbelt_det = []
        for person_det in day_person_det:
            sb = self.detect_seatbelts_in_crop(day_img, person_det['box'])
            day_seatbelt_det.extend(sb)
        day_seatbelt_eval = self.evaluate_seatbelt_detection(day_seatbelt_det, day_gt, day_person_det, 'day')

        # Record detected boxes for mAP (day)
        for det in day_person_det:
            self.all_detections['day']['person'].append({
                'confidence': det['confidence'],
                'box': det['box'],
                'class_name': 'person'
            })
        for det in day_seatbelt_eval['filtered_detections']:
            cls_name = det['class_name']
            self.all_detections['day'][cls_name].append({
                'confidence': det['confidence'],
                'box': det['box'],
                'class_name': cls_name,
                'implicit': det.get('implicit', False)
            })

        # Save results
        result = {
            'image_name': img_name,
            'night': {
                'person_detections': night_person_det,
                'person_eval': night_person_eval,
                'seatbelt_detections': night_seatbelt_eval['filtered_detections'],
                'seatbelt_eval': night_seatbelt_eval
            },
            'day': {
                'person_detections': day_person_det,
                'person_eval': day_person_eval,
                'seatbelt_detections': day_seatbelt_eval['filtered_detections'],
                'seatbelt_eval': day_seatbelt_eval
            }
        }
        self.detailed_results.append(result)

        # Visualization
        if self.config['save_visualization']:
            night_results = {
                'person_detections': night_person_det,
                'seatbelt_detections': night_seatbelt_eval['filtered_detections']
            }
            day_results = {
                'person_detections': day_person_det,
                'seatbelt_detections': day_seatbelt_eval['filtered_detections']
            }
            self.visualize_detections(night_img, day_img, night_results, day_results, night_gt, day_gt, img_name)

        return result

    def split_comparison_image(self, img):
        """Split a comparison image (left = night, right = day)."""
        width, height = img.size
        mid_x = width // 2
        night_img = img.crop((0, 0, mid_x, height))
        day_img   = img.crop((mid_x, 0, width, height))
        return night_img, day_img, mid_x

    def calculate_metrics(self, stats_dict):
        """Calculate precision, recall, and F1 from TP, FP, FN."""
        TP = stats_dict['TP']
        FP = stats_dict['FP']
        FN = stats_dict['FN']
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        return {'precision': precision, 'recall': recall, 'f1_score': f1, 'TP': TP, 'FP': FP, 'FN': FN}

    def compute_average_precision(self, detections, gt_boxes, num_gt, iou_threshold):
        """
        Compute AP
        """
        if num_gt == 0:
            # No ground truths?
            return None

        # Sort detections by descending confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        matched_gt = set()

        for idx, det in enumerate(detections):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt:
                    continue
                iou = calculate_iou(det['box'], gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp[idx] = 1
                matched_gt.add(best_gt_idx)
            else:
                fp[idx] = 1

        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        precision = cum_tp / (cum_tp + cum_fp + 1e-16)
        recall = cum_tp / float(num_gt)

        # Compute AP by numerical integration (area under precision-recall curve)
        ap = 0.0
        for i in range(len(detections)):
            if i == 0:
                delta_recall = recall[i]
            else:
                delta_recall = recall[i] - recall[i - 1]
            ap += precision[i] * delta_recall

        return ap

    def compute_map(self):
        """Compute AP per class and mAP (mean of classes) for both night and day."""
        reports_map = {}
        for image_type in ['night', 'day']:
            reports_map[image_type] = {}
            class_list = ['person', 'seatbelt', 'no_seatbelt']
            ap_list = []
            for cls in class_list:
                dets = self.all_detections[image_type][cls]
                gt_boxes_cls = self.gt_boxes[image_type][cls]
                num_gt = self.all_ground_truths[image_type][cls]
                ap = self.compute_average_precision(dets, gt_boxes_cls, num_gt, self.config['iou_match_threshold'])
                reports_map[image_type][cls] = ap
                if ap is not None:
                    ap_list.append(ap)
            # mean AP over classes (excluding classes with None)
            if ap_list:
                reports_map[image_type]['mAP'] = sum(ap_list) / len(ap_list)
            else:
                reports_map[image_type]['mAP'] = None
        return reports_map

    def generate_reports(self):
        """Generate evaluation reports (precision/recall/F1) + mAP."""
        print("\n" + "="*60)
        print("Generating Reports...")
        print("="*60)

        # Compute detection metrics (precision, recall, F1)
        reports = {}
        for image_type in ['night', 'day']:
            reports[image_type] = {
                'person_detection': self.calculate_metrics(self.stats[image_type]['person_detection']),
                'seatbelt_detection': self.calculate_metrics(self.stats[image_type]['seatbelt_detection'])
            }

        # Build DataFrames
        person_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'True Positives', 'False Positives', 'False Negatives'],
            'Night (Original)': [
                f"{reports['night']['person_detection']['precision']:.3f}",
                f"{reports['night']['person_detection']['recall']:.3f}",
                f"{reports['night']['person_detection']['f1_score']:.3f}",
                reports['night']['person_detection']['TP'],
                reports['night']['person_detection']['FP'],
                reports['night']['person_detection']['FN']
            ],
            'Day (Generated)': [
                f"{reports['day']['person_detection']['precision']:.3f}",
                f"{reports['day']['person_detection']['recall']:.3f}",
                f"{reports['day']['person_detection']['f1_score']:.3f}",
                reports['day']['person_detection']['TP'],
                reports['day']['person_detection']['FP'],
                reports['day']['person_detection']['FN']
            ]
        })

        seatbelt_df = pd.DataFrame({
            'Metric': ['Precision', 'Recall', 'F1-Score', 'True Positives', 'False Positives', 'False Negatives'],
            'Night (Original)': [
                f"{reports['night']['seatbelt_detection']['precision']:.3f}",
                f"{reports['night']['seatbelt_detection']['recall']:.3f}",
                f"{reports['night']['seatbelt_detection']['f1_score']:.3f}",
                reports['night']['seatbelt_detection']['TP'],
                reports['night']['seatbelt_detection']['FP'],
                reports['night']['seatbelt_detection']['FN']
            ],
            'Day (Generated)': [
                f"{reports['day']['seatbelt_detection']['precision']:.3f}",
                f"{reports['day']['seatbelt_detection']['recall']:.3f}",
                f"{reports['day']['seatbelt_detection']['f1_score']:.3f}",
                reports['day']['seatbelt_detection']['TP'],
                reports['day']['seatbelt_detection']['FP'],
                reports['day']['seatbelt_detection']['FN']
            ]
        })

        # Save CSVs
        person_csv = os.path.join(self.output_dirs['results'], 'person_detection_comparison.csv')
        seatbelt_csv = os.path.join(self.output_dirs['results'], 'seatbelt_detection_comparison.csv')
        person_df.to_csv(person_csv, index=False)
        seatbelt_df.to_csv(seatbelt_csv, index=False)

        # Save detailed results JSON
        detailed_json = os.path.join(self.output_dirs['results'], 'detailed_results.json')
        with open(detailed_json, 'w') as f:
            json.dump(self.detailed_results, f, indent=2, default=str)

        # Print detection metrics
        print("\n" + "="*60)
        print("PERSON DETECTION RESULTS")
        print("="*60)
        print(person_df.to_string(index=False))

        print("\n" + "="*60)
        print("SEATBELT DETECTION RESULTS")
        print("="*60)
        print(seatbelt_df.to_string(index=False))

        # Compute and print mAP
        map_reports = self.compute_map()
        print("\n" + "="*60)
        print("mAP RESULTS")
        print("="*60)
        for image_type in ['night', 'day']:
            print(f"{image_type.upper()}:")
            for cls in ['person', 'seatbelt', 'no_seatbelt']:
                ap_val = map_reports[image_type].get(cls)
                if ap_val is None:
                    print(f"  AP_{cls}: None (no GT)")
                else:
                    print(f"  AP_{cls}: {ap_val:.3f}")
            mAP_val = map_reports[image_type].get('mAP')
            if mAP_val is None:
                print("  mAP: None")
            else:
                print(f"  mAP: {mAP_val:.3f}")

        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        person_f1_diff = reports['day']['person_detection']['f1_score'] - reports['night']['person_detection']['f1_score']
        seatbelt_f1_diff = reports['day']['seatbelt_detection']['f1_score'] - reports['night']['seatbelt_detection']['f1_score']

        print(f"\nPerson Detection F1-Score Change: {person_f1_diff:+.3f}")
        if person_f1_diff > 0:
            print("  ✓ Night‑to‑Day conversion IMPROVED person detection")
        elif person_f1_diff < 0:
            print("  ✗ Night‑to‑Day conversion DEGRADED person detection")
        else:
            print("  = No change in person detection F1")

        print(f"\nSeatbelt Detection F1-Score Change: {seatbelt_f1_diff:+.3f}")
        if seatbelt_f1_diff > 0:
            print("  ✓ Night‑to‑Day conversion IMPROVED seatbelt detection")
        elif seatbelt_f1_diff < 0:
            print("  ✗ Night‑to‑Day conversion DEGRADED seatbelt detection")
        else:
            print("  = No change in seatbelt detection F1")

        print("\n" + "="*60)
        print(f"Reports saved to: {self.output_dirs['results']}")
        print("="*60 + "\n")

        return {'detection_metrics': reports, 'map_metrics': map_reports}

    def run(self):
        """Run the full pipeline on all comparison images."""
        print("\n" + "="*60)
        print("Seatbelt Detection Comparison Pipeline")
        print("="*60)
        print(f"Comparison Images Dir: {self.config['comparison_images_dir']}")
        print(f"Labels Directory: {self.config['labels_dir']}")
        print(f"Output Directory: {self.config['output_dir']}")
        print(f"Device: {self.device}")
        print("="*60 + "\n")

        image_files = [
            f for f in os.listdir(self.config['comparison_images_dir'])
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        if not image_files:
            print("No comparison images found, exiting.")
            return None

        print(f"Found {len(image_files)} comparison images.\n")
        self.stats['total_images'] = len(image_files)

        for img_file in tqdm(image_files, desc="Processing Images"):
            img_path = os.path.join(self.config['comparison_images_dir'], img_file)
            base_name = img_file.replace('comparison_', '').replace('.png', '').replace('.jpg', '')
            label_file = base_name + '.txt'
            label_path = os.path.join(self.config['labels_dir'], label_file)

            if not os.path.exists(label_path):
                print(f"⚠ Label file missing for {img_file}, skipping.")
                continue

            try:
                print(f"\nProcessing {img_file}")
                result = self.process_image(img_path, label_path)
                if result:
                    self.stats['images_processed'] += 1

                    # print per-image summary
                    night_persons = len(result['night']['person_detections'])
                    day_persons = len(result['day']['person_detections'])
                    night_seatbelts = len([d for d in result['night']['seatbelt_detections'] if d['class_name']=='seatbelt'])
                    day_seatbelts = len([d for d in result['day']['seatbelt_detections'] if d['class_name']=='seatbelt'])
                    night_no_seat = len([d for d in result['night']['seatbelt_detections'] if d['class_name']=='no_seatbelt'])
                    day_no_seat = len([d for d in result['day']['seatbelt_detections'] if d['class_name']=='no_seatbelt'])

                    print(f"  Night: {night_persons} person(s), {night_seatbelts} seatbelt(s), {night_no_seat} no‑seatbelt(s)")
                    print(f"  Day:   {day_persons} person(s), {day_seatbelts} seatbelt(s), {day_no_seat} no‑seatbelt(s)")

            except Exception as e:
                print(f"✗ Error on {img_file}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print("\n" + "="*60)
        print(f"Processed {self.stats['images_processed']} / {self.stats['total_images']} images.")
        print("="*60)

        if self.stats['images_processed'] > 0:
            return self.generate_reports()
        else:
            print("No images were successfully processed; skipping report generation.")
            return None

if __name__ == "__main__":
    pipeline = SeatbeltDetectionPipeline(CONFIG)
    pipeline.run()
