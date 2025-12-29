#@title Model Usage to Cropping

import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# Metrics imports
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # LPIPS :contentReference[oaicite:2]{index=2}
from cleanfid import fid

# ==================== CONFIGURATION ====================
CONFIG = {
    # Directories
    'input_dir': '/content/drive/MyDrive/Night2Day/Data/CloseVehicle/Night/Test',
    'output_dir': '/content/drive/MyDrive/Night2Day/Output/close_inference_results',
    'model_path': '/content/drive/MyDrive/Night2Day/Models/checkpoints/checkpoint_epoch_50.pth',

    # Real-day image directory for FID computation
    'real_day_dir': '/content/drive/MyDrive/Night2Day/Data/CloseVehicle/Day',

    # Model parameters
    'yolo_model_path': 'yolov8n.pt',
    'img_size': 512,
    'num_residual_blocks': 9,
    'ngf': 64,
    'ndf': 64,

    # Discriminator threshold
    'discriminator_strictness': 0.3,

    # YOLO detection parameters
    'yolo_confidence': 0.35,
    'yolo_min_area': 1500,
    'yolo_iou_threshold': 0.45,
    'vehicle_classes': [2, 3, 5, 7],

    # Output settings
    'save_failed_discriminator': True,
    'save_detection_debug': True,
    'save_full_comparison': True,
    'comparison_padding': 20,

    # Processing
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'batch_processing': False,
}
# ======================================================

# ==================== MODEL DEFINITIONS ====================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, input_channels=3, num_residual_blocks=9, ngf=64):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, input_channels, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_channels=3, ndf=64):
        super().__init__()

        def disc_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(input_channels, ndf, normalize=False),
            *disc_block(ndf, ndf * 2),
            *disc_block(ndf * 2, ndf * 4),
            *disc_block(ndf * 4, ndf * 8),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(ndf * 8, 1, 4, padding=1),
        )

    def forward(self, x):
        return self.model(x)
# ==========================================================

class Night2DayInference:
    def __init__(self, config):
        self.config = config
        self.device = config['device']

        # Output directories
        self.output_dirs = {
            'converted': os.path.join(config['output_dir'], 'converted_images'),
            'failed_disc': os.path.join(config['output_dir'], 'failed_discriminator'),
            'detections': os.path.join(config['output_dir'], 'detection_debug'),
            'crops': os.path.join(config['output_dir'], 'vehicle_crops'),
            'comparisons': os.path.join(config['output_dir'], 'vehicle_comparisons'),
            'full_comparisons': os.path.join(config['output_dir'], 'full_image_comparisons'),
        }
        for d in self.output_dirs.values():
            os.makedirs(d, exist_ok=True)

        # Load models
        self.load_models()

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((config['img_size'], config['img_size']), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        # Metrics
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', reduction='mean', normalize=False
        ).to(self.device)

        # For FID
        self.generated_for_fid = []

        # CSV for storing per-image metrics
        self.csv_path = os.path.join(config['output_dir'], 'eval_metrics.csv')
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_name', 'discriminator_score', 'lpips', 'sift'])

        # Stats
        self.stats = {
            'total_images': 0,
            'passed_discriminator': 0,
            'failed_discriminator': 0,
            'total_vehicles_detected': 0,
            'total_comparisons_created': 0,
            'full_comparisons_created': 0,
        }

    def load_models(self):
        print("Loading models...")
        self.generator = Generator(
            num_residual_blocks=self.config['num_residual_blocks'],
            ngf=self.config['ngf']
        ).to(self.device)
        self.discriminator = Discriminator(ndf=self.config['ndf']).to(self.device)

        checkpoint = torch.load(self.config['model_path'], map_location=self.device)
        self.generator.load_state_dict(checkpoint['G_night2day'])
        self.discriminator.load_state_dict(checkpoint['D_day'])
        self.generator.eval()
        self.discriminator.eval()

        self.yolo_model = YOLO(self.config['yolo_model_path'])
        print("Models loaded.")

    def check_discriminator(self, img_tensor):
        with torch.no_grad():
            d_out = self.discriminator(img_tensor)
            score = torch.sigmoid(d_out).mean().item()
        passed = score >= self.config['discriminator_strictness']
        return passed, score

    def generate_day_image(self, night_img_path):
        img = Image.open(night_img_path).convert('RGB')
        original_size = img.size
        t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            gen = self.generator(t)
        return gen, t, original_size

    def tensor_to_image(self, tensor, target_size=None):
        arr = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
        arr = (arr * 0.5 + 0.5).clip(0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))
        if target_size:
            img = img.resize(target_size, Image.BICUBIC)
        return img

    def compute_sift_score(self, img1, img2):
        # img1, img2: PIL images
        g1 = np.array(img1.convert('L'))
        g2 = np.array(img2.convert('L'))
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(g1, None)
        kp2, des2 = sift.detectAndCompute(g2, None)
        if des1 is None or des2 is None:
            return 0.0
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        if len(kp1) == 0 or len(kp2) == 0:
            return 0.0
        score = len(good) / max(len(kp1), len(kp2))
        return float(score)

    def detect_vehicles(self, image):
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        results = self.yolo_model(
            img_cv,
            conf=self.config['yolo_confidence'],
            iou=self.config['yolo_iou_threshold'],
            verbose=False
        )[0]
        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls in self.config['vehicle_classes']:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)
                if area >= self.config['yolo_min_area']:
                    detections.append({
                        'box': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(box.conf[0]),
                        'class': cls,
                        'area': area
                    })
        return detections

    def draw_detections(self, image, detections):
        img2 = image.copy()
        draw = ImageDraw.Draw(img2)
        class_names = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
        for det in detections:
            x1, y1, x2, y2 = det['box']
            cls = class_names.get(det['class'], 'Vehicle')
            draw.rectangle([x1, y1, x2, y2], outline='lime', width=3)
            draw.text((x1, y1 - 10), f"{cls} {det['confidence']:.2f}", fill='lime')
        return img2

    def create_full_image_comparison(self, night_img, day_img, img_name, disc_score):
        width, height = night_img.size
        pad = self.config['comparison_padding']
        comp_w = width * 2 + pad * 3
        comp_h = height + pad * 2 + 60
        comp = Image.new('RGB', (comp_w, comp_h), color='white')
        comp.paste(night_img, (pad, pad + 40))
        comp.paste(day_img, (width + pad * 2, pad + 40))
        draw = ImageDraw.Draw(comp)
        draw.text((pad + width // 2 - 70, 5), "NIGHT (Original)", fill='black')
        draw.text((width + pad * 2 + width // 2 - 70, 5), "DAY (Generated)", fill='black')
        draw.text((comp_w // 2 - 100, comp_h - 20), f"Discriminator Score: {disc_score:.4f}", fill='blue')
        filename = f"comparison_{img_name}"
        save_path = os.path.join(self.output_dirs['full_comparisons'], filename)
        comp.save(save_path)
        self.stats['full_comparisons_created'] += 1
        return save_path

    def create_comparison(self, night_crop, day_crop, detection_info, img_name, idx):
        max_h = max(night_crop.height, day_crop.height)
        max_w = max(night_crop.width, day_crop.width)
        night_resized = night_crop.resize((max_w, max_h), Image.BICUBIC)
        day_resized = day_crop.resize((max_w, max_h), Image.BICUBIC)
        pad = self.config['comparison_padding']
        comp_w = max_w * 2 + pad * 3
        comp_h = max_h + pad * 2 + 40
        comp = Image.new('RGB', (comp_w, comp_h), 'white')
        comp.paste(night_resized, (pad, pad + 30))
        comp.paste(day_resized, (max_w + pad * 2, pad + 30))
        draw = ImageDraw.Draw(comp)
        draw.text((pad + max_w // 2 - 50, 5), "NIGHT (Original)", fill='black')
        draw.text((max_w + pad * 2 + max_w // 2 - 50, 5), "DAY (Generated)", fill='black')
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        cls_name = class_names.get(detection_info['class'], 'vehicle')
        filename = f"{os.path.splitext(img_name)[0]}_{cls_name}_{idx:03d}.png"
        save_path = os.path.join(self.output_dirs['comparisons'], filename)
        comp.save(save_path)
        return save_path

    def process_image(self, img_path):
        img_name = os.path.basename(img_path)
        self.stats['total_images'] += 1

        generated_tensor, night_tensor, original_size = self.generate_day_image(img_path)
        passed, disc_score = self.check_discriminator(generated_tensor)
        night_img = Image.open(img_path).convert('RGB')
        day_img = self.tensor_to_image(generated_tensor, original_size)

        # LPIPS
        lpips_score = self.lpips(night_tensor, generated_tensor).item()

        # Compute SIFT similarity
        sift_score = self.compute_sift_score(night_img, day_img)

        if self.config['save_full_comparison']:
            self.create_full_image_comparison(night_img, day_img, img_name, disc_score)

        # If discriminator fails:
        if not passed:
            self.stats['failed_discriminator'] += 1
            if self.config['save_failed_discriminator']:
                day_img.save(os.path.join(self.output_dirs['failed_disc'], f"failed_{img_name}"))
        else:
            self.stats['passed_discriminator'] += 1
            converted_path = os.path.join(self.output_dirs['converted'], f"day_{img_name}")
            day_img.save(converted_path)
            self.generated_for_fid.append(converted_path)

            # YOLO detect + debug + crop comparisons
            detections = self.detect_vehicles(day_img)
            if self.config['save_detection_debug']:
                debug = self.draw_detections(day_img, detections)
                debug.save(os.path.join(self.output_dirs['detections'], f"detect_{img_name}"))
            for idx, det in enumerate(detections):
                x1, y1, x2, y2 = det['box']
                night_crop = night_img.crop((x1, y1, x2, y2))
                day_crop = day_img.crop((x1, y1, x2, y2))
                self.create_comparison(night_crop, day_crop, det, img_name, idx)
                self.stats['total_comparisons_created'] += 1

        # Add metrics to CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([img_name, disc_score, lpips_score, sift_score])

    def run(self):
        print("Starting inference …")
        image_files = [
            f for f in os.listdir(self.config['input_dir'])
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        for img_file in tqdm(image_files, desc="Process images"):
            try:
                self.process_image(os.path.join(self.config['input_dir'], img_file))
            except Exception as e:
                print(f"Error on {img_file}: {e}")

        # Compute FID
        real_dir = self.config.get('real_day_dir', None)
        if real_dir and len(self.generated_for_fid) > 0:
            try:
                fid_val = fid.compute_fid(real_dir, self.output_dirs['converted'])
                print(f"FID score: {fid_val:.4f}")
            except Exception as e:
                print("Error computing FID:", e)
                fid_val = None

            # Write FID to CSV
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["FID_full_dataset", fid_val, "", ""])

        self.print_statistics()

    def print_statistics(self):
        print("\n--- Statistics ---")
        for k, v in self.stats.items():
            print(f"{k}: {v}")
        print(f"Evaluation CSV: {self.csv_path}")


if __name__ == "__main__":
    pipeline = Night2DayInference(CONFIG)
    pipeline.run()
