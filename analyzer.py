import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision.models import ResNet50_Weights
import streamlit as st

class AIPhotoAnalyzer:
    def __init__(self):
        self.device = torch.device('cpu')  # Use CPU for better compatibility
        
        try:
            self.scene_classifier = models.resnet50(weights=ResNet50_Weights.DEFAULT)
            self.scene_classifier.eval()
            self.scene_classifier.to(self.device)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.scene_classifier = None
        
        self.categories = [f"Category_{i}" for i in range(1000)]
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def analyze_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        results = {
            'scene_type': self.detect_scene(image),
            'composition_score': self.analyze_composition(image),
            'lighting_quality': self.analyze_lighting(image),
            'technical_quality': self.assess_technical_quality(image)
        }
        return results

    def detect_scene(self, image):
        if self.scene_classifier is None:
            return [("Model not loaded", 0.0)]
        
        try:
            img_tensor = self.transform(image)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.scene_classifier(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            top3_prob, top3_catid = torch.topk(probabilities, 3)
            return [(self.categories[idx], prob.item()) for prob, idx in zip(top3_prob, top3_catid)]
        except Exception as e:
            st.error(f"Scene detection error: {str(e)}")
            return [("Error in scene detection", 0.0)]

    def analyze_composition(self, image):
        try:
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.mean(edges > 0)
            
            h, w = gray.shape
            left_half = gray[:, :w//2]
            right_half = np.fliplr(gray[:, w//2:])
            symmetry_score = 1 - np.mean(np.abs(left_half - right_half)) / 255.0
            
            h_thirds = h // 3
            w_thirds = w // 3
            thirds_score = 0
            for i in range(1, 3):
                for j in range(1, 3):
                    region = edges[
                        max(0, i * h_thirds - 20):min(h, i * h_thirds + 20),
                        max(0, j * w_thirds - 20):min(w, j * w_thirds + 20)
                    ]
                    thirds_score += np.mean(region > 0)
            
            composition_score = edge_density * 0.3 + symmetry_score * 0.3 + thirds_score * 0.4
            
            return {
                'score': float(min(max(composition_score, 0), 1)),
                'edge_density': float(edge_density),
                'symmetry': float(symmetry_score),
                'thirds_alignment': float(thirds_score)
            }
        except Exception as e:
            st.error(f"Composition analysis error: {str(e)}")
            return {
                'score': 0.0,
                'edge_density': 0.0,
                'symmetry': 0.0,
                'thirds_alignment': 0.0
            }

    def analyze_lighting(self, image):
        try:
            img_np = np.array(image)
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            brightness = float(np.mean(l_channel) / 255.0)
            contrast = float(np.std(l_channel) / 255.0)
            color_temp = float(np.mean(b_channel) - np.mean(a_channel)) / 255.0
            dynamic_range = float(np.percentile(l_channel, 95) - np.percentile(l_channel, 5)) / 255.0
            
            l_hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
            overexposed = float(np.sum(l_hist[-10:]) / np.sum(l_hist))
            underexposed = float(np.sum(l_hist[:10]) / np.sum(l_hist))
            color_balance = 1.0 - abs(float(np.mean(a_channel) - np.mean(b_channel))) / 255.0
            
            lighting_score = (
                brightness * 0.25 +
                contrast * 0.25 +
                dynamic_range * 0.2 +
                (1 - overexposed) * 0.1 +
                (1 - underexposed) * 0.1 +
                color_balance * 0.1
            )
            
            return {
                'score': float(min(max(lighting_score, 0), 1)),
                'brightness': brightness,
                'contrast': contrast,
                'color_temperature': color_temp,
                'dynamic_range': dynamic_range,
                'overexposed': overexposed,
                'underexposed': underexposed,
                'color_balance': color_balance
            }
        except Exception as e:
            st.error(f"Lighting analysis error: {str(e)}")
            return {
                'score': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'color_temperature': 0.0,
                'dynamic_range': 0.0,
                'overexposed': 0.0,
                'underexposed': 0.0,
                'color_balance': 0.0
            }

    def assess_technical_quality(self, image):
        try:
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(np.mean(np.abs(laplacian)) / 255.0)
            
            noise = float(np.std(gray) / 255.0)
            
            local_var = cv2.blur(gray.astype(float)**2, (5,5)) - cv2.blur(gray.astype(float), (5,5))**2
            detail_score = float(np.mean(local_var) / 255.0)
            
            technical_score = sharpness * 0.4 + (1 - noise) * 0.3 + detail_score * 0.3
            
            return {
                'score': float(min(max(technical_score, 0), 1)),
                'sharpness': sharpness,
                'noise_level': noise,
                'detail_level': detail_score
            }
        except Exception as e:
            st.error(f"Technical quality assessment error: {str(e)}")
            return {
                'score': 0.0,
                'sharpness': 0.0,
                'noise_level': 0.0,
                'detail_level': 0.0
            }
