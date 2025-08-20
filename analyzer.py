import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from torchvision.models import ResNet50_Weights

class AIPhotoAnalyzer:
    def __init__(self):
        self.device = torch.device('cpu')
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def analyze_image(self, image):
        """Analyze a single image and return comprehensive results"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        results = {
            'composition': self._analyze_composition(image),
            'lighting': self._analyze_lighting(image),
            'technical': self._analyze_technical(image),
            'scene_type': self._classify_scene(image)
        }
        
        # Calculate overall score
        results['overall_score'] = (
            results['composition']['score'] * 0.4 +
            results['lighting']['score'] * 0.3 +
            results['technical']['score'] * 0.3
        )
        
        return results

    def _analyze_composition(self, image):
        # Convert to numpy array for analysis
        img_np = np.array(image)
        
        # Calculate rule of thirds
        h, w = img_np.shape[:2]
        thirds_h = h // 3
        thirds_w = w // 3
        
        # Check points of interest near intersection points
        interest_points = []
        for i in range(1, 3):
            for j in range(1, 3):
                region = img_np[
                    max(0, i * thirds_h - 20):min(h, i * thirds_h + 20),
                    max(0, j * thirds_w - 20):min(w, j * thirds_w + 20)
                ]
                interest_points.append(np.mean(region))
        
        thirds_score = np.mean(interest_points) / 255.0
        
        # Calculate symmetry
        left = img_np[:, :w//2]
        right = np.fliplr(img_np[:, w//2:])
        symmetry = 1 - np.mean(np.abs(left - right)) / 255.0
        
        score = (thirds_score + symmetry) / 2
        
        return {
            'score': float(score),
            'thirds': float(thirds_score),
            'symmetry': float(symmetry)
        }

    def _analyze_lighting(self, image):
        img_np = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        # Calculate various lighting metrics
        brightness = np.mean(l_channel) / 255.0
        contrast = np.std(l_channel) / 255.0
        
        # Check for over/under exposure
        overexposed = np.mean(l_channel > 250)
        underexposed = np.mean(l_channel < 5)
        
        score = (
            (1 - abs(brightness - 0.5)) * 0.4 +
            contrast * 0.3 +
            (1 - overexposed) * 0.15 +
            (1 - underexposed) * 0.15
        )
        
        return {
            'score': float(score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'overexposed': float(overexposed),
            'underexposed': float(underexposed)
        }

    def _analyze_technical(self, image):
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Calculate sharpness
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.mean(np.abs(laplacian))
        
        # Calculate noise
        noise = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        
        score = (
            min(sharpness / 100, 1.0) * 0.6 +
            (1 - min(noise, 1.0)) * 0.4
        )
        
        return {
            'score': float(score),
            'sharpness': float(min(sharpness / 100, 1.0)),
            'noise': float(min(noise, 1.0))
        }

    def _classify_scene(self, image):
        try:
            input_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            _, predicted = torch.max(output.data, 1)
            return {
                'scene_type': str(predicted.item()),
                'confidence': float(probabilities[predicted].item())
            }
        except Exception as e:
            return {
                'scene_type': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
