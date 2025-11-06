"""Grad-CAM (Gradient-weighted Class Activation Mapping) for explainability."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("plant_disease_detector")


class GradCAM:
    """Grad-CAM explainability for CNNs."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layer: Optional[torch.nn.Module] = None
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Model to explain
            target_layer: Target layer for gradient computation (if None, auto-detect)
        """
        self.model = model
        self.model.eval()
        
        # Auto-detect target layer if not provided
        if target_layer is None:
            target_layer = self._get_default_target_layer()
        
        self.target_layer = target_layer
        
        # Hooks
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = target_layer.register_full_backward_hook(self._backward_hook)
        
        logger.info(f"Grad-CAM initialized with target layer: {target_layer.__class__.__name__}")
    
    def _get_default_target_layer(self) -> torch.nn.Module:
        """Auto-detect the last convolutional layer."""
        # Try to find the last conv layer in backbone
        target_layer = None
        
        # Check if model has backbone attribute
        if hasattr(self.model, 'backbone'):
            backbone = self.model.backbone
            
            # Iterate through modules to find last conv layer
            for module in backbone.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        else:
            # Search in entire model
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
        
        if target_layer is None:
            raise ValueError("Could not auto-detect target layer. Please specify manually.")
        
        return target_layer
    
    def _forward_hook(self, module, input, output):
        """Hook for forward pass to capture activations."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook for backward pass to capture gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index (if None, use predicted class)
            
        Returns:
            CAM heatmap (H, W) normalized to [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        score = output[0, target_class]
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Compute weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU (only positive contributions)
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        input_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Generate heatmap overlaid on input image.
        
        Args:
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            input_size: Size to resize heatmap (H, W)
            
        Returns:
            Heatmap (H, W, 3) as numpy array [0, 255]
        """
        # Generate CAM
        cam = self.generate_cam(input_tensor, target_class)
        
        # Resize to input size
        if input_size is None:
            input_size = (input_tensor.shape[2], input_tensor.shape[3])
        
        cam_resized = cv2.resize(cam, (input_size[1], input_size[0]))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def overlay_heatmap(
        self,
        original_image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlay heatmap on original image.
        
        Args:
            original_image: Original image (H, W, 3) as numpy array
            input_tensor: Input tensor (1, C, H, W)
            target_class: Target class index
            alpha: Overlay transparency (0=only image, 1=only heatmap)
            
        Returns:
            Overlaid image (H, W, 3) as numpy array [0, 255]
        """
        # Get heatmap
        h, w = original_image.shape[:2]
        heatmap = self.generate_heatmap(input_tensor, target_class, input_size=(h, w))
        
        # Ensure original image is uint8
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8)
        
        # Overlay
        overlaid = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
        
        return overlaid
    
    def save_visualization(
        self,
        original_image: np.ndarray,
        input_tensor: torch.Tensor,
        save_path: str,
        target_class: Optional[int] = None,
        alpha: float = 0.4,
        show_heatmap_only: bool = False
    ):
        """
        Save Grad-CAM visualization.
        
        Args:
            original_image: Original image (H, W, 3)
            input_tensor: Input tensor (1, C, H, W)
            save_path: Path to save visualization
            target_class: Target class index
            alpha: Overlay transparency
            show_heatmap_only: If True, save heatmap only
        """
        if show_heatmap_only:
            h, w = original_image.shape[:2]
            heatmap = self.generate_heatmap(input_tensor, target_class, input_size=(h, w))
            result = heatmap
        else:
            result = self.overlay_heatmap(original_image, input_tensor, target_class, alpha)
        
        # Save
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, result_bgr)
        
        logger.info(f"Grad-CAM visualization saved to {save_path}")
    
    def __del__(self):
        """Remove hooks on deletion."""
        self.forward_hook.remove()
        self.backward_hook.remove()


def explain_prediction(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: np.ndarray,
    target_class: Optional[int] = None,
    target_layer: Optional[torch.nn.Module] = None,
    alpha: float = 0.4
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Grad-CAM explanation for a prediction.
    
    Args:
        model: Model to explain
        input_tensor: Input tensor (1, C, H, W)
        original_image: Original image (H, W, 3)
        target_class: Target class index
        target_layer: Target layer for Grad-CAM
        alpha: Overlay transparency
        
    Returns:
        Tuple of (heatmap, overlaid_image)
    """
    gradcam = GradCAM(model, target_layer)
    
    h, w = original_image.shape[:2]
    heatmap = gradcam.generate_heatmap(input_tensor, target_class, input_size=(h, w))
    overlaid = gradcam.overlay_heatmap(original_image, input_tensor, target_class, alpha)
    
    return heatmap, overlaid
