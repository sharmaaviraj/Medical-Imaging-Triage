"""
Native PyTorch Grad-CAM (Gradient-weighted Class Activation Mapping) implementation
for the ResNet-18 classification model.

Provides visual feature attribution by mapping convolutional gradient activations
from layer4[-1] back to the spatial dimensions of the input dermoscopic image.
"""

from typing import Tuple, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image


class GradCAMExplainer:
    """
    Computes class-activation heatmaps for ResNet-18 without external dependencies.
    Targets model.layer4[-1] (the final residual block before global pooling).
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self.activations = []
        self.gradients = []
        self.handles = []

    def _register_hooks(self):
        self.activations.clear()
        self.gradients.clear()

        def forward_hook(module, input_tensor, output_tensor):
            self.activations.append(output_tensor)

        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.handles = [h_fwd, h_bwd]

    def _remove_hooks(self):
        for handle in self.handles:
            try:
                handle.remove()
            except Exception:
                pass
        self.handles.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: Image.Image,
        alpha: float = 0.40
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates Grad-CAM visual explanation for the malignant class output.

        Parameters:
            input_tensor: Preprocessed tensor (1, 3, 224, 224).
            original_image: Original PIL image for dimension reference and overlay.
            alpha: Heatmap blend opacity (0.0 to 1.0).

        Returns:
            Tuple of:
            - normalized_heatmap: 2D float array in [0.0, 1.0] matching original image size.
            - heatmap_rgb: 3D uint8 RGB array of the JET colormap representation.
            - overlay_rgb: 3D uint8 RGB array of the blended image.
        """
        orig_array = np.array(original_image.convert("RGB"))
        height, width = orig_array.shape[:2]

        self._register_hooks()

        try:
            # Enable gradients locally while keeping model in eval mode
            with torch.enable_grad():
                tensor_clone = input_tensor.clone().detach().requires_grad_(True)
                self.model.eval()
                self.model.zero_grad()

                logits = self.model(tensor_clone)

                # For binary classification with 1 scalar output, backprop the scalar logit
                # representing evidence towards the malignant class
                scalar_logit = logits[0, 0] if logits.ndim > 1 else logits[0]
                scalar_logit.backward(retain_graph=False)

            if not self.activations or not self.gradients:
                # Fallback for unexpected hook capture failure
                blank_map = np.zeros((height, width), dtype=np.float32)
                blank_rgb = np.zeros((height, width, 3), dtype=np.uint8)
                return blank_map, blank_rgb, orig_array.copy()

            # Shape: (1, 512, 7, 7)
            act = self.activations[0]
            grad = self.gradients[0]

            # Channel importance via Global Average Pooling of gradients: shape (1, 512, 1, 1)
            weights = grad.mean(dim=(2, 3), keepdim=True)

            # Weighted feature map combination
            cam = (weights * act).sum(dim=1, keepdim=True)

            # Apply ReLU: keep only features contributing positively to malignancy logit
            cam = torch.relu(cam)

            # Convert to numpy array: shape (7, 7)
            cam_np = cam.squeeze().detach().cpu().numpy()

            # Robust min-max scaling to [0, 1]
            cam_min = float(cam_np.min())
            cam_max = float(cam_np.max())
            spread = cam_max - cam_min

            if spread > 1e-8:
                cam_norm = (cam_np - cam_min) / spread
            else:
                # Uniform zero map if activations or gradients are completely flat
                cam_norm = np.zeros_like(cam_np, dtype=np.float32)

            # Upsample to original image resolution
            heatmap_resized = cv2.resize(
                cam_norm,
                (width, height),
                interpolation=cv2.INTER_LINEAR
            )

            # Generate colored heatmap using standard JET colormap
            heatmap_uint8 = np.uint8(np.clip(255 * heatmap_resized, 0, 255))
            heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

            # Generate alpha-blended overlay on original image
            blend_alpha = float(np.clip(alpha, 0.1, 0.9))
            overlay_float = (1.0 - blend_alpha) * orig_array + blend_alpha * heatmap_rgb
            overlay_rgb = np.uint8(np.clip(overlay_float, 0, 255))

            return heatmap_resized, heatmap_rgb, overlay_rgb

        finally:
            self._remove_hooks()
            self.model.zero_grad()
