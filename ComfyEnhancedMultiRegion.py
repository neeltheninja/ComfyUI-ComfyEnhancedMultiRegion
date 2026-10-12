import math
import torch
from nodes import MAX_RESOLUTION, ConditioningCombine, ConditioningSetMask
from comfy_extras.nodes_mask import MaskComposite, SolidMask
from .attention_couple import AttentionCouple, MODEL_TYPES

class ComfyMultiRegion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "negative": ("CONDITIONING",),
                "orientation": (["horizontal", "vertical"],),
                "num_regions": ("INT", {"default": 2, "min": 2, "max": 10, "step": 1}),
                "width": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8}),
                "isolation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Anima: how strongly self-attention between regions is blocked (0 = off, 1 = fully blocked). SDXL: only sharpens soft masks, no effect on the hard masks of this node."}),
            },
            "optional": {
                "cross_region_blend": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Mixes the cross-attention output of regions with similar prompts. Set to 0 if attributes leak between regions."}),
                **{f"positive_{i+1}": ("CONDITIONING",) for i in range(10)},
                **{f"ratio_{i+1}": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}) for i in range(9)},
                **{f"weight_{i+1}": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}) for i in range(10)},
                # kept last so widget values of existing workflows stay in place
                "model_type": (MODEL_TYPES,),
                "isolation_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Anima: fraction of the sampling steps (0 = first, noisiest step) that run WITHOUT isolation. Early steps decide pose and composition, so a value around 0.2-0.4 lets the regions interact before isolation takes over."}),
                "isolation_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Anima: isolation stops at this fraction of the sampling steps (1 = until the end). Lower it to let the regions blend in the last steps."}),
                "base_positive": ("CONDITIONING", {"tooltip": "Anima: prompt for the WHOLE image (the interaction, e.g. 'hand on the other girl's shoulder', plus scene and shared tags). Leave out per-character attributes."}),
                "base_weight": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Anima: how much of the base prompt is mixed into every region (0 = off). Only used when base_positive is connected."}),
                "mask_feather": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1, "tooltip": "Anima: soften the region borders over this many tokens (1 token = 16 px). Hides the seam between regions; 0 = hard border."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING",)
    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(self, model, negative, orientation, num_regions, width, height, isolation_factor, **kwargs):
        try:
            cross_region_blend = kwargs.get("cross_region_blend", 0.3)
            model_type = kwargs.get("model_type", MODEL_TYPES[0])
            isolation_start = kwargs.get("isolation_start", 0.0)
            isolation_end = kwargs.get("isolation_end", 1.0)
            base_positive = kwargs.get("base_positive")
            base_weight = kwargs.get("base_weight", 0.4)
            mask_feather = kwargs.get("mask_feather", 0)
            positives = [kwargs.get(f"positive_{i+1}") for i in range(num_regions)]
            ratios = [kwargs.get(f"ratio_{i+1}", 1.0 / num_regions) for i in range(num_regions - 1)]
            weights = [kwargs.get(f"weight_{i+1}", 1.0) for i in range(num_regions)]

            if any(pos is None for pos in positives):
                raise ValueError(f"Expected {num_regions} positive conditionings, but some are missing")

            # Normalize ratios
            ratios.append(max(0, 1.0 - sum(ratios)))  # Ensure non-negative
            total = sum(ratios)
            ratios = [r / total for r in ratios]

            # Create masks for each region
            masks = self.create_masks(ratios, orientation, width, height)

            # Apply masks and weights to positive conditionings
            conditioned_masks = [ConditioningSetMask().append(pos, mask, "default", weight)[0] for pos, mask, weight in zip(positives, masks, weights)]

            # Combine all conditioned masks
            positive_combined = conditioned_masks[0]
            for mask in conditioned_masks[1:]:
                positive_combined = ConditioningCombine().combine(positive_combined, mask)[0]

            return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention", isolation_factor, cross_region_blend, model_type, isolation_start, isolation_end, base_positive, base_weight, mask_feather)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None, None, None

    @staticmethod
    def create_masks(ratios, orientation, width, height):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        masks = torch.zeros((len(ratios), height, width), device=device)
        start = 0

        for i, ratio in enumerate(ratios):
            if orientation == "horizontal":
                region_width = math.floor(width * ratio)
                end = min(start + region_width, width)
                masks[i, :, start:end] = 1.0
            else:  # vertical
                region_height = math.floor(height * ratio)
                end = min(start + region_height, height)
                masks[i, start:end, :] = 1.0
            start = end

        return masks

NODE_CLASS_MAPPINGS = {
    "Comfy Multi-Region": ComfyMultiRegion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfy Multi-Region": "Comfy Multi-Region",
}