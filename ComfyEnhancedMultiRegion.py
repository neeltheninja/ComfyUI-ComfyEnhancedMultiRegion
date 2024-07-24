import math
from nodes import MAX_RESOLUTION, ConditioningCombine, ConditioningSetMask
from comfy_extras.nodes_mask import MaskComposite, SolidMask
from .attention_couple import AttentionCouple

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
            },
            "optional": {
                **{f"positive_{i+1}": ("CONDITIONING",) for i in range(10)},  # Support up to 10 regions
                **{f"ratio_{i+1}": ("FLOAT", {"default": 0.5, "min": 0, "max": 1.0, "step": 0.01}) for i in range(9)}  # n-1 ratios needed for n regions
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "POSITIVE",
        "NEGATIVE",
    )

    FUNCTION = "process"
    CATEGORY = "loaders"

    def process(self, model, negative, orientation, num_regions, width, height, **kwargs):
        positives = [kwargs.get(f"positive_{i+1}") for i in range(num_regions)]
        ratios = [kwargs.get(f"ratio_{i+1}", 1.0 / num_regions) for i in range(num_regions - 1)]

        if any(pos is None for pos in positives):
            raise ValueError(f"Expected {num_regions} positive conditionings, but some are missing")

        # Normalize ratios
        ratios.append(1.0 - sum(ratios))
        total = sum(ratios)
        ratios = [r / total for r in ratios]

        # Create masks for each region
        solid_mask_zero = SolidMask().solid(0.0, width, height)[0]
        masks = []
        start = 0

        for i, ratio in enumerate(ratios):
            if orientation == "horizontal":
                region_width = math.floor(width * ratio)
                mask_rect = (start, 0, region_width, height)
                start += region_width
            else:  # vertical
                region_height = math.floor(height * ratio)
                mask_rect = (0, start, width, region_height)
                start += region_height

            solid_mask = SolidMask().solid(1.0, mask_rect[2], mask_rect[3])[0]
            mask_composite = MaskComposite().combine(solid_mask_zero, solid_mask, mask_rect[0], mask_rect[1], "add")[0]
            masks.append(mask_composite)

        # Apply masks to positive conditionings
        conditioned_masks = [ConditioningSetMask().append(pos, mask, "default", 1.0)[0] for pos, mask in zip(positives, masks)]

        # Combine all conditioned masks
        positive_combined = conditioned_masks[0]
        for mask in conditioned_masks[1:]:
            positive_combined = ConditioningCombine().combine(positive_combined, mask)[0]

        return AttentionCouple().attention_couple(model, positive_combined, negative, "Attention")

NODE_CLASS_MAPPINGS = {
    "Comfy Multi-Region": ComfyMultiRegion
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Comfy Multi-Region": "Comfy Multi-Region",
}
