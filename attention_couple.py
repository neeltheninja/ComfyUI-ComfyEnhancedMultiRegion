import torch
import torch.nn.functional as F
import copy
import comfy
from comfy.ldm.modules.attention import optimized_attention

def get_masks_from_q(masks, q, original_shape):
    if original_shape[2] * original_shape[3] == q.shape[1]:
        down_sample_rate = 1
    elif (original_shape[2] // 2) * (original_shape[3] // 2) == q.shape[1]:
        down_sample_rate = 2
    elif (original_shape[2] // 4) * (original_shape[3] // 4) == q.shape[1]:
        down_sample_rate = 4
    else:
        down_sample_rate = 8

    ret_masks = []
    for mask in masks:
        if isinstance(mask, torch.Tensor):
            size = (original_shape[2] // down_sample_rate, original_shape[3] // down_sample_rate)
            mask_downsample = F.interpolate(mask.unsqueeze(0), size=size, mode="nearest")
            mask_downsample = mask_downsample.view(1,-1, 1).repeat(q.shape[0], 1, q.shape[2])
            ret_masks.append(mask_downsample)
        else:  # coupling処理なしの場合
            ret_masks.append(torch.ones_like(q))

    ret_masks = torch.cat(ret_masks, dim=0)
    return ret_masks

def set_model_patch_replace(model, patch, key):
    to = model.model_options["transformer_options"]
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    to["patches_replace"]["attn2"][key] = patch

class AttentionCouple:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "mode": (["Attention", "Latent"], ),
                "isolation_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "attention_couple"
    CATEGORY = "loaders"

    def attention_couple(self, model, positive, negative, mode, isolation_factor):
        if mode == "Latent":
            return (model, positive, negative)  # latent coupleの場合は何もしない

        self.negative_positive_masks = []
        self.negative_positive_conds = []
        self.isolation_factor = isolation_factor

        new_positive = copy.deepcopy(positive)
        new_negative = copy.deepcopy(negative)

        dtype = model.model.diffusion_model.dtype
        device = comfy.model_management.get_torch_device()

        # maskとcondをリストに格納する
        for conditions in [new_negative, new_positive]:
            conditions_masks = []
            conditions_conds = []
            if len(conditions) != 1:
                mask_norm = torch.stack([cond[1]["mask"].to(device, dtype=dtype) * cond[1]["mask_strength"] for cond in conditions])
                mask_norm = mask_norm / mask_norm.sum(dim=0)  # 合計が1になるように正規化(他が0の場合mask_strengthの効果がなくなる)
                conditions_masks.extend([mask_norm[i] for i in range(mask_norm.shape[0])])
                conditions_conds.extend([cond[0].to(device, dtype=dtype) for cond in conditions])
                del conditions[0][1]["mask"]  # latent coupleの無効化のため
                del conditions[0][1]["mask_strength"]
            else:
                conditions_masks = [False]
                conditions_conds = [conditions[0][0].to(device, dtype=dtype)]
            self.negative_positive_masks.append(conditions_masks)
            self.negative_positive_conds.append(conditions_conds)
        self.conditioning_length = (len(new_negative), len(new_positive))

        new_model = model.clone()
        self.sdxl = hasattr(new_model.model.diffusion_model, "label_emb")
        if not self.sdxl:
            for id in [1,2,4,5,7,8]:  # id of input_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[0].attn2), ("input", id))
            set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[0].attn2), ("middle", 0))
            for id in [3,4,5,6,7,8,9,10,11]:  # id of output_blocks that have cross attention
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[0].attn2), ("output", id))
        else:
            for id in [4,5,7,8]:  # id of input_blocks that have cross attention
                block_indices = range(2) if id in [4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.input_blocks[id][1].transformer_blocks[index].attn2), ("input", id, index))
            for index in range(10):
                set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.middle_block[1].transformer_blocks[index].attn2), ("middle", id, index))
            for id in range(6):  # id of output_blocks that have cross attention
                block_indices = range(2) if id in [3, 4, 5] else range(10)  # transformer_depth
                for index in block_indices:
                    set_model_patch_replace(new_model, self.make_patch(new_model.model.diffusion_model.output_blocks[id][1].transformer_blocks[index].attn2), ("output", id, index))

        return (new_model, [new_positive[0]], [new_negative[0]])  # pool outputは・・・後回し

    def make_patch(self, module):           
        def patch(q, k, v, extra_options):
            len_neg, len_pos = self.conditioning_length
            cond_or_uncond = extra_options["cond_or_uncond"]
            q_list = q.chunk(len(cond_or_uncond), dim=0)
            b = q_list[0].shape[0]

            masks_uncond = get_masks_from_q(self.negative_positive_masks[0], q_list[0], extra_options["original_shape"])
            masks_cond = get_masks_from_q(self.negative_positive_masks[1], q_list[0], extra_options["original_shape"])

            cond_size = self.negative_positive_conds[1][0].shape[1]
            context_uncond = torch.cat([cond[:, :cond_size] for cond in self.negative_positive_conds[0]], dim=0)
            context_cond = torch.cat([cond[:, :cond_size] for cond in self.negative_positive_conds[1]], dim=0)

            k_uncond = module.to_k(context_uncond)
            k_cond = module.to_k(context_cond)
            v_uncond = module.to_v(context_uncond)
            v_cond = module.to_v(context_cond)

            out = []
            for i, c in enumerate(cond_or_uncond):
                if c == 0:
                    masks = masks_cond
                    k = k_cond
                    v = v_cond
                    length = len_pos
                else:
                    masks = masks_uncond
                    k = k_uncond
                    v = v_uncond
                    length = len_neg

                q_target = q_list[i].repeat(length, 1, 1)
                k = torch.cat([k[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)
                v = torch.cat([v[i].unsqueeze(0).repeat(b,1,1) for i in range(length)], dim=0)

                # Convert all tensors to the same dtype as q_target
                k = k.to(dtype=q_target.dtype)
                v = v.to(dtype=q_target.dtype)
                masks = masks.to(dtype=q_target.dtype)

                # Apply sharpened masks based on isolation factor
                sharpened_masks = self.sharpen_masks(masks, self.isolation_factor)
                
                qkv = optimized_attention(q_target, k, v, extra_options["n_heads"])
                qkv = qkv * sharpened_masks
                qkv = qkv.view(length, b, -1, module.heads * module.dim_head).sum(dim=0)

                out.append(qkv)

            out = torch.cat(out, dim=0)
            return out
        return patch

    def sharpen_masks(self, masks, isolation_factor):
        # Convert isolation_factor to a tensor with the same device and dtype as masks
        isolation_factor_tensor = torch.tensor(isolation_factor, device=masks.device, dtype=masks.dtype)
        
        # Create a sharper transition based on the isolation factor
        sharpened = torch.pow(masks, torch.exp(isolation_factor_tensor))
        
        # Normalize the sharpened masks
        sharpened = sharpened / (sharpened.sum(dim=0, keepdim=True) + 1e-6)
        
        return sharpened

NODE_CLASS_MAPPINGS = {
    "Attention couple": AttentionCouple
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Attention couple": "Load Attention couple",
}