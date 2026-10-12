# Comfy EnhancedMultiRegion
![shapes](docs/images/image.png)
## What this node does: 

This is simple custom node for [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) which helps to generate images with _regional prompting_ way easier.

If you want to draw different regions together without blending their features, check out this custom node.

| ⭕ with ComfyEnhancedMultiRegion (I mistakenly kept "3D rendered cone" in all of the prompts XD)| ❌ without ComfyEnhancedMultiRegion |
| --- | --- |
| ![shapes](docs/images/ComfyUI_postColor_00295_.png) | ![shapes](docs/images/ComfyUI_postColor_00296_.png) |
| Prompt 1: ((blue cube)), 3D rendered cone,  white background, (far left)
Prompt 2: ((yellow sphere)), 3D rendered cone,  red background 
Prompt 3: ((purple straight cone)), 3D rendered cone, black background, (far right) | _Single Prompt_: ((blue cube)), 3D rendered cone,  white background, (far left), ((yellow sphere)), 3D rendered cone,  red background, ((purple straight cone)), 3D rendered cone, black background, (far right) |

This is a fork of [**Danand/ComfyUI-ComfyCouple**](https://github.com/Danand/ComfyUI-ComfyCouple), implementing FP16 support and upto 10 different regions in a single image.

## Installation

1. Change directory to custom nodes of **ComfyUI**:

   ```bash
   cd ~/ComfyUI/custom_nodes
   ```

2. Clone this repo here:

   ```bash
   git clone https://github.com/neeltheninja/ComfyUI-ComfyEnhancedMultiRegion.git
   ```

3. Restart **ComfyUI**.

## Usage

1. Right click in workflow.
2. Choose node: **loaders → ComfyEnhancedMultiRegion**
3. Connect inputs (as many positive conditionings as you need), a single negative conditioning, and the clips. Convert the width and height Widgets to Inputs and feed the image latent size to the node. (Refer to my workflow).
4. Connect outputs

Example workflow is [here](workflows/MultiRegionalWorkflow.json).

### Model type (SDXL / Anima)

The `model_type` input (last widget on the node) selects which architecture the attention patch targets:

- **SDXL (Illustrious / Pony / SDXL Base)**: default, same behavior as before (SD1.5 is still detected automatically).
- **Anima**: for [Anima](https://huggingface.co/circlestone-labs/Anima) (Cosmos-Predict2 DiT). Encode the region prompts with the Anima text encoder (Qwen3 0.6B) as in a normal Anima workflow; the node passes them through the model's LLM adapter internally. The region layout is read from the real latent at sampling time, so `width`/`height` only define the proportions of the regions (they do not have to match the image size). Preferably use multiples of 16 (one Anima token covers 16×16 pixels).

If `model_type` does not match the connected model, the node prints an error to the console and outputs nothing.

**Recommended starting point (Anima)**

`isolation_factor` 0, `cross_region_blend` 0, `2girls`/interaction words as needed in the region prompts, optionally a `base_positive` with the interaction and scene. In a test with two characters touching across the region border this was enough to keep the attributes of each region separate (cross-attention is already limited to each region's own prompt), and it is the setting where arms and hands can cross the border. Only raise `isolation_factor` if attributes still leak.

**Reducing leakage between regions (Anima)**

Two inputs control leakage:

- `cross_region_blend`: mixes the output of regions whose prompts are similar. Prompts that share most tags (quality tags, `2girls`, ...) count as similar, so this leaks by design (the default 0.3 is meant for SDXL). Set it to 0 first.
- `isolation_factor`: strength of the self-attention block between regions (0 = off, 1 = fully blocked; attention towards other regions is scaled by `(1 - isolation_factor)²`). Anima's self-attention is global, so this is the knob that stops text bound to one region (e.g. `red eyes`) from leaking into another through the image tokens. It also cuts anything that crosses the border (an arm is cut off at the border at 0.9+) and can make the regions look disconnected, so use the lowest value that works.

**Keeping interaction between regions (Anima)**

Isolation and interaction pull in opposite directions: the same self-attention that can leak `red eyes` is also what lets two characters touch, look at each other or share a pose. If you need some isolation but lose interaction, `isolation_start` and `isolation_end` (fractions of the sampling steps, 0 = first/noisiest step) limit isolation to a window:

- `isolation_start` 0.2-0.4: the first steps run without isolation, so pose and composition are decided with the regions "seeing" each other, then isolation takes over while colors and details form.
- `isolation_end` below 1: isolation stops before the last steps so the regions can blend.
- The defaults (0 and 1) isolate during the whole run. In a hires pass with denoise below 1 the window is relative to that pass's own steps.

**Characters touching across the border / visible seams (Anima)**

A region only sees its own prompt, so a hand that crosses the border is drawn by whichever region it lands in, and that region's prompt does not mention it. Two optional inputs help:

- `base_positive` + `base_weight`: a prompt for the *whole* image, mixed into every region (`base_weight` 0.3-0.5). Put the interaction (`facing each other, hand on the other girl's shoulder, 2girls`) and the shared scene/quality tags here, and keep the per-character attributes (hair, eyes, outfit) in the region prompts. The negative pass is not affected.
- `mask_feather`: softens the region borders over that many tokens (1 token = 16 px; try 2-3). Tokens inside the soft band are also left out of the self-attention isolation, so the two regions can connect there.

Both default to off, and the widgets are placed last so existing workflows keep their values.

On SDXL, `isolation_factor` only sharpens soft masks and has no effect on the hard masks this node generates.



## Credits

- [**@laksjdjf**](https://github.com/laksjdjf) – [Attention-Couple](https://github.com/laksjdjf/attention-couple-ComfyUI) and [**@Danand**](https://github.com/Danand) – [Comfy-Couple](https://github.com/Danand/ComfyUI-ComfyCouple).
