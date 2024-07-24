# Comfy EnhancedMultiRegion

## What this node does: 

This is simple custom node for [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) which helps to generate images with _regional prompting_ way easier.

If you want to draw different regions together without blending their features, check out this custom node.

| ⭕ with ComfyEnhancedMultiRegion | ❌ without ComfyEnhancedMultiRegion |
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
3. Connect inputs (as many positive conditionings as you need), leave the rest unconnected
Connect outputs

Example workflow is [here](workflows/workflow-comfy-couple.json).


## Credits

- [**@laksjdjf**](https://github.com/laksjdjf) – [Attention-Couple](https://github.com/laksjdjf/attention-couple-ComfyUI) and [Comfy-Couple](https://github.com/Danand/ComfyUI-ComfyCouple).
- [**@pythongosssss**](https://github.com/pythongosssss) – [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) used for capturing SVG for `README.md`
- [**@Meina**](https://civitai.com/user/Meina) – [MeinaMix V11](https://civitai.com/models/7240/meinamix) used in example.
