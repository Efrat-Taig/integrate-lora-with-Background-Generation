# integrate-lora-with-Background-Generation

This repository provides guidance and tools for connecting a LoRA (Low-Rank Adaptation) model to a Bira's [background generation model](https://huggingface.co/briaai/BRIA-2.3-ControlNet-BG-Gen). This process allows you to enhance the capabilities of your text-to-image generation pipeline by combining the strengths of both models.

## Overview

In this guide, we'll walk you through the process of:

1. Loading a pre-trained LoRA model
2. Connecting it to a compatible background generation model
3. Using the combined model to generate images with custom backgrounds

This README builds upon our previous guide on training LoRA models, which can be found [here](https://github.com/Efrat-Taig/training-lora/tree/main).

## Prerequisites

Before you begin, make sure you have:

- A trained LoRA model (see our previous guide on LoRA training)
- A compatible background generation model (model card [here](https://huggingface.co/briaai/BRIA-2.3-ControlNet-BG-Gen))
- Required libraries and installation specified [here](https://huggingface.co/briaai/BRIA-2.3-ControlNet-BG-Gen)

## Important Note

Remember that LoRA models must be connected to the same type of foundation model they were trained on. For example, a LoRA model trained on [Bria 2.3](https://huggingface.co/briaai/BRIA-2.3) can only be connected to other Bria 2.3-based models, not to SDXL or other model types.



## Getting Started

Follow these steps to get started with connecting your LoRA model to a background generator:

1. **Run the Initial Code**
   - First, run the initial code [run_BG.py](https://github.com/Efrat-Taig/integrate-lora-with-Background-Generation/blob/main/run_BG.py) to ensure everything is set up correctly and all installations are in order.
   - This should be a straightforward process, following the installation instructions in the model card [here](https://huggingface.co/briaai/BRIA-2.3-ControlNet-BG-Gen).
   - If you encounter any issues during this step, please don't hesitate to reach out for assistance.

2. **Select a Foreground Image**
   - Choose an image from which you'll extract the foreground object.
   - For example, I used my profile picture, but you can select any image you prefer.

3. **Connect the LoRA Model**
   - We'll now connect the LoRA model that we trained in the previous tutorial.
   - Run the script [run_BG_with_LORA.py](https://github.com/Efrat-Taig/integrate-lora-with-Background-Generation/blob/main/run_BG_with_LORA.py) to combine the LoRA model with the background generator.
  

  ## Results

Before we dive into the results of connecting our LoRA model to a background generator, it's important to understand the context of our work. This project builds upon our previous repository on training LoRA models, which you can find [here](https://github.com/Efrat-Taig/training-lora/tree/main).

### Background

In our previous work:
1. We created a dataset of [Modern Blurred SeaView](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView) .
2. We [trained](https://github.com/Efrat-Taig/training-lora/edit/main/README.md) a LoRA model on this dataset.
3. We demonstrated how the model improved over time, generating images increasingly similar to our dataset.

Here's a sample image from our dataset:
Sample from my [Modern Blurred SeaView](https://huggingface.co/datasets/Negev900/Modern_Blurred_SeaView) Dataset:

<img src="https://github.com/Efrat-Taig/training-lora/blob/main/Data_set_sample.png" width="400">>


And here's an example of how our LoRA model improved over time:

<img src="https://github.com/Efrat-Taig/training-lora/blob/main/lora_res_1.png" width="400">


For a full understanding of the LoRA training process and to see more examples, please refer to our previous repository [here](https://github.com/Efrat-Taig/training-lora/tree/main)or to this [paper](https://github.com/Efrat-Taig/training-lora/tree/main)


### Current Project: Connecting LoRA to Background Generator

In this project, we're taking the next step: connecting our trained LoRA model to a background generation model. This allows us to apply the style and characteristics learned by our LoRA model to generate new backgrounds while preserving foreground subjects.

To achieve this, we need to run the script [run_BG_with_LORA.py](https://github.com/Efrat-Taig/integrate-lora-with-Background-Generation/blob/main/run_BG_with_LORA.py). 

#### Purpose of the Script

The main purpose of running this script is to integrate our trained LoRA model with the background replacement process. Here's what the script does:

1. It loads our pre-trained LoRA model.
2. It connects the LoRA model to the background generation model.
3. It applies the style and characteristics learned by our LoRA to the background generation process.

By running this script, we're essentially telling the background generation model to consider the patterns and styles it learned from our specific dataset (in this case, modern blurred sea views) when creating new backgrounds.

#### How to Run the Script

To run the script, use the following command in your terminal:

```
python run_BG_with_LORA.py
```

Make sure you have all the necessary dependencies installed and that you're in the correct directory before running the script.

#### Expected Outcome

After running the script, you should be able to generate new backgrounds that not only replace the original background but also reflect the style of your trained LoRA model. This means the new backgrounds should resemble the modern, blurred sea views from our training dataset, while still preserving the foreground subject of the original image.

The following image series demonstrates the evolution of our background generation process, starting from the original picture, progressing through a basic background swap without LoRA, and then showing the improvements achieved at various checkpoints of our LoRA model integration:

<img src="https://github.com/Efrat-Taig/integrate-lora-with-Background-Generation/blob/main/Lora_bg.png"  alt="LoRA Background Generation Progress">

## Final Notes

This project demonstrates the powerful capabilities of integrating LoRA models with background generation techniques. By fine-tuning our model on a specific dataset of modern, blurred sea views, we've shown how LoRA can significantly enhance the quality and style-consistency of generated backgrounds.

Key takeaways:
- LoRA integration allows for more controlled and stylistically consistent background generation.
- The progression through LoRA checkpoints shows clear improvements in adhering to the desired style.
- This technique preserves the original subject while transforming the background, offering a versatile tool for image editing.

We encourage you to experiment with this approach using your own datasets and style preferences. The possibilities for creative applications are vast, from personalized photo editing to creating themed visual content.


We welcome your feedback, contributions, and any creative uses you find for this technology. If you have questions or want to share your results, please open an issue or submit a pull request in this repository.


If you have any questions or need further assistance, feel free to reach out to me here, or connect via email (efrat@bria.ai) and [LinkedIn](https://www.linkedin.com/in/efrattaig/). 

For academic users interested in accessing the model, please [register here]([link](https://docs.google.com/forms/d/1sSjxqS_2T4RB0dxnPjpygm7EXxa3RYNm2e4PUXQKnLo/edit)) for access and further details. You can also refer to this [model information link](https://huggingface.co/briaai) or [article](https://medium.com/@efrat_37973/bridging-the-gap-from-academic-ai-to-ethical-business-models-89327517b940) for additional insights.





