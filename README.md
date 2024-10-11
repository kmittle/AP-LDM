# AP-LDM
Official implementation of [***AP-LDM: Attentive and Progressive Latent Diffusion Model for Training-Free High-Resolution Image Generation***](https://arxiv.org/abs/2410.06055v1).

![image](fig/teaser.png) 

**AP-LDM** is a framework that enables the rapid synthesis of high-quality, high-resolution images without the need for retraining.

It consists of two stages: (1) synthesizing high-quality images at the training resolution using **Attentive Guidance**, and (2) generating finer high-resolution images through pixel upsampling and "diffusion-denoising."

## Overview of AP-LDM
![image](fig/AP-LDM.png) 
* Attentive Guidance enhances the structural consistency of the latent representation using a parameter-free self-attention mechanism, which is achieved through linear weighting.
* It allows users to adjust the linear weighting factor of Attentive Guidance (_i.e._, `attn_guidance_scale`) to synthesize the desired images. For example, as shown in the figure below, using a larger guidance scale may result in more details, richer colors, and stronger contrast.
![image](fig/ablation_guidance_scale.png)
* When using Attentive Guidance, it is necessary to delay its effect by 3 to 5 steps (in the case of 50-step sampling), which results in higher-quality images.
* In the second stage, AP-LDM divides the upsampling process into several sub-stages, where the number of sub-stages can be specified. Additionally, the number of diffusion-denoising steps in each sub-stage can also be specified. This is achieved by providing an initialization rate list (i.e., `init_rates`).

## Parameter
* gpu_ids: Determines which GPU to use. If set to `None`, the CPU will be used.
* 
