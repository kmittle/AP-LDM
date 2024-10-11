# AP-LDM
Official implementation of [***AP-LDM: Attentive and Progressive Latent Diffusion Model for Training-Free High-Resolution Image Generation***](https://arxiv.org/abs/2410.06055v1).

![image](fig/teaser.png) 

**AP-LDM** is a framework that enables the rapid synthesis of high-quality, high-resolution images without the need for retraining.

It consists of two stages: (1) synthesizing high-quality images at the training resolution using **Attentive Guidance**, and (2) generating finer high-resolution images through pixel upsampling and "diffusion-denoising."

## Overview of AP-LDM
![image](fig/AP-LDM.png) 
* Attentive Guidance enhances the structural consistency of the latent representation using a parameter-free self-attention mechanism, which is achieved through linear weighting.
* It allows users to adjust the linear weighting factor of Attentive Guidance (_i.e._, the guidance scale) to synthesize the desired images. For example, as shown in the figure below, using a larger guidance scale may result in more details, richer colors, and stronger contrast.
