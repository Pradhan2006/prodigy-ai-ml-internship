# Task 02 — Image Generation with Pre-trained Models

## Objective

The objective of this task is to generate images from text prompts using a pre-trained Stable Diffusion model.

## Technology Used

- Python 3.13
- PyTorch
- Hugging Face Diffusers
- Hugging Face Transformers
- Accelerate
- Safetensors
- Stable Diffusion v1.5

## Project Description

This project demonstrates text-to-image generation using a pre-trained Stable Diffusion model.

The user enters a text prompt through the terminal. The program loads the pre-trained Stable Diffusion model and generates an image based on the provided prompt.

The generated image is saved locally inside the `outputs` directory.

## Project Structure

```text
Task-02-Stable-Diffusion/
│
├── generate_image.py
├── requirements.txt
├── README.md
│
└── outputs/
    ├── generated_image.png
    └── generated images
