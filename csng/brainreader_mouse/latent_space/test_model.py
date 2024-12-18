import os

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def latents_to_pil(pipe, latents):
    # reverse scaling
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        # decode sample
        image = pipe.vae.decode(latents).sample

    # normalize, shift from -1 and 1 to 0 to 1 and clamp values in that range
    image = (image / 2 + 0.5).clamp(0, 1)

    # permute used to convert tensor to image, with channels as last dimension
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

    # convert to format expected by PIL images (scale to 255, round, convert to unsigned 8 bit integers)
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def save_reconstructed_images(images, output_path):
    """Saves the tensor of reconstructed images to a file."""
    images_output = os.path.join(output_path, "images")
    os.makedirs(images_output, exist_ok=True)
    for i, img in enumerate(images):
        img_path = os.path.join(images_output, f"image_{i:04d}.png")
        img.save(img_path)

    torch.save(images, os.path.join(output_path, "recons.pt"))
    print(f"Saved reconstructed images to {output_path}")


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_path = "results/fully_connected/2024-12-14-16-56-50"
    file_name = "predictions.npy"

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
    )
    pipe = pipe.to(DEVICE)

    latents = np.load(os.path.join(results_path, file_name))
    latents = torch.from_numpy(latents).to(DEVICE, dtype=torch.float16)

    reconstructed_images = latents_to_pil(pipe, latents)
    save_reconstructed_images(reconstructed_images, results_path)


if __name__ == "__main__":
    main()
