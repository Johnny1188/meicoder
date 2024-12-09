import argparse
import logging
import os
import pickle

import torch
from diffusers import StableDiffusionPipeline
from utils import get_stim_transform

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

stim_transform = get_stim_transform(device="cuda")


def load_stimulus_data(file_path):
    """Load stimulus data from a pickle file."""
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data["stim"], data["resp"]
    except Exception as e:
        # logger.error(f"Error loading {file_path}: {e}")
        return None, None


def transform_image(image):
    """Apply the necessary transformations to an image."""
    return stim_transform(image)


def store_latent_vector(latent_vector, output_dir, file_name):
    """Store the latent vector in a pickle file with the same name as the stimulus."""
    # base_name = os.path.splitext(file_name)[0]  # Remove the extension
    # output_file_path = os.path.join(output_dir, f"latent_{base_name}.pkl")
    output_file_path = os.path.join(output_dir, file_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_file_path, "wb") as f:
            pickle.dump(
                latent_vector.squeeze(0).cpu().numpy(), f
            )  # Save the tensor as numpy array
        logger.info(
            f"Stored latent vector for {file_name} as {output_file_path}"
        )
    except Exception as e:
        logger.error(f"Error saving latent vector for {file_name}: {e}")


def get_latent_vector(image, vae):
    """
    Get the latent vector of the image using the encoder.

    Args:
        image (Tensor): The transformed image tensor.
        vae (Vae): The VAE to obtain the latent vector.

    Returns:
        Tensor: The corresponding latent vector.
    """
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist

    latent_vector = (
        latent_dist.mean
    )  # Take the mean of the latent distribution
    return latent_vector * 0.18215  # Apply the scaling factor


def reconstruct_image_from_latent(latent_vector, vae):
    """
    Decode the latent vector to get an image.

    Args:
        latent_vector (Tensor): The given latent vector.
        vae (Vae): The VAE to obtain the decoded image.

    Returns:
        Tensor: The corresponding decoded image.
    """
    with torch.no_grad():
        reconstructed_image_tensor = vae.decode(latent_vector).sample

    # Denormalize and clamp
    reconstructed_image_tensor = (reconstructed_image_tensor / 2 + 0.5).clamp(
        0, 1
    )
    return reconstructed_image_tensor * 255


def setup_model():
    """Set up the Stable Diffusion model pipeline and return the encoder."""
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    )
    pipe = pipe.to(device)

    encoder = pipe.vae
    return encoder


def process_stimuli_files(dataset_dir, output_dir, encoder):
    """Process each stimulus file, compute the latent vector, and store it."""
    stimulus_files = [
        f for f in os.listdir(dataset_dir) if f.endswith((".pkl", ".pickle"))
    ]

    for stimulus_file in stimulus_files:
        stimulus_path = os.path.join(dataset_dir, stimulus_file)

        # Load stimulus data
        stimulus_image, _ = load_stimulus_data(stimulus_path)
        if stimulus_image is None:
            continue  # Skip files that couldn't be loaded

        # Transform image
        transformed_image = transform_image(stimulus_image)

        # Get the latent vector
        latent_vector = get_latent_vector(transformed_image, encoder)

        # Store the latent vector
        store_latent_vector(latent_vector, output_dir, stimulus_file)


def main():
    parser = argparse.ArgumentParser(
        description="Process stimulus data and compute latent vectors."
    )
    parser.add_argument(
        "--dataset",
        choices=["train", "val", "test"],
        default="train",
        required=True,
        help="Specify the dataset to process (train, valid, or test).",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(
        os.environ["DATA_PATH"], "brainreader", "data", "1", args.dataset
    )
    output_dir = os.path.join(
        os.environ["DATA_PATH"],
        "brainreader",
        "latent_vectors",
        args.dataset,
    )

    # Set up the model and encoder
    encoder = setup_model()

    # Process and store latent vectors
    process_stimuli_files(dataset_dir, output_dir, encoder)


if __name__ == "__main__":
    main()
