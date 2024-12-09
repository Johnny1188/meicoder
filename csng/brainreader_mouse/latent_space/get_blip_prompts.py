import argparse
import logging
import os
import pickle

import torch
import torchvision
from diffusers import StableDiffusionPipeline
from transformers import BlipForConditionalGeneration, BlipProcessor
from utils import get_stim_transform

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

device = "cuda"

blip_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((512, 512)),
    ]
)


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
    return blip_transform(image)


def store_caption(caption, output_dir, file_name):
    """Store the latent vector in a pickle file with the same name as the stimulus."""
    # base_name = os.path.splitext(file_name)[0]  # Remove the extension
    # output_file_path = os.path.join(output_dir, f"latent_{base_name}.pkl")
    output_file_path = os.path.join(output_dir, file_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        with open(output_file_path, "wb") as f:
            pickle.dump(caption, f)  # Save the tensor as numpy array
        logger.info(
            f"Stored latent vector for {file_name} as {output_file_path}"
        )
    except Exception as e:
        logger.error(f"Error saving latent vector for {file_name}: {e}")


def get_caption(image, processor, model):
    with torch.no_grad():
        blip_preprocessed = processor(images=image, return_tensors="pt").to(
            device
        )
        caption_ids = model.generate(**blip_preprocessed, max_length=10)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    return caption


def setup_model():
    """Set up the Stable Diffusion model pipeline and return the encoder."""
    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    return processor, model


def process_stimuli_files(dataset_dir, output_dir, processor, model):
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
        caption = get_caption(transformed_image, processor, model)

        # Store the latent vector
        store_caption(caption, output_dir, stimulus_file)


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
        "captions",
        args.dataset,
    )

    # Set up the model and encoder
    processor, model = setup_model()

    # Process and store latent vectors
    process_stimuli_files(dataset_dir, output_dir, processor, model)


if __name__ == "__main__":
    main()
