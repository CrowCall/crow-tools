import numpy as np
import torch
from embedder.ispa.features import FeatureBasedISPAPredictor
import os

PATH = os.path.dirname(__file__)

# Initialize the AVES feature predictor.
ispa_f_predictor = FeatureBasedISPAPredictor(
    feature_type='aves',
    kmeans_model=os.path.join(PATH, 'ispa', 'models', 'kmeans.aves.pkl'),
    phoneme_map=os.path.join(PATH, 'ispa', 'models', 'c2p.aves.json'),
    aves_config_path=os.path.join(PATH, 'ispa', 'models', 'aves-base-bio.torchaudio.model_config.json'),
    aves_model_path=os.path.join(PATH, 'ispa', 'models', 'aves-base-bio.torchaudio.pt')
)

def generate_embeddings(waveform):
    """
    Given an audio waveform and its sample rate, extract an embedding using the AVES feature extractor

    Args:
        waveform (torch.Tensor or np.ndarray): Audio waveform.
            - If a torch.Tensor, expected shape is (batch, time) or (time,).
            - If a numpy array, conversion to a tensor will be attempted.
        sample_rate (int): The sample rate of the waveform.

    Returns:
        np.ndarray: A 2D array where each row corresponds to the feature embeddings (25 embeddings per second)
    """
    # Ensure the waveform is a torch.Tensor (add batch dimension if needed).
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Remove the batch dimension
    waveform = waveform.squeeze(0)

    # Calculate the number of samples per chunk (120 seconds).
    chunk_size = 8000 * 120
    embeddings_chunks = []

    # Process the waveform in chunks (so large files don't crash)
    num_samples = waveform.shape[0]
    for start in range(0, num_samples, chunk_size):
        # Take a chunk (if the final chunk is shorter, that's fine)
        chunk = waveform[start:start + chunk_size]
        # Add batch dimension back for feature extraction.
        chunk = chunk.unsqueeze(0)  # shape (1, chunk_time)
        # Extract features; expected output shape: (1, time, feature_dim)
        features = ispa_f_predictor.feature_extractor(chunk)
        # Remove batch dimension.
        features = features.squeeze(0)  # shape: (time, feature_dim)
        # Convert to numpy.
        features_np = features.detach().numpy()
        embeddings_chunks.append(features_np)

    # Reassemble embeddings by concatenating along the time axis.
    full_embedding = np.concatenate(embeddings_chunks, axis=0)
    return full_embedding

# Example usage:
if __name__ == "__main__":
    # For demonstration, create a dummy waveform
    sample_rate = 8000
    duration_seconds = 10
    # Generate a random waveform as an example (normally you'd load real audio).
    dummy_waveform = torch.randn(sample_rate * duration_seconds)

    embedding = generate_embeddings(dummy_waveform)
    print("Generated embedding shape:", embedding.shape)
