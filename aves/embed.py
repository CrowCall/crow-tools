import numpy as np
import torch
from aves.ispa.features import FeatureBasedISPAPredictor
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
    Given an audio waveform and its sample rate, extract an embedding using the AVES feature extractor.

    Args:
        waveform (torch.Tensor or np.ndarray): Audio waveform.
            - If a torch.Tensor, expected shape is (batch, time) or (time,).
            - If a numpy array, conversion to a tensor will be attempted.
        sample_rate (int): The sample rate of the waveform.

    Returns:
        np.ndarray: A 1D array representing the mean embedding extracted from the waveform.
    """
    # Ensure the waveform is a torch.Tensor (add batch dimension if needed).
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    if waveform.dim() == 1:
        # Add a batch dimension.
        waveform = waveform.unsqueeze(0)

    # Extract features; expected shape is (batch, time, feature_dim).
    features = ispa_f_predictor.feature_extractor(waveform)
    # Remove the batch dimension.
    features = features.squeeze(0)  # now (time, feature_dim)

    # Convert to NumPy.
    features_np = features.detach().numpy()
    # Compute the mean embedding over the time axis.
    embedding = np.mean(features_np, axis=0)
    return embedding


# Example usage:
if __name__ == "__main__":
    # For demonstration, create a dummy waveform (e.g. 3 seconds of audio at 8000 Hz).
    sample_rate = 8000
    duration_seconds = 3
    # Generate a random waveform as an example (normally you'd load real audio).
    dummy_waveform = torch.randn(sample_rate * duration_seconds)

    embedding = generate_embeddings(dummy_waveform)
    print("Generated embedding shape:", embedding.shape)
