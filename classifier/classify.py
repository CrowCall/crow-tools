import os
import torch
import numpy as np
from classifier.model import CrowClassifier

# Set up paths.
PATH = os.path.dirname(__file__)
checkpoint_path = os.path.join(PATH, "models", "best_model.ckpt")

# Determine device.
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model only once.
if os.path.exists(checkpoint_path):
    _model = CrowClassifier.load_from_checkpoint(checkpoint_path)
else:
    _model = CrowClassifier()
_model.to(device)
_model.eval()


def predict_embedding(embedding, device=device):
    """
    Given an embedding (numpy array of shape [768]), perform inference using the
    pre-loaded model and return a predicted label dictionary in the new format:

    {
      "crowCount": int in [1,2,3,4],
      "crowAge": int in [1,2],
      "alert": bool,
      "begging": bool,
      "softSong": bool,
      "rattle": bool,
      "mob": bool,
      "quality": int in [1,2,3],
    }
    """
    # Ensure the embedding is a float32 numpy array and add a batch dimension.
    embedding_tensor = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0).to(device)

    # Perform inference.
    with torch.no_grad():
        outputs = _model(embedding_tensor)

    new_pred = {}
    new_pred["crowCount"] = torch.argmax(outputs["crowCount"], dim=1).item() + 1
    new_pred["crowAge"] = torch.argmax(outputs["crowAge"], dim=1).item() + 1
    new_pred["alert"] = (outputs["alert"].squeeze() > 0).item()
    new_pred["begging"] = (outputs["begging"].squeeze() > 0).item()
    new_pred["softSong"] = (outputs["softSong"].squeeze() > 0).item()
    new_pred["rattle"] = (outputs["rattle"].squeeze() > 0).item()
    new_pred["mob"] = (outputs["mob"].squeeze() > 0).item()
    new_pred["quality"] = torch.argmax(outputs["quality"], dim=1).item() + 1

    return new_pred


if __name__ == "__main__":
    # Example: Replace with your actual numpy array of 768 numbers.
    dummy_embedding = np.random.randn(768)
    predicted_label = predict_embedding(dummy_embedding)
    print("Predicted Label:", predicted_label)
