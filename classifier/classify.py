import torch
import numpy as np
from classifier.model import CrowClassifier
import os

PATH = os.path.dirname(__file__)
checkpoint_path = os.path.join(PATH, "models", "best_model.ckpt")

def predict_embedding(embedding, device="cpu"):
    """
    Given an embedding (numpy array of shape [768]), load the model,
    perform inference, and return a predicted label dictionary in the new format:

    {
      "crowCount": int in [1,2,3,4],   # predicted as argmax+1 from the new model
      "crowAge": int in [1,2],         # predicted as argmax+1 from the new model
      "alert": bool,                   # default False (new field)
      "begging": bool,                 # from model binary output
      "softSong": bool,                # from model binary output
      "rattle": bool,                  # from model binary output
      "mob": bool,                     # from model binary output
      "quality": int in [1,2,3],       # predicted as argmax+1 from the new model
    }
    """
    # Ensure the embedding is a float32 numpy array and add a batch dimension.
    embedding_tensor = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0).to(device)

    # Load the model from checkpoint if available.
    if checkpoint_path:
        model = CrowClassifier.load_from_checkpoint(checkpoint_path)
    else:
        model = CrowClassifier()
    model.to(device)
    model.eval()

    # Perform inference.
    with torch.no_grad():
        outputs = model(embedding_tensor)

    new_pred = {}

    # CrowCount: use argmax and add 1.
    new_pred["crowCount"] = torch.argmax(outputs["crowCount"], dim=1).item() + 1

    # CrowAge: use argmax and add 1.
    new_pred["crowAge"] = torch.argmax(outputs["crowAge"], dim=1).item() + 1

    # For binary outputs, threshold at 0.
    new_pred["alert"] = (outputs["alert"].squeeze() > 0).item()
    new_pred["begging"] = (outputs["begging"].squeeze() > 0).item()
    new_pred["softSong"] = (outputs["softSong"].squeeze() > 0).item()
    new_pred["rattle"] = (outputs["rattle"].squeeze() > 0).item()
    new_pred["mob"] = (outputs["mob"].squeeze() > 0).item()

    # Quality: use argmax and add 1.
    new_pred["quality"] = torch.argmax(outputs["quality"], dim=1).item() + 1

    return new_pred


if __name__ == "__main__":
    # Example: Replace with your actual numpy array of 768 numbers.
    dummy_embedding = np.random.randn(768)

    # Choose device: "cuda" if GPU is available, otherwise "cpu".
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predicted_label = predict_embedding(dummy_embedding, device=device)
    print("Predicted Label:", predicted_label)
