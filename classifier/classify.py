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
      "crowCount": int in [1,2,3,4],   # here, 1 = single, 2 = multiple (no info for 3 or 4 from old model)
      "crowAge": int in [1,2],          # 1 = adult, 2 = juvenile
      "alert": bool,                   # default False (new field)
      "begging": bool,                 # food related calls
      "grief": bool,                   # default False (new field)
      "softSong": bool,                # from model
      "rattle": bool,                  # from model
      "mob": bool,                     # default False (new field)
      "quality": int in [1,2,3],        # 1 = bad/low (if badQuality or human true), else default 2
      "reviewed": bool                 # default False
    }
    """
    # Ensure the embedding is a float32 numpy array and add batch dimension.
    embedding_tensor = torch.from_numpy(embedding.astype(np.float32)).unsqueeze(0).to(device)

    # Load the model: if a checkpoint is provided, load from checkpoint.
    if checkpoint_path:
        model = CrowClassifier.load_from_checkpoint(checkpoint_path)
    else:
        model = CrowClassifier()
    model.to(device)
    model.eval()

    # Inference: disable gradients.
    with torch.no_grad():
        outputs = model(embedding_tensor)

    # Old model mappings
    crowCount_classes = ["single", "multiple"]
    crowAge_classes = ["adult", "juvenile"]

    new_pred = {}

    # Map crowCount: old "single" -> 1, "multiple" -> 2.
    old_crowCount = crowCount_classes[torch.argmax(outputs["crowCount"], dim=1).item()]
    new_pred["crowCount"] = 1 if old_crowCount == "single" else 2

    # Map crowAge: old "adult" -> 1, "juvenile" -> 2.
    old_crowAge = crowAge_classes[torch.argmax(outputs["crowAge"], dim=1).item()]
    new_pred["crowAge"] = 1 if old_crowAge == "adult" else 2

    # New fields that aren't in the old model
    new_pred["alert"] = False
    new_pred["begging"] = (outputs["begging"].squeeze() > 0).item()
    new_pred["grief"] = False

    # Copy binary outputs for softSong and rattle.
    new_pred["softSong"] = (outputs["softSong"].squeeze() > 0).item()
    new_pred["rattle"] = (outputs["rattle"].squeeze() > 0).item()

    # New field: mob default.
    new_pred["mob"] = False

    # For quality, use the old "badQuality" and "human" outputs:
    # if either is true, quality = 1, else default to 2.
    badQuality = (outputs["badQuality"].squeeze() > 0).item()
    human = (outputs["human"].squeeze() > 0).item()
    new_pred["quality"] = 1 if (badQuality or human) else 2

    # Additional new fields.
    new_pred["reviewed"] = False

    return new_pred


if __name__ == "__main__":
    # Example: Replace with your actual numpy array of 768 numbers.
    dummy_embedding = np.random.randn(768)

    # Choose device: "cuda" if GPU available or "cpu".
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predicted_label = predict_embedding(dummy_embedding, device=device)
    print("Predicted Label:", predicted_label)
