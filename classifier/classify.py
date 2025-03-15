import torch
import numpy as np
from model import CrowClassifier


def predict_embedding(embedding, checkpoint_path="logs/checkpoints/best_model.ckpt", device="cpu"):
    """
    Given an embedding (numpy array of shape [768]), load the model,
    perform inference, and return a predicted label dictionary.
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

    # Process outputs:
    # For multi-class outputs, we'll map indices to labels.
    crowCount_classes = ["single", "multiple"]
    crowAge_classes = ["adult", "juvenile"]

    pred = {}
    pred["crowCount"] = crowCount_classes[torch.argmax(outputs["crowCount"], dim=1).item()]
    pred["crowAge"] = crowAge_classes[torch.argmax(outputs["crowAge"], dim=1).item()]

    # For binary outputs, threshold logits at zero.
    pred["begging"] = (outputs["begging"].squeeze() > 0).item()
    pred["softSong"] = (outputs["softSong"].squeeze() > 0).item()
    pred["rattle"] = (outputs["rattle"].squeeze() > 0).item()
    pred["badQuality"] = (outputs["badQuality"].squeeze() > 0).item()
    pred["human"] = (outputs["human"].squeeze() > 0).item()

    return pred


if __name__ == "__main__":
    # Example: Replace with your actual numpy array of 768 numbers.
    dummy_embedding = np.random.randn(768)

    # Choose device: "cuda" if GPU available or "cpu".
    device = "cuda" if torch.cuda.is_available() else "cpu"

    predicted_label = predict_embedding(dummy_embedding, device=device)
    print("Predicted Label:", predicted_label)
