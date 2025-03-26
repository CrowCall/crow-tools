import os
import json

# Set up paths.
PATH = os.path.dirname(__file__)
OLD_LABELS_PATHS = [os.path.join(PATH, "..", ".cache", "labels.json"),
                    os.path.join(PATH, "..", ".cache", "auto_labels.json")]
OUTPUT_PATH = os.path.join(PATH, "..", ".cache", "migrated_labels.json")

# Prepare a new dictionary.
new_labels = {}

for OLD_LABELS_PATH in OLD_LABELS_PATHS:
    # Load old labels.
    with open(OLD_LABELS_PATH, "r") as f:
        old_labels = json.load(f)

    for key, old in old_labels.items():
        # Check if any of the hard-to-find flags is True.
        file_id, start, end = key.split("-")
        segment_length = int(end) - int(start)
        reviewed = old.get("reviewed", False)
        if not reviewed and "auto_labels" in OLD_LABELS_PATH:
            continue
        if not (old.get("rattle", False) or old.get("softSong", False) or old.get("human", False) or old.get("mob", False) or old.get("begging", False) or old.get("quality", 2) in [1,3]):
            continue
        if not segment_length == 3.0:
            continue

        # Map crowCount: assume "multiple" becomes 4, else 1.
        if not reviewed:
            crow_count = 2 if old.get("crowCount", "").lower() == "multiple" else 1
            crow_age = 1 if old.get("crowAge", "").lower() == "adult" else 2
            quality = 1 if old.get("human", False) else 2
            if old.get("rattle", False) or old.get("softSong", False):
                quality = 2 # force rattle and softSong to average quality
        else:
            crow_count = old.get("crowCount", 1)
            crow_age = old.get("crowAge", 1)
            quality = old.get("quality", 2)

        # Determine cluster based on flag priority: human > rattle > softSong.
        if old.get("human", False):
            cluster = 100
        elif old.get("rattle", False):
            cluster = 200
        elif old.get("softSong", False):
            cluster = 300
        elif old.get("begging", False):
            cluster = 400
        elif old.get("mob", False):
            cluster = 500
        else:
            cluster = 0  # Fallback; should not occur given our filter.

        # Build new label dictionary.
        new_label = {
            "crowCount": crow_count,
            "crowAge": crow_age,
            "alert": False,
            "begging": old.get("begging", False),
            "grief": False,  # No corresponding old field; defaulting to False.
            "softSong": old.get("softSong", False),
            "rattle": old.get("rattle", False),
            "mob": False,  # No corresponding old field; defaulting to False.
            "quality": quality,
            "reviewed": True,
            "cluster": cluster
        }

        new_labels[key] = new_label

# Write the new labels to the output file.
with open(OUTPUT_PATH, "w") as f:
    json.dump(new_labels, f, indent=4)

print(f"Migrated {len(new_labels)} labels to {OUTPUT_PATH}")
