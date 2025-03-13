import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
filepath = "labeler-vue/public/mixes/mix-dataset.json"
with open(filepath, 'r') as f:
    dataset = json.load(f)

# Flatten all segments into a single list of records
records = []
for item in dataset:
    segments = item.get("segments", [])
    records.extend(segments)

# Create a DataFrame from the records
df = pd.DataFrame(records)

# Compute the frequency distributions
file_id_counts = df['file_id'].value_counts().reset_index()
file_id_counts.columns = ['file_id', 'count']

original_key_counts = df['original_key'].value_counts().reset_index()
original_key_counts.columns = ['original_key', 'count']

# Display the distributions as tables
print("Distribution of file_id:")
print(file_id_counts)
print("\nDistribution of original_key:")
print(original_key_counts)

# Plot the distribution of file_id
plt.figure(figsize=(12, 6))
plt.bar(file_id_counts['file_id'], file_id_counts['count'])
plt.xlabel('file_id')
plt.ylabel('Count')
plt.title('Distribution of file_id')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Plot the distribution of original_key
plt.figure(figsize=(12, 6))
plt.bar(original_key_counts['original_key'], original_key_counts['count'])
plt.xlabel('original_key')
plt.ylabel('Count')
plt.title('Distribution of original_key')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
