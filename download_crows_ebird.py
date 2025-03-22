import csv
import os
import requests

csv_path = os.path.join(".cache", "csv", "crows.csv")
library_dir = os.path.join(".cache", "library")
os.makedirs(library_dir, exist_ok=True)
download_count = 0
skipped_count = 0

with open(csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        # Now this should read the correct field name without BOM
        catalog_number = row["ML Catalog Number"]
        age_sex = row["Age/Sex"].lower()
        rating = float(row["Average Community Rating"])
        lat = float(row["Latitude"] or "0")
        long = float(row["Longitude"] or "0")

        download_url = f"https://cdn.download.ams.birds.cornell.edu/api/v2/asset/{catalog_number}/mp3"
        
        filename = f"{catalog_number}.mp3"
        filepath = os.path.join(library_dir, filename)

        # Skip download if the file already exists
        if os.path.exists(filepath):
            print(f"File already exists, skipping: {filepath}")
            skipped_count += 1
            continue

        download_count += 1
        print(f"Downloading {download_url} to {filepath}")
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            with open(filepath, "wb") as out_file:
                for chunk in response.iter_content(chunk_size=8192):
                    out_file.write(chunk)
        else:
            print(f"Failed to download {download_url}: status code {response.status_code}")

print(f"Downloaded {download_count} records, Skipped {skipped_count} records")
