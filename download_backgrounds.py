import csv
import os
import requests
import random
random.seed(42)

# Background file names
background_filenames = [
    "black-capped-chickadee.csv",
    "blue-jay.csv",
    "downy-woodpecker.csv",
    "european-starling.csv",
    "northern-cardinal.csv",
    "northern-mocking.csv",
    "robin.csv"
]

# Limit downloads per file
max_downloads = 50
download_count = 0
skipped_count = 0

for background_filename in background_filenames:
    print(f"Downloading {background_filename} - {max_downloads} downloads")
    csv_path = os.path.join(".cache/csv/", background_filename)
    library_dir = ".cache/backgrounds/"
    os.makedirs(library_dir, exist_ok=True)
    background_count = 0
    age_sexes = {}

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)  # Load all rows into a list
        random.shuffle(rows)  # Shuffle the list in-place

    for row in rows:
            # Now this should read the correct field name without BOM
            catalog_number = row["ML Catalog Number"]
            age_sex = row["Age/Sex"].lower()
            rating = float(row["Average Community Rating"])
            lat = float(row["Latitude"] or "0")
            long = float(row["Longitude"] or "0")

            download_url = f"https://cdn.download.ams.birds.cornell.edu/api/v2/asset/{catalog_number}/mp3"

            filename = f"{catalog_number}.mp3"
            filepath = os.path.join(library_dir, filename)

            if age_sex not in age_sexes:
                age_sexes[age_sex] = 1
            else:
                age_sexes[age_sex] += 1

            # Skip download if the file already exists
            if os.path.exists(filepath):
                print(f"File already exists, skipping: {filepath}")
                skipped_count += 1

            if "juv" in age_sex:
                skipped_count += 1
                continue

            if rating < 4.0:
                skipped_count += 1
                continue

            download_count += 1
            background_count += 1

            print(f"Downloading {download_url} to {filepath}")
            if not os.path.exists(filepath):
                response = requests.get(download_url, stream=True)
                if response.status_code == 200:
                    with open(filepath, "wb") as out_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            out_file.write(chunk)
                else:
                    print(f"Failed to download {download_url}: status code {response.status_code}")

            if background_count >= 100:
                background_count = 0
                print(f"Done with {background_filename} downloads")
                break

print(f"Downloaded {download_count} records, Skipped {skipped_count} records")
