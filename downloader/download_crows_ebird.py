import csv
import os
import requests

from crowtools.datasets import get_library_dir, read_library_catalog_rows, select_catalog_rows

PATH = os.path.dirname(__file__)

def start_downloads(percent=100, selected_ids=None, cache_base=None):
    """Download audio from the Macaulay library."""
    library_base = get_library_dir("macaulay", cache_base)
    csv_path = os.path.join(library_base, "library.csv")
    library_dir = os.path.join(library_base, "audio")
    os.makedirs(library_dir, exist_ok=True)
    download_count = 0
    skipped_count = 0

    rows = read_library_catalog_rows("macaulay", cache_base)
    if selected_ids is not None:
        reader = select_catalog_rows(rows, selected_ids=selected_ids)
    else:
        reader = list(rows)
        if percent < 100:
            limit = int(len(reader) * (percent / 100.0))
            reader = reader[:max(1, limit)]

    for row in reader:
        catalog_number = row["ML Catalog Number"]
        download_url = f"https://cdn.download.ams.birds.cornell.edu/api/v2/asset/{catalog_number}/mp3"

        filename = f"{catalog_number}.mp3"
        filepath = os.path.join(library_dir, filename)

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

if __name__ == "__main__":
    start_downloads()
