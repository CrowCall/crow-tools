import requests
import csv
import os
from datetime import datetime

from crowtools.datasets import get_library_dir

PATH = os.path.dirname(__file__)

# -- CONFIG: Adjust as needed --
SPECIES_QUERY = "American+Crow"
BASE_API_URL = "https://www.xeno-canto.org/api/2/recordings"

def fetch_all_recordings(query):
    """
    Fetch all pages from Xeno-Canto API for the given query (species).
    Returns a list of recording dictionaries.
    """
    all_recs = []
    page_number = 1

    while True:
        url = f"{BASE_API_URL}?query={query}&page={page_number}"
        resp = requests.get(url).json()
        recs = resp.get("recordings", [])
        if not recs:
            break

        all_recs.extend(recs)

        # Stop if we reached the last page
        if page_number >= int(resp["numPages"]):
            break

        page_number += 1

    return all_recs

def download_mp3(xc_id, mp3_url, output_dir):
    """
    Download MP3 file to local OUTPUT_DIR/{xc_id}.mp3
    """
    if mp3_url.startswith("//"):
        mp3_url = "https:" + mp3_url  # Fix if it starts with //

    filename = f"{xc_id}.mp3"
    filepath = os.path.join(output_dir, filename)

    if not os.path.exists(filepath):
        try:
            print(f"Downloading {mp3_url}")
            response = requests.get(mp3_url, timeout=30)
            with open(filepath, 'wb') as f_out:
                f_out.write(response.content)
        except Exception as e:
            print(f"Error downloading {mp3_url}: {e}")

def start_downloads(percent=100, selected_ids=None, cache_base=None):
    library_base = get_library_dir("xeno-canto", cache_base)
    output_csv = os.path.join(library_base, "library.csv")
    output_dir = os.path.join(library_base, "audio")
    os.makedirs(output_dir, exist_ok=True)

    # Fetch all recordings
    recordings = fetch_all_recordings(SPECIES_QUERY)
    print(f"Found {len(recordings)} recordings for '{SPECIES_QUERY}'.")
    if selected_ids is not None:
        selected_lookup = {str(value) for value in selected_ids}
        recordings = [rec for rec in recordings if str(rec.get("id", "")) in selected_lookup]
    elif percent < 100:
        limit = int(len(recordings) * (percent / 100.0))
        recordings = recordings[:max(1, limit)]

    fieldnames = [
        "ML Catalog Number",
        "Date",
        "Latitude",
        "Longitude",
        "Recordist",
        "Media notes",
        "Age/Sex",
        "Average Community Rating",
        "Filename"
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for rec in recordings:
            xc_id = rec.get("id", "")
            if not xc_id:
                continue  # skip if no id

            # Parse date if it's valid
            date_str = rec.get("date", "")
            try:
                _ = datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                # If the date is missing or invalid, skip
                print(f"Skipping ID={xc_id}, invalid date '{date_str}'")
                continue

            # Convert lat/lng to floats (default to 0 if missing)
            lat_str = rec.get("lat", "0")
            lng_str = rec.get("lng", "0")
            try:
                lat = float(lat_str)
            except:
                lat = 0.0
            try:
                lng = float(lng_str)
            except:
                lng = 0.0

            # Age/Sex and rating not provided by Xeno-Canto
            age_sex = "unknown"
            rating = "0.0"

            # Download the MP3
            mp3_url = rec.get("file", "")
            if mp3_url:
                download_mp3(xc_id, mp3_url, output_dir)

            # Write CSV row
            row = {
                "ML Catalog Number": xc_id,
                "Date": date_str,
                "Latitude": str(lat),
                "Longitude": str(lng),
                "Recordist": rec.get("rec", "") or "Unknown",
                "Media notes": rec.get("rmk", ""),
                "Age/Sex": age_sex,
                "Average Community Rating": rating,
                "Filename": f"{xc_id}.mp3"
            }
            writer.writerow(row)

    print(f"Done! CSV written to '{output_csv}'. MP3s downloaded to '{output_dir}/'.")

if __name__ == "__main__":
    start_downloads()
