import requests
import csv
import os
from datetime import datetime

# -- CONFIG: Adjust as needed --
SPECIES_QUERY = "American+Crow"
OUTPUT_CSV = os.path.join(".cache", "csv", "crows-xeno-canto.csv")
OUTPUT_DIR = os.path.join(".cache", "library")
BASE_API_URL = "https://www.xeno-canto.org/api/2/recordings"

# Create output folder if not existing
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def download_mp3(xc_id, mp3_url):
    """
    Download MP3 file to local OUTPUT_DIR/{xc_id}.mp3
    """
    if mp3_url.startswith("//"):
        mp3_url = "https:" + mp3_url  # Fix if it starts with //

    filename = f"{xc_id}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(filepath):
        try:
            print(f"Downloading {mp3_url}")
            response = requests.get(mp3_url, timeout=30)
            with open(filepath, 'wb') as f_out:
                f_out.write(response.content)
        except Exception as e:
            print(f"Error downloading {mp3_url}: {e}")

def main():
    # 1) Fetch all recordings from Xeno-Canto
    recordings = fetch_all_recordings(SPECIES_QUERY)
    print(f"Found {len(recordings)} recordings for '{SPECIES_QUERY}'.")

    # 2) Prepare CSV
    #    We'll keep just a few columns.
    #    - "ML Catalog Number" => Xeno-Canto "id"
    #    - "Date" => rec["date"]
    #    - "Latitude" => rec["lat"]
    #    - "Longitude" => rec["lng"]
    #    - "Recordist" => rec["rec"]
    #    - "Media notes" => rec["rmk"]
    #    - "Age/Sex" => placeholder "unknown"
    #    - "Average Community Rating" => placeholder "0.0"
    #    - "Filename" => e.g. 543339.mp3

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

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
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
                download_mp3(xc_id, mp3_url)

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

    print(f"Done! CSV written to '{OUTPUT_CSV}'. MP3s downloaded to '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()
