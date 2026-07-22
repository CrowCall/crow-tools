import csv
import os
from datetime import datetime

import requests

from crowtools.datasets import get_library_dir

PATH = os.path.dirname(__file__)

# -- CONFIG: Adjust as needed --
# Xeno-canto retired the keyless v2 API; v3 is the current endpoint and needs a
# free API key (https://xeno-canto.org/account). v3 also expects tag-based
# queries, so the species is given as gen:/sp: rather than a free-text name.
SPECIES_QUERY = "gen:Corvus sp:brachyrhynchos"
BASE_API_URL = "https://xeno-canto.org/api/3/recordings"
API_KEY_ENV_VAR = "XENO_CANTO_API_KEY"


def resolve_api_key(api_key=None):
    return api_key or os.environ.get(API_KEY_ENV_VAR)


def fetch_all_recordings(query, api_key):
    """
    Fetch all pages from the Xeno-Canto v3 API for the given query.
    Returns a list of recording dictionaries.
    """
    all_recs = []
    page_number = 1

    while True:
        resp = requests.get(
            BASE_API_URL,
            params={"query": query, "key": api_key, "page": page_number},
            timeout=30,
        )
        if resp.status_code != 200:
            detail = resp.text[:200].replace("\n", " ")
            print(f"Xeno-Canto API returned HTTP {resp.status_code}: {detail}")
            break

        data = resp.json()
        recs = data.get("recordings", [])
        if not recs:
            break

        all_recs.extend(recs)

        # Stop if we reached the last page
        if page_number >= int(data.get("numPages", 1)):
            break

        page_number += 1

    return all_recs


def recording_file_url(rec):
    """Pull the downloadable audio URL out of a v3 recording, fixing the scheme.

    v3 usually returns `file` as a URL string, but has been seen to nest it as
    {"url": ...}; handle both so a format tweak upstream doesn't break ingest.
    """
    file_url = rec.get("file")
    if isinstance(file_url, dict):
        file_url = file_url.get("url")
    if not file_url:
        return ""
    if file_url.startswith("//"):
        file_url = "https:" + file_url
    return file_url


def download_mp3(xc_id, mp3_url, output_dir):
    """
    Download MP3 file to local OUTPUT_DIR/{xc_id}.mp3
    """
    if not mp3_url:
        return

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


def start_downloads(percent=100, selected_ids=None, cache_base=None, api_key=None):
    api_key = resolve_api_key(api_key)
    if not api_key:
        print(
            "No Xeno-Canto API key found. The v3 API requires a free key: create "
            "one at https://xeno-canto.org/account and set the "
            f"{API_KEY_ENV_VAR} environment variable. Skipping Xeno-Canto."
        )
        return

    library_base = get_library_dir("xeno-canto", cache_base)
    output_csv = os.path.join(library_base, "library.csv")
    output_dir = os.path.join(library_base, "audio")
    os.makedirs(output_dir, exist_ok=True)

    # Fetch all recordings
    recordings = fetch_all_recordings(SPECIES_QUERY, api_key)
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

    if not recordings:
        print(f"No Xeno-Canto recordings resolved for '{SPECIES_QUERY}'. Preserving existing catalog at '{output_csv}'.")
        return

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
            download_mp3(xc_id, recording_file_url(rec), output_dir)

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
