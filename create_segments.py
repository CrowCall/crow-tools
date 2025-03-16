from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from datetime import datetime
import csv
import json
import os

csv_path = "labeler-vue/public/csv/crows.csv"
library_dir = "labeler-vue/public/library"
segments_path = "labeler-vue/public/segments.json"
segments = {}

if os.path.exists(segments_path):
    with open(segments_path) as json_file:
        segments = json.load(json_file)

with open(csv_path, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        catalog_number = row["ML Catalog Number"]
        age_sex = row["Age/Sex"].lower()
        rating = float(row["Average Community Rating"])
        lat = float(row["Latitude"] or "0")
        long = float(row["Longitude"] or "0")
        if row["Date"]:
            parsed_date = datetime.strptime(row["Date"], "%Y-%m-%d")
        else:
            print("Invalid date")
            continue

        filename = f"{catalog_number}.mp3"
        filepath = os.path.join(library_dir, filename)

        if os.path.exists(filepath) and catalog_number not in segments:
            # Load and initialize the BirdNET-Analyzer models.
            analyzer = Analyzer()

            CONFIDENCE_THRESHOLD = 0.65
            recording = Recording(
             analyzer,
             filepath,
             lat=lat,
             lon=long,
             date=parsed_date,
             min_conf=CONFIDENCE_THRESHOLD,
            )
            recording.analyze()

            # Collect crow detections
            detections = []
            for d in recording.detection_list:
                if d.confidence > CONFIDENCE_THRESHOLD and d.scientific_name == "Corvus brachyrhynchos":
                    detections.append({ "common_name": d.common_name, "scientific_name": d.scientific_name,
                                         "start_time": d.start_time, "end_time": d.end_time, "confidence": d.confidence })

            # {"common_name": "American Crow", "scientific_name": "Corvus brachyrhynchos", "start_time": 15.0, "end_time": 30.0, "confidence": 0.7989989399909974}
            if detections:
                print(f">>>>> Found {len(detections)} detections in {filename}")
                segments[catalog_number] = detections
            else:
                print(f"<<<<<< No detections in {filename}")

            if len(segments) % 10 == 0:
                # Write output to JSON file
                print(f"***** Saved Segments for Files: {len(segments)}")
                open("labeler-vue/public/segments.json", "w").write(json.dumps(segments))

# Write output to JSON file
print(f"***** Saved Segments for Files: {len(segments)}")
open("labeler-vue/public/segments.json", "w").write(json.dumps(segments))