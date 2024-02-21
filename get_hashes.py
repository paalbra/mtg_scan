import datetime
import io
import json

import imagehash
import PIL
import requests
import sys

if len(sys.argv) != 2:
    print("ERROR: Need path to card json file. https://scryfall.com/docs/api/bulk-data .")
    sys.exit(1)

print("Loading file...")
with open(sys.argv[1]) as f:
    cards = json.loads(f.read())

start_time = datetime.datetime.now()

print("Downloading and hashing images...")
with open("hashes.txt", "w") as f:
    for index, card in enumerate(cards, start=1):
        if index % 1000 == 0:
            seconds_per_image = (datetime.datetime.now() - start_time).total_seconds() / index
            seconds_left = (len(cards) - index) * seconds_per_image

            now = datetime.datetime.now()
            delta = str(now - start_time).split(".")[0]  # Remove microseconds
            eta = (now + datetime.timedelta(seconds=seconds_left)).isoformat(timespec="seconds")
            now = now.isoformat(timespec="seconds")

            print(f"Status {now}: Passed {delta} (ETA: {eta}) {index} of {len(cards)} cards.")

        try:
            uri = card["image_uris"]["normal"]
            response = requests.get(uri)
            response.raise_for_status()
            image_hash = imagehash.phash(PIL.Image.open(io.BytesIO(response.content)))
        except Exception as e:
            print("ERROR", card["name"], str(e))
            continue

        f.write(f"{image_hash} {card['name']}\n")

print("Done")
