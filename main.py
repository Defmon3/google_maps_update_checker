# !/usr/bin/env python3
"""
Module Documentation: Image Fetcher and Comparer

This script is designed to automate the fetching, storing, and comparison of geographic images from the Google Maps API,
based on latitude and longitude coordinates. It utilizes OpenCV for image comparison,
requests for image fetching, and dotenv for environment variable management.
The script maintains a JSON file for image metadata, including coordinates, zoom level, and image filenames.

Ensure all dependencies are installed using pip.
Dependencies:
- OpenCV (cv2): For image processing and comparison.
- Requests: For fetching images from the Google Maps API.
- Python-dotenv: For loading the Google Maps API key from a .env file.
- Tabulate: For formatting the output of updates.

Requires:
Google maps static api key
https://developers.google.com/maps/documentation/maps-static/overview
Note:
Ensure your Google Maps API key has access to the Static Maps API and that the usage limits are sufficient for your needs.


Usage:
1. Run the script. It will create a images.json and a .env file if they don't exist.
2. Edit the .env file to include your Google Maps API key.

2. Edit images.json to include the locations you want to track.
3. Run the script again to fetch images for each location specified in images.json. The script compares new images with previously stored ones and updates the metadata accordingly.

Functions:
- format_filename(lat: str | float, long: str | float) -> str:
    Formats latitude and longitude into a filename-friendly string.

- save_image(filepath: Path, content: bytes) -> None:
    Saves image content to a file.

- download_image(lat: str | float, long: str | float, zoom: int = 17) -> bool:
    Fetches an image from the Google Maps API and saves it temporarily.

- images_are_a_match(stored_image: Path, temp_image: Path) -> bool:
    Compares two images and determines if they are significantly different.

- load_metadata() -> List[Dict] | None:
    Loads image metadata from a JSON file.

- save_metadata(metadata: List[Dict]) -> None:
    Saves image metadata to a JSON file.

- update_images(metadata: List[Dict]) -> List[Dict]:
    Checks for new images, compares them with existing ones, and updates metadata accordingly.

- load_google_maps_api_key() -> None:
    Loads the Google Maps API key from an environment variable.

- main():
    Main function to orchestrate the script's workflow.

The script outputs a table summarizing the updates after each run, indicating whether images were unchanged, changed, or newly added.

author: https://github.com/Defmon3

"""
import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import requests
from dotenv import load_dotenv
from tabulate import tabulate

NEW_IMAGES = Path('new_images')
OLD_IMAGES = Path('old_images')
IMAGE_METADATA_FILE = 'images.json'
NEW_IMAGES.mkdir(exist_ok=True)
OLD_IMAGES.mkdir(exist_ok=True)

TEMPLATE_URL = "https://maps.googleapis.com/maps/api/staticmap?&zoom={zoom}&size=600x300&maptype=hybrid&markers=color:red%7Clabel:S%7C{lat},{long}&key={api_key}"
JSON_TEMPLATE = [{
    "lat": "0",
    "long": "0",
    "zoom": 10,
    "name": "Location 1",
}, {
    "lat": "0",
    "long": "0",
    "zoom": 10,
    "name": "Location 2"

}]


def format_filename(lat: str | float, long: str | float) -> str:
    """Format latitude and longitude into a filename-friendly string."""
    return f"{lat}_{long}".replace(".", "_").replace("-", "n") + ".jpg"


def save_image(filepath: Path, content: bytes) -> None:
    """Save image content to a file."""
    with open(filepath, 'wb') as image_file:
        image_file.write(content)


def download_image(lat: str | float, long: str | float, zoom: int = 17) -> True | False:
    """Fetch an image from a URL and return the path where it was saved."""

    url = TEMPLATE_URL.format(lat=lat, long=long, api_key=os.getenv("GOOGLE_MAPS_API_KEY"), zoom=zoom)
    response = requests.get(url)
    response.raise_for_status()
    new_filepath = NEW_IMAGES / format_filename(lat, long)
    save_image(new_filepath, response.content)
    return new_filepath


def images_are_a_match(old_image: Path, new_image: Path) -> bool:
    """Compare two images. Return True if they are different, False otherwise."""
    img1 = cv2.imread(str(old_image))
    img2 = cv2.imread(str(new_image))
    if img1 is None or img2 is None:
        return False  # Can't compare if one image didn't load properly
    difference = cv2.absdiff(img1, img2)
    return np.array_equal(difference, np.zeros(difference.shape))


def load_metadata() -> List[Dict] | None:
    """Load image metadata from a JSON file."""

    if Path(IMAGE_METADATA_FILE).exists():
        with open(IMAGE_METADATA_FILE) as file:
            return json.load(file)
    else:
        print("No images.json file found. Creating one with a template, enter your own data and re-run the script.")
        save_metadata(JSON_TEMPLATE)


def save_metadata(metadata: List[Dict]) -> None:
    """Save image metadata to a JSON file."""
    with open(IMAGE_METADATA_FILE, 'w') as file:
        json.dump(metadata, file, indent=4)


def update_images(metadata: List[Dict]):
    """Check for new images, compare with existing ones, update metadata accordingly."""
    updates = []
    if not metadata:
        return updates

    for entry in metadata:
        lat = entry["lat"]
        long = entry["long"]
        zoom = entry.get("zoom", 17)

        new_image = download_image(lat, long, zoom)
        old_image = OLD_IMAGES / format_filename(lat, long)
        entry["last_pulled"] = datetime.now().isoformat()
        if not old_image.exists() or not images_are_a_match(old_image, new_image):
            status = "New" if not old_image.exists() else "Changed"
            save_image(old_image, new_image.read_bytes())
            entry["last_changed"] = datetime.now().isoformat()
            updates.append({"Name": entry["name"], "Location": f"{lat}, {long}", "Status": status,
                            "old_image": old_image.absolute(), "new_image": new_image.absolute()})
        else:
            updates.append({"Name": entry["name"], "Location": f"{lat}, {long}", "Status": "Unchanged",
                            "old_image": old_image.absolute(), "new_image": new_image.absolute()})
    return updates


def google_key_loaded():
    """Load the Google Maps API key from an environment variable."""
    load_dotenv()
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        Path(".env").write_text("GOOGLE_MAPS_API_KEY=<your_key_here>")
        print("No GOOGLE_MAPS_API_KEY environment variable found. Open the .env file and enter your API key.")
        return False
    elif not re.match(r'^[A-Za-z0-9_-]{35,40}$', api_key):
        print("Invalid GOOGLE_MAPS_API_KEY. Check the .env file and enter a valid API key.")
        return False
    return True


def image_to_base64(path):
    """Convert an image file to a base64 string."""
    with open(path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def generate_html(metadata: List[Dict]) -> None:
    """Generate an HTML file to display old and new images side by side with base64 encoding."""
    for entry in metadata:
        # Convert image paths to base64 strings
        entry['old_image_base64'] = image_to_base64(entry['old_image'])
        entry['new_image_base64'] = image_to_base64(entry['new_image'])

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Image Comparison</title>
        <style>
            body {font-family: Arial, sans-serif; margin: 20px;}
            .container {display: flex; flex-wrap: wrap; justify-content: space-around;}
            .image-pair {margin-bottom: 20px;}
            img {max-width: 100%; max-height: 200px;}
            .caption {text-align: center;}
        </style>
    </head>
    <body>
        <h2>Image Comparison</h2>
        <div class="container">
            {% for entry in metadata %}
                {% if entry.status == 'Changed' %}
                    <div class="image-pair">
                        <img src="data:image/jpeg;base64,{{ entry.old_image_base64 }}" alt="Old Image">
                        <img src="data:image/jpeg;base64,{{ entry.new_image_base64 }}" alt="New Image">
                        <p class="caption">{{ entry.Name }} ({{ entry.Location }})</p>
                    </div>
                {% endif %}
            {% endfor %}
        </div>
    </body>
    </html>
    """
    from jinja2 import Template
    template = Template(html_template)
    html_content = template.render(metadata=metadata)
    with open("image_comparison.html", "w") as html_file:
        html_file.write(html_content)
    os.system("image_comparison.html")


def main():
    if google_key_loaded():

        metadata = load_metadata()
        updates = update_images(metadata)
        if updates:
            save_metadata(metadata)
            print(tabulate(updates, headers="keys", tablefmt="pretty"))
            generate_html(updates)
        else:
            print("No updates to report.")


if __name__ == "__main__":
    main()
