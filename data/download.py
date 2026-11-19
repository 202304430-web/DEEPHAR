"""
Downloads and extracts UCI-HAR (Human Activity Recognition Using Smartphones).
Source: UCI ML Repository. 9-channel raw inertial signals, 128-step windows @50Hz,
pre-windowed (2.56s, 50% overlap), 6 classes, 30 subjects.
"""
import os
import zipfile
import urllib.request

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def download_and_extract():
    os.makedirs(RAW_DIR, exist_ok=True)
    zip_path = os.path.join(RAW_DIR, "uci_har.zip")
    inner_dir = os.path.join(RAW_DIR, "UCI HAR Dataset")

    if os.path.isdir(inner_dir):
        print(f"[skip] already extracted at {inner_dir}")
        return inner_dir

    print(f"Downloading UCI-HAR to {zip_path} ...")
    urllib.request.urlretrieve(URL, zip_path)

    print("Extracting outer zip ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(RAW_DIR)

    # The outer zip contains an inner "UCI HAR Dataset.zip" in some mirrors; handle both.
    inner_zip = os.path.join(RAW_DIR, "UCI HAR Dataset.zip")
    if os.path.isfile(inner_zip) and not os.path.isdir(inner_dir):
        with zipfile.ZipFile(inner_zip, "r") as z:
            z.extractall(RAW_DIR)

    if not os.path.isdir(inner_dir):
        raise RuntimeError("Extraction failed: 'UCI HAR Dataset' folder not found. Check mirror structure.")

    os.remove(zip_path)
    print(f"Done. Raw data at: {inner_dir}")
    return inner_dir


if __name__ == "__main__":
    download_and_extract()
