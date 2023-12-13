"""A script to download and unzip the test data from Google Drive. Requires requests and pyzipper."""

import os
import sys

import pyzipper
import requests


def download_file_from_google_drive(id_, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={"id": id_}, stream=True)
    token = get_confirm_token(response)
    if token:
        params = {"id": id_, "confirm": token}
        response = session.get(URL, params=params, stream=True)
    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def unzip_file(file_path, extract_to, password_):
    with pyzipper.AESZipFile(file_path, compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES) as zf:
        zf.extractall(extract_to, pwd=str.encode(password_))


if __name__ == "__main__":
    file_id = sys.argv[1]
    password = sys.argv[2]
    location = sys.argv[3]

    zip_file_path = "data.zip"

    print("Downloading data...")
    download_file_from_google_drive(file_id, zip_file_path)
    print(f"Unzipping data to {location}...")
    os.makedirs(location, exist_ok=True)
    unzip_file(zip_file_path, location, password)
