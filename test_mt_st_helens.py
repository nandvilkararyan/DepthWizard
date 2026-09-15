import os
import sys
import urllib.request
from cli_test import test_single_image

def main():
    image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Mount_St_Helens_from_space.jpg/1200px-Mount_St_Helens_from_space.jpg'
    image_path = 'output/test_scratch/mt_st_helens_satellite.jpg'

    if not os.path.exists(image_path):
        print("Downloading Mount St Helens image...")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        urllib.request.urlretrieve(image_url, image_path)
        print("Download complete.")

    print(f"Running pipeline on {image_path}...")
    test_single_image(image_path)
    print("Pipeline complete. Check the output directory for results.")

if __name__ == '__main__':
    main()
