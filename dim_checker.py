from PIL import Image
import sys
import os


def get_image_dimensions(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        return width, height
    except FileNotFoundError:
        print(f"Error: Image file not found at '{image_path}'")
        return None
    except Exception as e:
        print(f"Error opening or reading image '{image_path}': {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python get_image_dimensions.py <image_file_path> [another_image_file_path ...]")
        sys.exit(1)

    image_paths = sys.argv[1:]
    for path in image_paths:
        dimensions = get_image_dimensions(path)
        if dimensions:
            width, height = dimensions
            print(f"'{path}': Width = {width} pixels, Height = {height} pixels")
