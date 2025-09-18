from PIL import Image
import os

def extract_max_rectangle(image_path, area=None, is_polygon=False):
    """
    Extracts the bounding rectangle of a polygon from an image and saves it as a new file.

    Args:
        image_path (str): Path to the input image.
        area (list of [x, y]): List of polygon coordinates (from YOLO).
        is_polygon (bool): Whether to crop using polygon coordinates or not.

    Returns:
        str: Path to the cropped image (or original if no polygon).
    """
    if is_polygon and area and len(area) > 0:
        # Open image
        image = Image.open(image_path)
        width, height = image.size

        # Get bounding rectangle from polygon points
        xs = [x for x, y in area]
        ys = [y for x, y in area]
        min_x, min_y = max(0, int(min(xs))), max(0, int(min(ys)))
        max_x, max_y = min(width, int(max(xs))), min(height, int(max(ys)))

        # Ensure width and height > 0
        crop_width = max(1, max_x - min_x)
        crop_height = max(1, max_y - min_y)

        cropped = image.crop((min_x, min_y, min_x + crop_width, min_y + crop_height))
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_polygon{ext}"
        cropped.save(output_path)

        return output_path
    else:
        # No polygon provided, return original path
        return image_path
