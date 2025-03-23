import cv2
import numpy as np

def read_image(image_path):
    """Read an image from a file."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return image

def convert_to_grayscale(image):
    """Convert an image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def resize_image(image, width, height):
    """Resize an image to the given dimensions."""
    return cv2.resize(image, (width, height))

def apply_edge_detection(image):
    """Apply Canny edge detection to an image."""
    return cv2.Canny(image, 100, 200)

def save_image(image, output_path):
    """Save an image to a file."""
    cv2.imwrite(output_path, image)

def main():
    image_path = 'path_to_your_image.jpg'  # Replace with your image path
    output_path = 'output_image.jpg'       # Replace with your desired output path

    # Read the image
    image = read_image(image_path)

    # Convert the image to grayscale
    grayscale_image = convert_to_grayscale(image)

    # Resize the image
    resized_image = resize_image(grayscale_image, width=256, height=256)

    # Apply edge detection
    edges = apply_edge_detection(resized_image)

    # Save the processed image
    save_image(edges, output_path)

    print(f"Processed image saved to {output_path}")

if __name__ == '__main__':
    main()
