import os
import matplotlib.pyplot as plt
from PIL import Image


def visualize_images_with_numbering_below(folder_path):
    # Get list of image files in the folder (assumes there are exactly 9 images)
    image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    # Check if there are exactly 9 images
    if len(image_files) != 9:
        raise ValueError("The folder must contain exactly 9 images.")

    # Create a 3x3 subplot
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    # Load and display each image in the grid
    for i, ax in enumerate(axes.flat):
        img = Image.open(os.path.join(folder_path, image_files[i]))
        ax.imshow(img)
        ax.axis('off')  # Hide the axes
        # Add numbering below each image
        ax.set_title(f'Image {i + 1}', fontsize=12, pad=10)  # Pad adds space below the image

    # Adjust spacing between images
    plt.tight_layout()
    plt.savefig(os.path.join(folder_path, 'memory_combined'))
    plt.show()

# Example usage: Provide the path to the folder containing the images
folder_path = 'out/choosen memory'
visualize_images_with_numbering_below(folder_path)
