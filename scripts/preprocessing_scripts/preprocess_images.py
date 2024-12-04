from pathlib import Path
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def plot_img_size(img_list:list, folder_path:Path, save_path:Path) -> None:
    """
    Plot the image size.
    :param img_list: a list containing images path.
    :param folder_path: a Path where the images are stored.
    :param save_path: a Path where preprocessing outputs will be stored.
    """
    # Dictionary to store image dimensions
    size_counts = {}

    for img in img_list:
        img_path = folder_path.joinpath(img)

        try:
            # Open the image and extract dimensions
            with Image.open(img_path) as image:
                width, height = image.size
                size = f"{width}x{height}"
                size_counts[size] = size_counts.get(size, 0) + 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Convert dictionary to lists for plotting
    sizes = list(size_counts.keys())
    counts = list(size_counts.values())

    # Sort by count for better visualization
    sorted_sizes_counts = sorted(zip(sizes, counts), key=lambda x: x[1], reverse=True)
    sorted_sizes, sorted_counts = zip(*sorted_sizes_counts)

    # Plot the distribution of image sizes
    plt.figure(figsize=(12, 8))
    bars = plt.bar(sorted_sizes, sorted_counts, color="skyblue", alpha=0.8) # Vertical bar plot
    plt.title("Distribution of Image Sizes", fontsize=14)
    plt.xlabel("Image Size (Width x Height)", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=45, ha="right")  # Rotate x-axis labels for better readability
    
    # Annotate each bar with its count
    for bar, count in zip(bars, sorted_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,  # Center of the bar
            bar.get_height() + 0.5,  # Slightly above the top of the bar
            str(count),  # The count value as a string
            ha="center",  # Center alignment
            va="bottom",  # Bottom alignment
            fontsize=10,  # Font size for the annotations
            color="black"  # Annotation text color
        )

    plt.tight_layout()
    plt.savefig(save_path.joinpath("plot_img_size.png"))
    plt.show()

def resize_images(size:tuple[int, int], img_list:list, folder_path:Path, save_path:Path) -> None:
    """
    Resize images to the specified size.
    :param size: the new image size.
    :param img_list: a list containing images path.
    :param folder_path: a Path where the images are stored.
    :param save_path: a Path where preprocessing outputs will be stored.
    """
    for img in img_list:
        img_path = folder_path.joinpath(img)

        if not img_path.exists():
            raise FileNotFoundError(f"{img_path} doesn't exist")
        
        try:
            # Open the image
            with Image.open(img_path) as image:
                img_w, _ = image.size
                resized_img = None
                # Gradually upscale the image (only for small images)
                if img_w == 74 or img_w == 100:
                    resized_img = image.resize((150,150), Image.Resampling.LANCZOS)
                
                resized_img = image.resize(size, Image.Resampling.LANCZOS)
                
                # Save the resized image to the output directory
                output_path = save_path.joinpath(img)
                macro_category_folder = output_path.parent
                macro_category_folder.mkdir(parents=True, exist_ok=True)
                resized_img.save(output_path)
        except Exception as e:
            print(f"Error processing {img}: {e}")

def main_preprocessing_images(data:pd.DataFrame, folder_path:Path, save_path:Path) -> None:
    """
    :param data: a DataFrame containing the crawled data
    :param folder_path: a Path where the images are stored.
    :param save_path: a Path where preprocessing outputs will be stored.
    """
    print("------------Preprocessing image data------------")
    plot_img_size(data["image_path"], folder_path, save_path)

    print("Resizing image to 224x224...")
    resize_images((224,224), data["image_path"], folder_path, save_path)

    