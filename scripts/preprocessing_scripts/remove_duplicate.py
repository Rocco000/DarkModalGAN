import imagehash
import pprint #to print in a fancy way
from PIL import Image
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import csv
import shutil
import matplotlib.pyplot as plt

def calculate_image_hash(image_path: Path) -> str:
    """
    Calculate the hash of an image using Perceptual Hashing (pHash).

    :param image_path: the image path on witch apply the hashing technique
    :return: the hash value of the given image.
    """

    image = Image.open(image_path)
    return imagehash.phash(image)

def show_images_side_by_side(image1_path: Path, image2_path: Path):
    """
    Display two images side by side in a single window.
    
    :param image1_path: Path of the first image.
    :param image2_path: Path of the second image.
    """
    # Open the images
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    # Create a new figure
    plt.figure(figsize=(10, 5))

    # Display the first image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1")
    plt.axis("off")

    # Display the second image on the right
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2")
    plt.axis("off")

    # Show the images side by side and wait until the window is closed
    plt.show()

def find_duplicates(image_list: list) -> list:
    """    
    Find duplicate images based on their hash value.

    :param image_list: the image list on which apply the hashing technique.
    :return: a list of images to be removed.
    """

    hash_dict = {}
    remove_list = list()

    for img in image_list:
        img_path = Path(img)

        if img_path.is_file():
            # Calculate the hash value
            img_hash = calculate_image_hash(img_path)

            if img_hash in hash_dict:
                #print(f"Duplicate found: {img} is a duplicate of {hash_dict[img_hash]}")
                #show_images_side_by_side(img, hash_dict[img_hash])
                remove_list.append(img)
            else:
                hash_dict[img_hash] = img
        else:
            raise FileNotFoundError(f"{img_path} is not a file")
    
    return remove_list

def remove_duplicates(csv_file: pd.DataFrame, image_list: list) -> pd.DataFrame:
    """
    Remove duplicate images and update the CSV file.
    :param csv_file: DataFrame containing the mapping between images and tabular data.
    :param image_list: a list of images to be removed.
    :return: the preprocessed dataset without duplicate images
    """
    # Get the image name
    img_name = [Path(img).name for img in image_list]

    print("Images to be removed:")
    x = csv_file[csv_file["image_path"].apply(lambda x: Path(x).name).isin(img_name)]
    print(x)

    csv_filtered = csv_file[~csv_file["image_path"].apply(lambda x: Path(x).name).isin(img_name)]

    print("CSV without invalid images")
    print(csv_filtered)

    return csv_filtered

if __name__ == '__main__':
    csv_path = input("Provide the CSV path where image mapping is stored:\n")
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"The CSV file {csv_path} does not exist!")

    save_path = input("Provide the folder path where the filtered data will be saved:\n")
    save_path = Path(save_path)

    save_path.mkdir(exist_ok=True)

    # Read the csv file
    df = pd.read_csv(csv_path)

    to_remove = list()
    count_deleted = {}

    marketplaces = df["marketplace"].unique()

    for marketplace in tqdm(marketplaces, desc="Processing marketplaces", unit="marketplace"):
        # Get macro-categories
        macro_categories = df[df["marketplace"] == marketplace]["macro_category"].unique()

        for macro_category in tqdm(macro_categories, desc=f"Processing macro-categories of {marketplace}", unit="macro-category"):
            # Get all micro-categories of the given macro-category
            micro_categories = df[(df["marketplace"] == marketplace) & (df["macro_category"] == macro_category)]["micro_category"].unique()

            for micro_category in tqdm(micro_categories, desc=f"Processing {macro_category}", unit="micro-category"):
                # Get all images with the given macro and micro categories
                image_list = df[(df["marketplace"] == marketplace) & (df["macro_category"] == macro_category) & (df["micro_category"] == micro_category)]["image_path"]

                image_list = [f"{csv_path.parents[0]}/{img}" for img in image_list]

                # Find duplicates
                duplicates = find_duplicates(image_list)
                to_remove = to_remove + duplicates
                count_deleted[f"{marketplace}-{macro_category}-{micro_category}"] = len(duplicates)
    
    # Delete duplicates
    filtered_data = remove_duplicates(df, to_remove)

    # Save the filterd dataset
    filtered_data.to_csv(save_path.joinpath("filtered_data.csv"), index= False)

    # Copy non-duplicate images in the save_path
    for row in filtered_data.itertuples():
        image_path = csv_path.parents[0].joinpath(row.image_path) # Old image path

        copy_path = save_path.joinpath(row.image_path) # New image path

        parent_folder = copy_path.parent
        parent_folder.mkdir(exist_ok=True)

        shutil.copy(image_path, copy_path)

    print("Number of images deleted per micro-cateogry (during remove_duplicates):")
    pprint.pprint(count_deleted)

    with open(save_path.joinpath("deleted_image_per_class.csv"), mode= "w", newline= "") as csv_file:
        writer = csv.writer(csv_file)

        # Write the header
        writer.writerow(["class", "deleted"])

        for key, value in count_deleted.items():
            writer.writerow([key, value])
