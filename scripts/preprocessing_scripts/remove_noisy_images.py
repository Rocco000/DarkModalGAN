from pathlib import Path
import pandas as pd
import re

def remove_noisy_images(data:pd.DataFrame, folder_path:Path, img_list: list) -> None:
    """
    Remove the corresponding rows to the image names in img_list from the csv_file and remove these images from the folder.

    :param data: DataFrame containing the mapping between images and tabular data.
    :param folder_path: Path which contains the crawled images.
    :param img_list: a list of image names to be removed.
    """
    # Get all image names in the CSV file
    csv_img_name = data["image_path"].apply(lambda x: x.split('/')[-1].strip().lower())
    img_list = [img.strip().lower() for img in img_list]

    # Check if all image names in the list are present in the extracted image names
    missing_images = [name for name in img_list if name not in csv_img_name.values]

    # Raise an exception if there are missing image names
    if missing_images:
        raise ValueError(f"The following image names are missing from the column 'image_path':\n {missing_images}")

    # Filter the DataFrame to exclude rows with matching values
    filtered_data = data[~data["image_path"].apply(lambda x: x.split('/')[-1]).isin(img_list)]

    # Delete images
    for img in img_list:
        macro_category = re.search(r"([a-zA-Z]+)", img).group(1)
        img_path = folder_path.joinpath(f"{macro_category}/{img}")
        if img_path.exists():
            img_path.unlink()
        else:
            raise FileNotFoundError(f"The file at {img_path} does not exist.")

    # Store the new dataset
    filtered_data.to_csv(folder_path.joinpath("filtered_data.csv"), index= False)

if __name__ == '__main__':
    # Delete specific images (optionally)
    choice = bool(int(input("Do you want to remove specific images? (0 = no, 1 = yes)\n")))

    if choice:
        csv_path = Path(input("Provide the CSV path where image mapping is stored:\n"))
        
        if not csv_path.exists():
            raise FileNotFoundError(f"The provided path doesn't exist {csv_path}")
        
        dataset = pd.read_csv(csv_path) # Read the CSV file

        csv_path2 = Path(input("Provide the CSV path containing the images to be removed:\n"))

        if not csv_path2.exists():
            raise FileNotFoundError(f"The provided path doesn't exist {csv_path2}")
        
        img_to_be_removed = pd.read_csv(csv_path2)
        remove_noisy_images(dataset, csv_path.parents[0], img_to_be_removed["image_path"])