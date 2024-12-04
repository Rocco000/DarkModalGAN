from pathlib import Path
import pandas as pd
from preprocessing_text import main_preprocessing_text_data
from preprocess_tabular_data import main_preprocessing_tabular_data
from preprocess_images import main_preprocessing_images

if __name__ == "__main__":
    csv_path = input("Provide the CSV path:\n")
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"The CSV file {csv_path} does not exist!")
    
    # Create the output folder
    preprocess_path = csv_path.parents[1].joinpath("preprocessed_data")
    preprocess_path.mkdir(exist_ok=True)

    dataset = pd.read_csv(csv_path, dtype={"image_path": str, "marketplace":str, "title":str, "description":str, "vendor":str, "origin": str, "destination":str, "currency":str, "price":float, "cryptocurrency":str, "crypto_price":float, "macro_category":str, "micro_category":str})

    print("START PREPROCESSING...")
    preprocessed_data = main_preprocessing_tabular_data(dataset, preprocess_path)
    print("\n\n")

    # Extract all countries
    countries = set(dataset["origin"].unique())
    countries = countries | set(dataset["destination"].unique())
    countries = list(countries)

    preprocessed_data = main_preprocessing_text_data(preprocessed_data, preprocess_path, countries)

    # Store the preprocessed data in a JSON file due to the one-hot encoding
    preprocessed_data.to_json(preprocess_path.joinpath("preprocessed_data.json"), orient="records", lines=True)

    main_preprocessing_images(preprocessed_data, csv_path.parent, preprocess_path)