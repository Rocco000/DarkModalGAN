import pandas as pd
from pathlib import Path

# Local import
from analysis_gpt_output import main_gpt_analysis

def update_dataset(dataset:pd.DataFrame, processed_batches:pd.DataFrame, save_path:Path) -> None:
    """
    Update the full_text field of input dataset with full_text field of processed_batches and store it in save_path.
    
    :param dataset: a DataFrame to update.
    :param processed_batches: a DataFrame containing all processed batches by GPT model.
    :param save_path: a Path where the updated dataset will be stored.
    """
    # Merge origin dataset with batches
    merged_data = dataset.merge(processed_batches[["image_path", "full_text"]], on="image_path", how='left', suffixes=('', '_batch'))

    # Check for NaN values in the 'full_text_batch' column (after the merge)
    nan_rows = merged_data[merged_data["full_text_batch"].isna()]

    if not nan_rows.empty:
        print(f"Unmatched row!!!\n{nan_rows}")
        raise Exception("Unmatched row during the join operation.")

    merged_data["full_text"] = merged_data["full_text_batch"]

    # Remove full_text_batch column
    updated_dataset = merged_data.drop(columns=["full_text_batch"])

    # Store the updated dataset
    updated_dataset.to_json(save_path.joinpath("final_dataset.json"), orient="records", lines=True)

if __name__ == "__main__":
    # Get processed batches by GPT model
    gpt_output_folder = Path(str(input("Provide the folder path where processed batches (by GPT model) are stored:\n")))

    if not gpt_output_folder.exists():
        raise FileNotFoundError(f"{gpt_output_folder} doesn't exist!")
    
    summarized_batches_path = gpt_output_folder.joinpath("summarized_batches")
    expanded_batches_path = gpt_output_folder.joinpath("expanded_batches")

    # Get the origin textual data splitted on token length
    split_path = Path(str(input("Provide the folder path where splitted dataset (based on maximum token length) has been stored:\n")))

    if not split_path.exists():
        raise FileNotFoundError(f"{split_path} doesn't exist!")

    # ANLYSIS OF GPT OUTPUT
    processed_batches = main_gpt_analysis(summarized_batches_path, expanded_batches_path, split_path)

    dataset_path = Path(str(input("Provide the folder path where JSON dataset (processed by my custom functions) is stored:\n")))

    if not dataset_path.exists():
        raise FileNotFoundError(f"{dataset_path} doesn't exist!")

    # Concatenate all batches
    processed_batches_df = pd.concat(processed_batches, ignore_index=True) 

    # Read origin dataset
    dataset = pd.read_json(dataset_path, orient="records", lines=True)

    final_path = Path(str(input("Provide a folder path where the updated dataset will be stored:\n")))
    final_path.mkdir(exist_ok=True)

    update_dataset(dataset, processed_batches_df, final_path)