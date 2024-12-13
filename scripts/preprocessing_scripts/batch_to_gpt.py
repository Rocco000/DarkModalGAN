import pandas as pd
from pathlib import Path

def split_csv_into_batches(input_file:Path, output_dir:Path, batch_size=20) -> None:
    """
    Splits a CSV file into batches of specified size and saves each batch to a separate CSV file.
    :param input_file: Path to the input CSV file.
    :param output_dir: Directory where the output CSV files will be saved.
    :param batch_size: Number of samples per batch (default is 20).
    """
    # Load the input CSV file
    data = pd.read_csv(input_file, dtype={"image_path": str, "title":str, "description":str, "full_text":str, "token_length":int})
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate the number of batches
    num_batches = (len(data) + batch_size - 1) // batch_size  # Ceiling division to also count the last batch
    
    for batch_num in range(num_batches):
        # Determine the start and end of the batch
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(data))
        
        # Extract the batch
        batch_data = data.iloc[start_idx:end_idx]
        
        # Save the batch to a CSV file
        output_file = output_dir.joinpath(f'batch_{batch_num + 1}.csv')
        batch_data.to_csv(output_file, index=False)
        
        print(f"Batch {batch_num + 1} saved to {output_file}")

if __name__ == "__main__":
    ge_csv = Path(str(input("Provide the CVS path which contains text with a token length greater than 510 tokens:\n")))

    if not ge_csv.exists():
        raise FileNotFoundError(f"{ge_csv} doesn't exist!")

    le_csv = Path(str(input("Provide the CSV path which contains text with a token length less equal to 510 tokens:\n")))

    if not le_csv.exists():
        raise FileNotFoundError(f"{le_csv} doesn't exist!")
    
    batch_path = Path(str(input("Provide a path to store batches:\n")))

    ge_path = batch_path.joinpath("ge_batches")
    le_path = batch_path.joinpath("le_batches")
    
    split_csv_into_batches(ge_csv, ge_path)
    split_csv_into_batches(le_csv, le_path)