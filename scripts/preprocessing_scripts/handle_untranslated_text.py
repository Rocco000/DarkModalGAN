import pandas as pd
from pathlib import Path
from preprocessing_text import initialize_bert_tokenizer, tokenization, is_written_in_english, plot_full_text_length, split_dataset

if __name__ == "__main__":
    json_path = Path(input("Provide the path of the json file:\n"))

    if not json_path.exists() or not json_path.is_file():
        raise FileNotFoundError(f"{json_path} doesn't exist!")

    # Load dataset
    df = pd.read_json(json_path, orient="records", lines=True)

    # Check text language
    df["lang"] = df["full_text"].apply(is_written_in_english) 

    # Extract non-english text
    non_english = df[df["lang"] == False]["lang"]
    print(f"Number of non-english text after pre-processing: {len(non_english)}")

    if len(non_english) > 0:
        # Manually translation
        for index, row in df[df['lang'] == False].iterrows():
            print(f"Non-English Text:\n{row['full_text']}\n")  # Print the non-English text
            
            # Input the manual translation
            manual_translation = input("Please provide the English translation:\n")
            manual_translation = manual_translation.strip().lower()
            
            # Update the DataFrame with the manual translation
            df.at[index, 'full_text'] = manual_translation
            df.at[index, 'lang'] = True

        df.drop(columns=['lang'], inplace=True) # Remove the lang field

        # Update dataset
        df.to_json(json_path, orient="records", lines=True)
        print("All translations completed and dataset updated.")

        tokenizer = initialize_bert_tokenizer()

        # Calculate the token length
        df["token_length"] = df["full_text"].apply(lambda x: len(tokenization(x, tokenizer)) if isinstance(x, str) else 0)

        save_path = Path(input("Provide a path to store the text analysis output:\n"))

        plot_full_text_length(df, save_path)

        split_dataset(df, 510, save_path)