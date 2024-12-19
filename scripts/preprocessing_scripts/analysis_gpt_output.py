import pandas as pd
from pathlib import Path
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer
import numpy as np
import matplotlib.pyplot as plt
import re

# Local import
from preprocessing_text import initialize_bert_tokenizer, tokenization, lemmatize_text
from preprocessing_text import expand_acronyms, expand_drug_acronym, replace_concentration, replace_quantity_annotation, replace_percentage, expand_unit_measurement_acronyms, remove_macro_category

def check_macro_category(text:str) -> bool:
    """
    Check whether there are labels in the GPT processed text.

    :param text: the input text.
    :return: True whether there is at least one label in the input text, False otherwise.
    """
    mc_list = [r"cannabis", r"dissociatives?", r"ecstasy", r"opioids?", r"psychedelics?", r"steroids?", r"stimulants?"]

    for pattern in mc_list:
        if re.search(pattern, text):
            return True

    return False

def preprocess_gpt_text(text:str, lemmatizer:WordNetLemmatizer, tokenizer:BertTokenizer) -> str:
    """
    Preprocess the GPT output.

    :param text: the input text.
    :param lemmatizer: a WordNetLemmatizer instance to apply lemmatization.
    :param tokenizer: a BertTokenizer instance to apply tokenization.
    :return: the processed text.
    """
    text = text.lower().strip()
    text = expand_acronyms(text)
    text = expand_drug_acronym(text)
    text = replace_concentration(text)
    text = replace_quantity_annotation(text)
    text = replace_percentage(text)
    text = expand_unit_measurement_acronyms(text)
    text = remove_macro_category(text)

    text = lemmatize_text(text, lemmatizer, tokenizer)

    return text

def process_batches(batches:list, lemmatizer:WordNetLemmatizer, tokenizer:BertTokenizer) -> list:
    """
    Process batches using custom functions, WordNetLemmatizer for lemmatization and BertTokenizer for tokenization.

    :param batches: a list of batch file paths.
    :param lemmatizer: a WordNetLemmatizer instance to apply the lemmatization.
    :param tokenizer: a BertTokenizer instance to apply tokenization.
    :return: a list of processed DataFrame.
    """
    processed_batches = list()

    for batch in batches:
        # Read batch
        batch_data = pd.read_csv(batch, dtype={"image_path": str, "title":str, "description":str, "full_text":str, "token_length":int})           

        # Preprocess GPT output text
        batch_data["full_text"] = batch_data["full_text"].apply(lambda x: preprocess_gpt_text(x, lemmatizer, tokenizer))

        # Compute the token length using BertTokenizer
        batch_data["bert_token_length"] = batch_data["full_text"].apply(lambda x: len(tokenization(x, tokenizer)))

        processed_batches.append(batch_data)

    return processed_batches
    
def plot_mean_token_length(mean_before:list, mean_after:list, start_point:int, title:str, save_path:Path) -> None:
    """
    Plots a bar plot showing the mean token lengths before and after preprocessing for each batch.

    :param mean_before: list of mean token lengths before preprocessing (one per batch).
    :param mean_after: list of mean token lengths after preprocessing (one per batch).
    :param title: title of the bar plot.
    :param save_path: a Path to store the plot.
    """
    # Number of batches
    num_batches = len(mean_before)

    batch_labels = [f"batch {start_point+i}" for i in range(num_batches)]
    
    # X positions for the bars
    x = np.arange(num_batches)
    width = 0.35  # Width of the bars
    
    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, mean_before, width, label='Before GPT Preprocessing')
    plt.bar(x + width / 2, mean_after, width, label='After GPT Preprocessing')

    # Add a horizontal line at 510
    plt.axhline(y=510, color='red', linestyle='--', linewidth=1, label='Target Length (510)')
    
    # Add labels, title, and legend
    plt.title(title, fontsize=14)
    plt.xlabel("Batch", fontsize=12)
    plt.ylabel("Mean Token Length", fontsize=12)
    plt.xticks(x, batch_labels)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_length_distribution(batches:list, save_path:Path) -> None:
    """
    Plot an histogram of the token length distribution.

    :param batches: a list of processed batches.
    :param save_path: a Path to store the plot.
    """
    token_length = list() 
    for batch in batches:
        # Store token length of the i-th batch
        token_length = token_length + list(batch["bert_token_length"])

    plt.figure(figsize=(10, 6))
    plt.hist(token_length, bins=50, color='blue', alpha=0.7)
    plt.axvline(512, color='red', linestyle='dashed', linewidth=1, label="512 Tokens") # Add a line to show the maximum token length
    plt.title("Token length distribution after preprocessing")
    plt.xlabel("Token length")
    plt.ylabel("Number of texts")
    plt.savefig(save_path.joinpath(f"histogram_token_length_processed.png"))
    plt.show()

def calculate_mean_token_length(dataset:pd.DataFrame, batches:list) -> tuple[list, list]:
    """
    Calculate the mean of token length before and after the processing step with GPT model.

    :param dataset: a DataFrame containing original textual data.
    :param batches: a list of processed batches.
    :return: two lists containing the mean of token length before and after the processing step by GPT model.
    """
    mean_before = list()
    mean_after = list()

    for batch in batches:
        # Store the mean of token length for the i-th batch
        mean_after.append(batch["bert_token_length"].mean())

        # Inner join to get the origin batch
        origin_batch = pd.merge(dataset, batch[["image_path"]], on="image_path", how='inner')

        # Store the origin mean of token length for the i-th batch
        mean_before.append(origin_batch["token_length"].mean())

    return mean_before, mean_after

def calculate_number_exceeding_samples(batches:list, max_token_length:int=510) -> int:
    """
    Calculate the number of exceeding samples based on maximum token length provided.

    :param batches: a list of processed batches.
    :param max_token_length: the maximum token length.
    :return: the number of exceeding samples
    """
    exceed_samples = 0

    for batch in batches:
        data_gt = batch[batch["bert_token_length"] > max_token_length]["bert_token_length"]

        exceed_samples += len(data_gt)

    return exceed_samples

def plot_exceeding_samples(values:list, title:str, save_path:Path) -> None:
    """
    Plot the number of exceeding samples before and after processing of GPT model.

    :param values: a list containing the number of exceeding samples before and after the processing step.
    :param title: the title of plot.
    :param save_path: a Path where the plot will be stored.
    """
    x = np.arange(len(values))  # Positions for the bars

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(x, values, color=['blue', 'green'], alpha=0.7, width=0.6)
    
    # Add labels, title, and legend
    plt.title(title, fontsize=14)
    plt.xlabel("Exceed samples", fontsize=12)
    plt.ylabel("Number of samples", fontsize=12)
    plt.xticks(x, labels=["Exceed before", "Exceed after"])

    # Annotate values above the bars
    for i, value in enumerate(values):
        plt.text(i, value + 2, str(value), ha='center', fontsize=10)
    
    # Show the plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main_gpt_analysis(summarized_batches_path:Path, expanded_batches_path:Path, split_path:Path) -> list:
    """
    Analize and process the processed batches by GPT model.

    :param summarized_batches_path: a Path where the summarized batches are stored.
    :param expanded_batches_path: a Path where expanded batches are stored.
    :param split_path: a Path where textual data has been splitted on maximum token length.
    :return: a list of processed batches.
    """
    # Read splitted data
    text_gt = pd.read_csv(split_path.joinpath("text_gt.csv"), dtype={"image_path": str, "title":str, "description":str, "full_text":str, "token_length":int})
    text_le_e = pd.read_csv(split_path.joinpath("text_le_e.csv"), dtype={"image_path": str, "title":str, "description":str, "full_text":str, "token_length":int})

    # Get a folder path to store the analysis results
    gpt_analysis = Path(str(input("Provide a folder path where the analysis of GPT output will be stored:\n")))
    gpt_analysis.mkdir(exist_ok=True)

    tokenizer = initialize_bert_tokenizer()
    lemmatizer = WordNetLemmatizer()

    exceed_after = 0
    mean_before = list() # To store the mean of token length per batch BEFORE GPT preprocessing
    mean_after = list() # To store the mean of token length per batch AFTER GPT preprocessing

    # SUMMARIZED BATCHES
    summarized_batches = [batch for batch in summarized_batches_path.iterdir() if batch.is_file()]

    processed_sum_batches = process_batches(summarized_batches, lemmatizer, tokenizer) # Process summarized text

    # Calculate the mean of token length per batch before and after processing
    summarized_before, summarized_after = calculate_mean_token_length(text_gt, processed_sum_batches)

    # Calculate the number of exceeding samples
    exceed_summarized_after = calculate_number_exceeding_samples(processed_sum_batches)

    # EXPANDED BATCHES
    expanded_batches = [batch for batch in expanded_batches_path.iterdir() if batch.is_file()]

    processed_exp_batches = process_batches(expanded_batches, lemmatizer, tokenizer)

    expanded_before, expanded_after = calculate_mean_token_length(text_le_e, processed_exp_batches)

    exceed_expanded_after = calculate_number_exceeding_samples(processed_exp_batches)

    mean_before = summarized_before + expanded_before
    mean_after = summarized_after + expanded_after

    # Compare the mean of token length per batch before and after GPT processing
    j = 1
    for i in range(0, len(mean_before), 10):
        end_index = min(i + 10, len(mean_before))
        mean_before_subset = mean_before[i:end_index]
        mean_after_subset = mean_after[i:end_index]
        plot_mean_token_length(mean_before_subset, mean_after_subset, i, f"Comparison mean token lenght ({j})", gpt_analysis.joinpath(f"mean_bar_plot{j}.png"))
        j += 1

    # Plot token lenght distribution
    all_batches = processed_sum_batches + processed_exp_batches
    plot_length_distribution(all_batches, gpt_analysis)

    exceed_after = exceed_summarized_after + exceed_expanded_after # All number of exceeding samples after GPT processing
    exceed_before = len(text_gt["image_path"])
    print(f"Number of samples which have a token length grater than 510 tokens are: {exceed_after}")
    print(f"Number of exceeding samples in summarized: {exceed_summarized_after}")
    print(f"Number of exceeding samples in expanded: {exceed_expanded_after}")

    values = [exceed_before, exceed_summarized_after]
    plot_exceeding_samples(values, "Exceeding samples in summarized (>510 tokens)", gpt_analysis.joinpath("exceeding_samples_summarized.png"))

    values = [exceed_before, exceed_after]

    plot_exceeding_samples(values, "Exceeding samples after GPT (>510 tokens)", gpt_analysis.joinpath("all_exceeding_samples_after_gpt.png"))

    return all_batches