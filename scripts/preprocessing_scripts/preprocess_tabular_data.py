from pathlib import Path
import pandas as pd
import csv
import re
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import seaborn as sns
import numpy as np

def convert_price(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all prices to American Dollar (USD) based on predefined exchange rates.
    :param data: a DataFrame containing product information
    :return: a DataFrame with all prices in american dollar.
    """
    exchange_rates = {
        "USD": 1.0,    # 1 USD = 1 USD
        "EUR": 1.07,   # 1 EUR = 1.07 USD
        "GBP": 1.29,   # 1 GBP = 1.29 USD
        "CAD": 0.72,   # 1 CAD = 0.72 USD
        "AUD": 0.66   # 1 AUD = 0.66 USD
    }

    # Remove commas from the price column (american format)
    #data["price"] = data["price"].str.replace(",", "")

    # Ensure the price column is numeric
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    data["price"] = data["price"] * data["currency"].map(exchange_rates)

    if data["price"].isnull().any():
        print("Warning: Some prices could not be converted due to missing exchange rates.")

    # Update the currency column to USD
    data["currency"] = "USD"

    return data

def convert_price_in_crypto(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert prices to Monero (XMR) for rows that don't have a cryptocurrency value.
    :param data: a DataFrame containing product information
    :return: a DataFrame without NaN or empty value for crypto_price column.
    """
    exchange_rates = {
        "USD": 0.0063,  # 1 USD = 0.0063 XMR
        "EUR": 0.0067,  # 1 EUR = 0.0067 XMR
        "GBP": 0.0081,  # 1 GBP = 0.0081 XMR
        "CAD": 0.0045,  # 1 CAD = 0.0045 XMR
        "AUD": 0.0041,  # 1 AUD = 0.0041 XMR
    }

    # Replace empty strings in the cryptocurrency column with NaN
    data["cryptocurrency"] = data["cryptocurrency"].replace("", pd.NA)

    # Filter rows where cryptocurrency is missing (NaN)
    mask = data["cryptocurrency"].isna()

    # Ensure the price column is numeric
    data["price"] = pd.to_numeric(data["price"], errors="coerce")

    # Convert price in Monero
    data.loc[mask, "crypto_price"] = data.loc[mask, "price"] * data.loc[mask, "currency"].map(exchange_rates)
    data.loc[mask, "cryptocurrency"] = "XMR"

    return data

def currency_info(data: pd.DataFrame) -> None:
    """
    Calculate the most frequent currency in the given DataFrame and counts the number of instance per currency and cryptocurrency
    :param data: a DataFrame containing product information
    """
    currency_list = data["currency"].unique()

    print("These are the currency in your data:")
    print(currency_list)

    most_frequent_currency = data["currency"].mode()[0]
    print(f"The most frequent currency in your data is: {most_frequent_currency}")

    currency_count = data["currency"].value_counts()
    print(f"Number of row per currency:\n{currency_count}")

    cryptocurrency_count = data["cryptocurrency"].value_counts()
    print(f"Number of row per cryptocurrency:\n{cryptocurrency_count}")

def fix_country(country: str) -> str:
    """
    Remove redundant information from the given string
    :param country: the country in the origin or destination field in the dataset.
    :return: a cleaned-up string without the redundant information.
    """
    country = country.strip()

    if country.startswith("France"): # For Cocorico
        return "France"
    elif country.startswith("European Union,"): # For Cocorico
        return "European Union"
    elif country == "Digital item or service": # For DrugHub
        # Special case handling for destination
        return "World Wide"
    else:
        # Remove acronym within parentheses
        fixed_string = re.sub(r"\(.*?\)","",country)
        return fixed_string.strip()

def fix_cocorico_origin_destination(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the origin and destination fields for rows which have Cocorico as marketplace.
    :param data: a DataFrame containing product information
    :return: a DataFrame with fixed value for origin and destination fields.
    """
    # Filter only the rows for the cocorico marketplace
    cocorico_mask = data["marketplace"] == "cocorico"

    # Apply the fix_country function on the origin and destination fields
    data.loc[cocorico_mask, "origin"] = data.loc[cocorico_mask, "origin"].apply(fix_country)
    data.loc[cocorico_mask, "destination"] = data.loc[cocorico_mask, "destination"].apply(fix_country)

    return data

def fix_drughub_destination(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fix the destination fields for rows which has DrugHub as marketplace. Specifically, substitute the destination 'Digital item or service' with 'World Wide'
    :param data: a DataFrame containing product information.
    :return: a DataFrame with fixed value for origin and destination fields.
    """
    # Filter only the rows for the drughub marketplace
    drugub_mask = data["marketplace"] == "drughub"
    data.loc[drugub_mask, "destination"] = data.loc[drugub_mask, "destination"].apply(fix_country)

    return data

def distribution_analysis(data:pd.DataFrame, column_name:str, save_path:Path) -> None:
    """
    Calculate and plot skewness, kurtosis, and descriptive statistics for a given column.
    :param data: a DataFrame containing the product information.
    :param column_name: the column name.
    :param save_path: a folder path in which plots will be stored.
    """
    if column_name not in data.columns:
        raise ValueError(f"{column_name} doesn't exist in the given dataset!")
    
    save_path = save_path.joinpath("plot_distribution_analysis")
    save_path.mkdir(exist_ok=True)

    # Calculate skewness
    data_skewness = skew(data[column_name])

    # Calculate kurtosis
    data_kurtosis = kurtosis(data[column_name], fisher=True)  # Fisher=True gives excess kurtosis (0 for normal distribution)

    print(f"Skewness of {column_name}: {data_skewness}")
    print(f"Kurtosis of {column_name}: {data_kurtosis}")
    stats_before = {
        "mean": data[column_name].mean(),
        "median": data[column_name].median(),
        "std": data[column_name].std(),
        "min": data[column_name].min(),
        "max": data[column_name].max(),
        "skewness": data_skewness,
        "kurtosis": data_kurtosis,
    }

    # Plot histogram
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column_name], bins=50, color='blue', alpha=0.7) #kde=True
    plt.title(f"Histogram of {column_name} column")
    plt.xlabel(f"{column_name}")
    plt.ylabel("Frequency")
    plt.savefig(save_path.joinpath(f"histogram_{column_name}.png"))
    plt.show()

    flag = bool(int(input("Are the data skewed? (0 = no, 1 = yes)\n")))

    if flag:
        # Log transformation
        data[column_name] = np.log1p(data[column_name])

        # Realculate skewness and kurtosis
        data_skewness = skew(data[column_name])
        data_kurtosis = kurtosis(data[column_name], fisher=True)  # Fisher=True gives excess kurtosis (0 for normal distribution)
        print(f"Skewness after log transformation: {data_skewness}")
        print(f"Kurtosis after log transformation: {data_kurtosis}")
        stats_after = {
            "mean": data[column_name].mean(),
            "median": data[column_name].median(),
            "std": data[column_name].std(),
            "min": data[column_name].min(),
            "max": data[column_name].max(),
            "skewness": data_skewness,
            "kurtosis": data_kurtosis,
        }

        # Plot histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column_name], bins=50, color='blue', alpha=0.7) #kde=True
        plt.title(f"Histogram of {column_name} column AFTER PREPROCESSING")
        plt.xlabel(f"{column_name}")
        plt.ylabel("Frequency")
        plt.savefig(save_path.joinpath(f"histogram_{column_name}_log_transformation.png"))
        plt.show()

        csv_path = save_path.joinpath(f"{column_name}_analysis.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["state", "mean", "median", "std", "min", "max", "skewness", "kurtosis"])
            writer.writerow(["before", *stats_before.values()])
            writer.writerow(["after", *stats_after.values()])

def label_encoding(label:str) -> int:
    """
    Apply the label encoding. Transform macro-category field in integer.
    :param label: the product label, namely the macro-category.
    :return: encoded label.
    """
    category_map = {
        "cannabis": 0,
        "dissociatives": 1,
        "ecstasy": 2,
        "opioids": 3,
        "psychedelics": 4,
        "steroids": 5,
        "stimulants": 6
    }

    if label in category_map.keys():
        return category_map[label]
    else:
        raise ValueError(f"{label} doesn't exist in the mapping!")

def one_hot_encoding(data:pd.DataFrame, column_name:str) -> pd.DataFrame:
    """
    Apply the one-hot encoding on the input column.
    :param data: a DataFrame containing the crawled data.
    :param column_name: the column name on which apply the one-hot encoding.
    :return: the modified DataFrame with column_name encoded.
    """
    if column_name not in data.columns:
        raise ValueError(f"{column_name} doesn't exist in the given dataset!")
    
    # Perform one-hot encoding
    unique_values = sorted(data[column_name].unique())  # Order column values
    print(f"The lenght of the bynary vector for {column_name} will be: {len(unique_values)}")
    one_hot_map = {val: idx for idx, val in enumerate(unique_values)}  # Map each category to an index

    # Replace the original column with one-hot encoded vectors
    data[column_name] = data[column_name].apply(lambda x: [1 if i == one_hot_map[x] else 0 for i in range(len(unique_values))])

    return data

def main_preprocessing_tabular_data(data: pd.DataFrame, save_path:Path) -> pd.DataFrame:
    """
    The main function to preprocess the tabular data.
    :param data: a DataFrame containing the crawled data.
    :return: a preprocessed DataFrame.
    """
    print("------------Preprocessing tabular data------------")
    data_copy = data.copy()
    currency_info(data_copy)
    data_copy = fix_cocorico_origin_destination(data_copy)
    data_copy = fix_drughub_destination(data_copy)
    
    print("Converting prices in american dollar...")
    data_copy = convert_price(data_copy)
    print("Defining crypto price for rows with NaN value...")
    data_copy = convert_price_in_crypto(data_copy)
    
    print("Distribution analysis for price and crypto_price columns...")
    distribution_analysis(data_copy, "price", save_path)
    distribution_analysis(data_copy, "crypto_price", save_path)

    # Concatenate macro and micro category values
    data_copy['micro_category'] = data_copy['macro_category'] + '_' + data_copy['micro_category']

    # Label encoding
    print("Label encoding...")
    data_copy["macro_category"] = data_copy["macro_category"].apply(label_encoding)

    # Feature encoding
    print("One-hot encoding for categorical data...")
    data_copy = one_hot_encoding(data_copy, "micro_category")
    data_copy = one_hot_encoding(data_copy, "origin")
    data_copy = one_hot_encoding(data_copy, "destination")

    return data_copy