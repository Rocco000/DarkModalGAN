from pathlib import Path
import pandas as pd
import numpy as np
from transformers import BertTokenizer
import torch
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """
    This class represents my multimodal dataset.
    """

    def __init__(self, ds:pd.DataFrame, img_folder:Path):
        """
        :param ds: a DataFrame instance.
        :param img_folder: a path to image folder.
        """
        super(MultimodalDataset, self).__init__()

        self.ds = ds

        if not img_folder.exists() or not img_folder.is_dir():
            raise FileNotFoundError(f"{img_folder} doesn't exist or it is not a folder")

        self.img_folder = img_folder

        # Define the transformation to apply to images
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors and scales [0, 255] to [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # # Normalize to range [-1, 1]
        ])

    def __len__(self):
        return self.ds.shape[0]

    def __getitem__(self, index):
        sample = self.ds.iloc[index]

        label = torch.tensor(sample["macro_category"], dtype=torch.int64)

        # Get TABULAR DATA
        len_o = len(sample["origin"])
        len_d = len(sample["destination"])
        len_m = len(sample["micro_category"])

        assert len_o == 31 # Check the length of one-hot encoding
        assert len_d == 18
        assert len_m == 19
        
        origin_info = torch.tensor(sample["origin"], dtype=torch.float32)
        destination_info = torch.tensor(sample["destination"], dtype=torch.float32)
        micro_category_info = torch.tensor(sample["micro_category"], dtype=torch.float32)
        price_info = torch.tensor([sample["price"]], dtype=torch.float32)
        crypto_price_info = torch.tensor([sample["crypto_price"]], dtype=torch.float32)
        
        tabular_data = torch.cat(
            [
                origin_info,
                destination_info,
                micro_category_info,
                price_info,
                crypto_price_info
            ],
            dim=0
        )

        assert tabular_data.size(0) == 70 # Check the size of the tensors is equal to 70

        # Get IMAGE
        img_path = self.img_folder.joinpath(sample["image_path"])

        if not img_path.exists():
            raise FileNotFoundError(f"{img_path} doens't exist")
        
        img = Image.open(img_path)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        image = self.transform(img)

        return {
            "image": image,
            "tabular": tabular_data,
            "label": label
        }
    
    def calculate_img_mean_std(img_list:list, folder_path:Path) -> tuple[list, list]: #, save_path:Path
        """
        Calculate the mean and standard deviation of dataset images per channel.

        :param img_list: a list containing image paths.
        :param folder_path: a Path where the images are stored.
        :return: a tuple containing the mean per channel in the first element and the standard deviation per channel in the second element.
        """
        pixel_values = list()

        for img in img_list:
            img_path = folder_path.joinpath(img)

            if not img_path.exists():
                raise FileNotFoundError(f"{img_path} doesn't exist")
            
            image = Image.open(img_path).convert("RGB")
            image = np.array(image) / 255.0  # Normalize to range [0, 1]
            pixel_values.append(image)

        pixel_values = np.stack(pixel_values)  # Shape: (N, H, W, C)
        print(pixel_values.shape())

        # Calculate mean and std. axis=(0, 1, 2) to collapse the first three dimensions (N, H, W), so to aggregate all pixel values across the dataset for each channel
        mean = pixel_values.mean(axis=(0, 1, 2))  # Mean per channel (R, G, B)
        std = pixel_values.std(axis=(0, 1, 2))    # Std per channel (R, G, B)

        print(f"Mean: {mean}")
        print(f"Standard Deviation: {std}")
        return mean, std


drug_terms = [
    "cannabidiol",
    "cannabidiolic acid",
    "cannabidivarin",
    "cannabigerol",
    "cannabinol",
    "concentrate",
    "ak47",
    "shake",
    "tetrahydrocannabinol",
    "tetrahydrocannabinolic acid",
    "rick simpson oil",
    "nandrolone phenylpropionate",
    "trenbolone",
    "boldenone",
    "turinabol",
    "dihydrotestosterone",
    "ligandrol",
    "nutrobal",
    "ostarine",
    "human chorionic gonadotropin",
    "human growth hormone",
    "clostebol",
    "nandrolone",
    "androstenedione",
    "dimethyltryptamine",
    "lysergic acid diethylamide",
    "isolysergic acid diethylamide",
    "metaphedrone",
    "mephedrone",
    "nexus",
    "psilacetin",
    "mebufotenin",
    "psilocin",
    "methylenedioxymethamphetamine",
    "amphetamine",
    "methamphetamine",
    "oxycontin",
    "oxycodone",
    "acetylcysteine",
    # Drug name
    "k8",
    "rp15",
    "tramadol",
    "roxycodone",
    "nervigesic",
    "pregabalin",
    "carisoprodol",
    "alprazolam",
    "xanax",
    "anavar",
    "benzodiazepine",
    "cocaine",
    "clenbuterol",
    "benzocaine",
    "clomiphene",
    "crack",
    "marijuana",
    "hashish",
    "nbome",
    "hydroxycontin chloride",
    "ketamine",
    "heroin",
    "adderall",
    "sativa",
    "indica",
    "cookie",
    "mushroom",
    "dihydrocodeine",
    "psilocybin"
]

def initialize_bert_tokenizer() -> BertTokenizer:
    """
    Initialize the BertTokenizer for tokenization and add the domain-specific terms to the dictionary.
    :return: a BertTokenizer instance.
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    print(f"Vocab size before: {len(tokenizer)}")

    new_tokens = [word for word in drug_terms if word not in tokenizer.vocab]

    for word in drug_terms:
        if word in tokenizer.vocab:
            print(word)

    tokenizer.add_tokens(new_tokens)

    print(f"Vocab size after: {len(tokenizer)}")
    print(f"Drug term: {len(drug_terms)}")

    return tokenizer

if __name__ == "__main__":
    img_path = Path(input("Provide the image folder:\n"))

    json_path = Path(input("Provide the dataset path:\n"))

    json_data = pd.read_json(json_path, orient="records", lines=True)

    tokenizer = initialize_bert_tokenizer()

    dataset = MultimodalDataset(json_data, tokenizer, 512, img_path)

    sample_dict = dataset.__getitem__(3)

    assert sample_dict["tabular"].size(0) == 70
    #assert sample_dict["full_text"].size(0) == 512
    #print(sample_dict["mask"].shape)