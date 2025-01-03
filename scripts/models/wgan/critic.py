import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    This class represent the Critic in the WGAN.
    """

    def __init__(self, num_classes:int, tabular_dim:int, in_channels:int, img_size:int, feature_map:int, label_embedding_size:int=128):
        """
        :param num_classes: the number of classes in the dataset.
        :param tabular_dim: dimension of the tabular data.
        :param in_channels: number of channels of the input images.
        :param img_size: the image size.
        :param feature_map: number of feature map in Convolutional layer.
        :param label_embedding_size: label embedding size. 
        """
        super(Critic, self).__init__()

        self.img_size = img_size

        # Embedding layer for labels.
        self.label_embedding_layer = nn.Embedding(num_classes, label_embedding_size)

        # TABULAR DATA
        # Input: (N, tabular_dim + label embedding)
        self.tabular_layer1 = self.tabular_block(tabular_dim + label_embedding_size, 256)
        self.tabular_layer2 = self.tabular_block(256, 256)
        self.tabular_layer3 = self.tabular_block(256, 512)

        # IMAGE DATA
        # Image input: (N, in_channels + 1, img_size, img_size) -> +1 represents the label embedding
        self.img_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels+1, out_channels=feature_map*2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.img_layer2 = self.img_block(in_channels=feature_map*2, out_channels= feature_map*4, kernel_size=4, stride=2, padding=1)
        self.img_layer3 = self.img_block(in_channels=feature_map*4, out_channels= feature_map*8, kernel_size=4, stride=2, padding=1)
        self.img_layer4 = self.img_block(in_channels=feature_map*8, out_channels= feature_map*16, kernel_size=4, stride=2, padding=1)
        self.img_layer5 = self.img_block(in_channels=feature_map*16, out_channels= feature_map*32, kernel_size=4, stride=2, padding=1)
        self.last_img_layer = nn.Conv2d(in_channels=feature_map*32, out_channels=1, kernel_size=4, stride=2, padding=0)
        self.flatter = nn.Flatten() # To flat the output of Conv2d layer

        # FUSION BLOCK
        # Input: 512 + C * H * W
        self.fusion_layers = nn.Sequential(
            self.fusion_block(512+4, 768),
            self.fusion_block(768, 512),
            self.fusion_block(512, 256),
            self.fusion_block(256, 128),
            nn.Linear(128, 1)
        )

        # PROJECTION LAYERS
        self.img_project = nn.Linear(label_embedding_size, img_size*img_size)

    def tabular_block(self, input_dim:int, intermediate_dim:int) -> nn.Sequential:
        """
        Define a nn.Sequential block for tabular modality.

        :param input_dim: input dimension.
        :param intermediate_dim: intermediate dimension.
        :return: a nn.Sequential containing one Linear layer and LeakyReLU activation function.
        """
        return nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.BatchNorm1d(intermediate_dim),
            nn.LeakyReLU(0.2)
        )

    def img_block(self, in_channels, out_channels, kernel_size, stride, padding) -> nn.Sequential:
        """
        Define a nn.Sequential block for image modality.
        
        :param in_channels: the number of channels of input images
        :param out_channels: the output channels.
        :param kernel_size: the kernel size.
        :param stride: the stride to indicate how much move the filter on the input image.
        :param padding: how many padding to add to the input image.
        :return: a nn.Sequential containing one Conv2d layer, then a BatchNorm layer and LeakyReLU activation function.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True), # affine= True to have learnable parameters
            nn.LeakyReLU(0.2)
        )
    
    def fusion_block(self, input_dim:int, intermediate_dim:int) -> nn.Sequential:
        """
        Define a nn.Sequential block for classification.

        :param input_dim: input dimension.
        :param intermediate_dim: intermediate dimension.
        :return: a nn.Sequential containing one Linear layer, on LayerNorm and LeakyReLU activation function. 
        """
        return nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            nn.LayerNorm(intermediate_dim),
            nn.LeakyReLU(0.2)
        )

    def tabular_forward(self, x, label_embedding):
        """
        Process tabular data.

        :param x: tabular data.
        :param label_embedding: the label embedding
        :return: hidden vector.
        """
        # Add the label embeddding to tabular data
        x = torch.cat([x, label_embedding], dim=1) # (N, tabular_dim + label_embedding_dim)

        assert x.size(1) == 70+128

        x = self.tabular_layer1(x)
        x = self.tabular_layer2(x)
        return self.tabular_layer3(x) # (N, 512)

    def img_forward(self, x, label_embedding):
        """
        Process tabular data.

        :param x: input image.
        :param label_embedding: the label embedding.
        :return: hidden feature map.
        """
        # Project label embedding in image space
        img_label_embedding = self.img_project(label_embedding) # (N, label_embedding_size) --> (N, self.img_size * self.img_size)

        assert img_label_embedding.size(1) == 50176

        # Reshape label embedding: (N, self.img_size * self.img_size) --> (N, 1, self.img_size, self.img_size) so we have (N, C, H, W)
        img_label_embedding = img_label_embedding.view(img_label_embedding.shape[0], 1, self.img_size, self.img_size)

        # Add the label embedding to the image on channel dimension
        x = torch.cat([x, img_label_embedding], dim=1) # (N, C + 1, H, W)

        x = self.img_layer1(x)
        x = self.img_layer2(x)
        x = self.img_layer3(x)
        x = self.img_layer4(x)
        x = self.img_layer5(x)
        return self.last_img_layer(x) # (N, 1, 2, 2)
    
    def fusion_forward(self, x):
        """
        Apply the last linear layers to provide the prediction.

        :param x: the input feature vector.
        :return: model prediction.
        """
        return self.fusion_layers(x)

    def forward(self, images, tabular, labels):
        """
        Predict whether the input is fake or real for that label.

        :param images: a batch of images.
        :param tabular: a batch of tabular data.
        :param labels: a batch of corresponding labels.
        :return: the model prediction.
        """
        # Create an embedding for each label
        label_embedding = self.label_embedding_layer(labels)

        assert label_embedding.size(1) == 128

        # TABULAR PROCESSING --> Input: (N, tabular_dim) + (N, label_embedding_size)
        tabular_output = self.tabular_forward(tabular, label_embedding) # (N, 512)

        assert tabular_output.size(1) == 512

        # IMAGE PROCESSING --> Input: (N, C, H, W) + (N, label_embedding_size)
        img_output = self.img_forward(images, label_embedding) # (N, 1, 2, 2)
        img_output = self.flatter(img_output) # Flat the image branch output: (N, C*H*W)

        assert img_output.size(1) == 4

        # CLASSIFICATION
        feature_vector = torch.cat([img_output, tabular_output], dim=1)

        assert feature_vector.size(1) == 512+4

        return self.fusion_forward(feature_vector)


from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from transformers import BertTokenizer

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

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    json_path = Path(input("Provide the dataset path:\n"))

    json_data = pd.read_json(json_path, orient="records", lines=True)

    sample = json_data.iloc[0] # Get first sample

    # Read tabular
    origin_info = torch.tensor(sample["origin"], dtype=torch.float32)
    destination_info = torch.tensor(sample["destination"], dtype=torch.float32)
    micro_category_info = torch.tensor(sample["micro_category"], dtype=torch.float32)
    price_info = torch.tensor([sample["price"]], dtype=torch.float32)
    crypto_price_info = torch.tensor([sample["crypto_price"]], dtype=torch.float32)
    
    tabular_data = torch.cat(
        [
            origin_info,
            destination_info,
            price_info,
            crypto_price_info,
            micro_category_info
        ],
        dim=0
    ).unsqueeze(0).to(device) # (1, tabular_dim)

    # Read image
    img_path = Path(input("Provide the image path:\n"))
    transformer = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors and scales [0, 255] to [0, 1]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # # Normalize to range [-1, 1]
    ])
    img = transformer(Image.open(img_path)).unsqueeze(0).to(device) # (1, 3, 224, 224)

    # Read Text
    """tokenizer = initialize_bert_tokenizer()

    pad_token = torch.tensor([tokenizer.convert_tokens_to_ids("[PAD]")], dtype=torch.int64).to(device)

    text = tokenizer.encode(sample["full_text"], padding="max_length", max_length=512, truncation=True)

    text = torch.tensor(text, dtype=torch.int64).unsqueeze(0).to(device) # (1, seq_len)
    print(f"text shape: {text.shape}")
    mask = (text != pad_token).unsqueeze(0).int().to(device) # (1, 1, seq_len)

    print(f"mask shape: {mask.shape}")
    print(mask)"""

    label = torch.tensor(4, dtype=torch.int64).unsqueeze(0).to(device) # (1, 1)

    disc = Critic(num_classes=7, tabular_dim=70, in_channels=3, img_size=224, feature_map=8).to(device) # vocab_size=len(tokenizer), seq_len=512+1
    disc.eval()

    #text_embedding = disc.get_text_embedding(text) # (1, seq_len) --> (1, seq_len, d_h)
    output = disc(img, tabular_data, label)

    assert output.size(1) == 1

    print(output)

if __name__ == "__main__":
    test()
