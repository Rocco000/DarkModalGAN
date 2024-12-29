import torch
import torch.nn as nn
import sys
import os

# Add parent_directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local import
from transformer_architecture import InputEmbedding, PositionalEncoding
from transformer_architecture import FeedForwardBlock, MultiHeadAttentionBlock, EncoderBlock
from transformer_architecture import Encoder, EncoderOnly

class Critic(nn.Module):
    """
    This class represent the Critic in the WGAN.
    """

    def __init__(self, num_classes:int, tabular_dim:int, in_channels:int, img_size:int, feature_map:int, vocab_size:int, seq_len:int, d_h:int=512, n_encoder_block:int=6, n_head:int=8, encoder_dropout:float=0.1, d_ff:int=2048, label_embedding_size:int=128):
        """
        :param num_classes: the number of classes in the dataset.
        :param tabular_dim: dimension of the tabular data.
        :param in_channels: number of channels of the input images.
        :param img_size: the image size.
        :param feature_map: number of feature map in Convolutional layer.
        :param vocab_size: the vocabulary size.
        :param d_h: the embedding size.
        :param n_encoder_block: number of encoder block.
        :param n_head: number of head.
        :param encoder_dropout: the dropout value for encoder.
        :param d_ff: output dimension of the first nn.Linear layer in FeedForward block.
        :param label_embedding_size: label embedding size. 
        """
        super(Critic, self).__init__()

        self.img_size = img_size
        self.d_h = d_h

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

        # TEXT DATA
        self.token_embedding_layer = InputEmbedding(d_h, vocab_size)
        self.encoder_only = self.text_block(seq_len + 1, d_h, n_encoder_block, n_head, encoder_dropout, d_ff) # +1 for label embedding

        # FUSION BLOCK
        # Input: 512 + d_h + C * H * W
        self.fusion_layers = nn.Sequential(
            self.fusion_block(512+d_h+4, 512),
            self.fusion_block(512, 256),
            self.fusion_block(256, 128),
            nn.Linear(128, 1)
        )

        # PROJECTION LAYERS
        self.text_project = nn.Linear(label_embedding_size, d_h)
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
    
    def text_block(self, seq_len:int, d_h:int, n_encoder_block:int, n_head:int, dropout:float, d_ff:int) -> EncoderOnly:
        """
        Define a Encoder-Only architecture.

        :param seq_len: the maximum sequence length, namely the maximum token length.
        :param d_h: the embedding size.
        :param n_encoder_block: number of encoder block.
        :param n_head: number of head.
        :param dropout: the dropout value.
        :param d_ff: output dimension of the first nn.Linear layer in FeedForward block.
        :return: a EncoderOnly model.
        """
        positional_layer = PositionalEncoding(d_h, seq_len, dropout)

        encoder_blocks = list()
        for _ in range(n_encoder_block):
            # Crete the Multi-Head Attention (Self-Attention) layer
            encoder_self_attention_block = MultiHeadAttentionBlock(d_h, n_head, dropout)

            # Crete the FeedForward layer
            feed_forward_block = FeedForwardBlock(d_h, d_ff, dropout)

            # Crete the i-th encoder block
            encoder_block = EncoderBlock(d_h, encoder_self_attention_block, feed_forward_block, dropout)

            encoder_blocks.append(encoder_block)

        # Create decoder
        encoder = Encoder(d_h, nn.ModuleList(encoder_blocks))

        encoder_only = EncoderOnly(encoder, positional_layer)

        # Initialize the parameters with Xavier technique
        for p in encoder_only.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        return encoder_only
    
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

    def txt_forward(self, x, mask, label_embedding):
        """
        Process tabular data.

        :param x: a sequence of text embedding.
        :param mask: a mask to hide paddding tokens.
        :param label_embedding: the label embedding.
        :return: hidden state of the encoder.
        """
        # Create token embedding
        # x = self.token_embedding_layer(x) # (N, seq_len) --> (N, seq_len, d_h)

        # assert x.size(1) == 512
        # assert x.size(2) == 512

        # Project label embedding in text embedding space
        txt_label_embedding = self.text_project(label_embedding) # (N, label_embedding_size) --> (N, d_h)

        assert txt_label_embedding.size(1) == self.d_h

        # Add the sequence dimension
        txt_label_embedding = txt_label_embedding.unsqueeze(1) # (N, d_h) --> (N, 1, d_h)

        assert txt_label_embedding.size(1) == 1

        # Add the label embedding at the beginning of the text
        x = torch.cat([txt_label_embedding, x], dim=1) # (N, seq_len+1, d_h)

        assert x.size(1) == 513

        # Extend the mask due to the label embedding
        extended_mask = torch.cat(
            [torch.ones((mask.size(0), 1, 1, 1), device=mask.device, dtype=mask.dtype), mask], dim=3
        )

        assert extended_mask.size(2) == 513

        return self.encoder_only(x, extended_mask) # (N, seq_len, d_h)
    
    def fusion_forward(self, x):
        """
        Apply the last linear layers to provide the prediction.

        :param x: the input feature vector.
        :return: model prediction.
        """
        return self.fusion_layers(x)
    
    def get_text_embedding(self, token_sequence:torch.Tensor) -> torch.Tensor:
        """
        Convert the input sequence of token-IDs in a sequence of embeddings.

        :param token_sequence: a sequence of token-IDs
        :return: a tensor containing the sequence of embeddings.
        """
        return self.token_embedding_layer(token_sequence) # (N, seq_len) --> (N, seq_len, d_h)

    def forward(self, images, text, mask, tabular, labels):
        """
        Predict whether the input is fake or real for that label.

        :param images: a batch of images.
        :param text: a batch of text embedding.
        :param mask: a mask to hide padding tokens.
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

        # TEXT PROCESSING --> Input: (N, seq_len) + (N, label_embedding_size)
        text_output = self.txt_forward(text, mask, label_embedding) # (N, seq_len, d_h)

        # Get the corresponding CLS embedding which represents the entire input
        cls_embedding = text_output[:, 1, :]  # (N, d_h)

        assert cls_embedding.size(1) == self.d_h

        # CLASSIFICATION
        feature_vector = torch.cat([img_output, cls_embedding, tabular_output], dim=1)

        assert feature_vector.size(1) == 512+self.d_h+4

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
    tokenizer = initialize_bert_tokenizer()

    pad_token = torch.tensor([tokenizer.convert_tokens_to_ids("[PAD]")], dtype=torch.int64).to(device)

    text = tokenizer.encode(sample["full_text"], padding="max_length", max_length=512, truncation=True)

    text = torch.tensor(text, dtype=torch.int64).unsqueeze(0).to(device) # (1, seq_len)
    print(f"text shape: {text.shape}")
    mask = (text != pad_token).unsqueeze(0).int().to(device) # (1, 1, seq_len)

    print(f"mask shape: {mask.shape}")
    print(mask)

    label = torch.tensor(4, dtype=torch.int64).unsqueeze(0).to(device) # (1, 1)

    disc = Critic(num_classes=7, tabular_dim=70, in_channels=3, img_size=224, feature_map=8, vocab_size=len(tokenizer), seq_len=512+1).to(device)
    disc.eval()

    text_embedding = disc.get_text_embedding(text) # (1, seq_len) --> (1, seq_len, d_h)
    output = disc(img, text_embedding, mask, tabular_data, label)

    assert output.size(1) == 1

    print(output)

if __name__ == "__main__":
    test()
