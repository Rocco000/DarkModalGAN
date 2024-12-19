import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add parent_directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local import
from transformer_architecture import InputEmbedding, PositionalEncoding
from transformer_architecture import FeedForwardBlock, MultiHeadAttentionBlock, GANDecoderBlock, ProjectionLayer
from transformer_architecture import Decoder, DecoderOnly
from transformer_architecture import causal_mask

class Generator(nn.Module):
    """
    This class represent the Generator in the WGAN.
    """

    def __init__(self, device, num_classes:int, z_dim:int, img_channels:int, img_size:int, feature_map:int, tabular_dim:list, vocab_size:int, seq_len:int, d_h:int=512, n_decoder_block:int=6, n_head:int=8, decoder_dropout:float=0.1, d_ff:int=2048, label_embedding_size:int=128):
        """
        :param device: device on which move the calculation. ??
        :param num_classes: the number of classes in the dataset.
        :param z_dim: dimension of noisy z vector.
        :param img_channels: number of channels of input images.
        :param img_size: the image size.
        :param feature_map: number of feature map in Convolutional layer.
        :param tabular_dim: a list containing the dimension of categorical features.
        :param vocab_size: the vocabulary size.
        :param seq_len: maximum sequence length, namely the maximum token length.
        :param d_h: embedding size.
        :param n_decoder_block: number of decoder block.
        :param n_head: number of head.
        :param decoder_dropout: the dropout value for decoder.
        :param d_ff: output dimension of the first nn.Linear layer in FeedForward block.
        :param label_embedding_size: the label embedding size.
        """
        super(Generator, self).__init__()

        self.device = device
        self.img_size = img_size
        self.seq_len = seq_len

        # Label embedding layer
        self.label_embedding_layer = nn.Embedding(num_classes, label_embedding_size) # to add the sample class to the noisy z vector

        # Generate SHARED LATENT REPRESENTATION
        self.shared_latent_layer = nn.Sequential(
            nn.Linear(z_dim + label_embedding_size, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )

        # TABULAR GENERATION LAYERS
        # Input: (N, latent representation dimension) -> (N, 1024)
        self.tabular_layer1 = self.tabular_block(1024, 512)
        self.tabular_layer2 = self.tabular_block(512, 256)
        self.tabular_layer3 = self.tabular_block(256, 128)
        # One linear layer per tabular feature
        self.origin_head = nn.Linear(128, tabular_dim[0])
        self.destination_head = nn.Linear(128, tabular_dim[1])
        self.micro_category_head = nn.Linear(128, tabular_dim[2])
        self.price_head = nn.Linear(128, 1)
        self.crypto_price_head = nn.Linear(128, 1)

        # IMAGE GENERATION LAYERS
        # Input: (N, latent representation dimension, 1, 1) -> (N, 1024, 1, 1)
        self.img_layer1 = self.img_block(in_channels=1024, out_channels= feature_map*64, kernel_size=4, stride=1, padding=0) # out: 512
        self.img_layer2 = self.img_block(in_channels=feature_map*64, out_channels= feature_map*32, kernel_size=4, stride=2, padding=1) # out: 256
        self.img_layer3 = self.img_block(in_channels=feature_map*32, out_channels= feature_map*16, kernel_size=4, stride=2, padding=1) # out: 128
        self.img_layer4 = self.img_block(in_channels=feature_map*16, out_channels= feature_map*8, kernel_size=4, stride=2, padding=1) # out: 64
        self.img_layer5 = self.img_block(in_channels=feature_map*8, out_channels= feature_map*4, kernel_size=4, stride=2, padding=1) # out: 32
        self.img_layer6 = self.img_block(in_channels=feature_map*4, out_channels= feature_map*2, kernel_size=4, stride=2, padding=1) # out: 16
        self.last_img_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=feature_map*2, out_channels=img_channels, kernel_size=4, stride=2, padding=1), # out: 3
            nn.Tanh() # To normalize image in range [-1, 1] to have a more stable training
        )

        # TEXT GENERATION LAYERS
        self.text_projection = nn.Linear(1024, d_h)
        self.token_embedding_layer = InputEmbedding(d_h, vocab_size)
        self.decoder_only = self.text_block(vocab_size, seq_len+1, d_h, n_decoder_block, n_head, decoder_dropout, d_ff) # +1 to add the label embedding at the beginning of the sentece

    def tabular_block(self, input_dim:int, intermediate_dim:int) -> nn.Sequential:
        """
        Define a nn.Sequential block for tabular modality.

        :param input_dim: input dimension.
        :param intermediate_dim: intermediate dimension.
        :return: a nn.Sequential containing one Linear layer, BatchNorm1d layer, and LeakyReLU activation function.
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
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False), # CHIEDERE PERCHE' FALSE
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def text_block(self, vocab_size:int, seq_len:int, d_h:int, n_decoder_block:int, n_head:int, dropout:float, d_ff:int) -> DecoderOnly:
        """
        Define a Decoder-Only architecture.

        :param vocab_size: the vocabulary size.
        :param seq_len: the maximum sequence length, namely the maximum token length.
        :param d_h: the embedding size.
        :param n_decoder_block: number of decoder block.
        :param n_head: number of head.
        :param dropout: the dropout value.
        :param d_ff: output dimension of the first nn.Linear layer in FeedForward block.
        :return: a DecoderOnly model.
        """
        # Create the positional encoding layer
        positional_layer = PositionalEncoding(d_h, seq_len, dropout)

        # Crete decoder blocks
        decoder_blocks = list()
        for _ in range(n_decoder_block):
            # Crete the Multi-Head Attention (Self-Attention) layer
            decoder_self_attention_block = MultiHeadAttentionBlock(d_h, n_head, dropout)
            
            # Crete the FeedForward layer
            feed_forward_block = FeedForwardBlock(d_h, d_ff, dropout)

            # Crete the i-th decoder block
            decoder_block = GANDecoderBlock(d_h, decoder_self_attention_block, feed_forward_block, dropout)

            decoder_blocks.append(decoder_block)

        # Create decoder
        decoder = Decoder(d_h, nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer = ProjectionLayer(d_h, vocab_size)

        # Create Decoder-only architecture
        decoder_only = DecoderOnly(decoder, positional_layer, projection_layer)

        # Initialize the parameters with Xavier technique
        for p in decoder_only.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return decoder_only

    def tabular_forward(self, latent_vector, tau):
        """
        Generate the tabular data.

        :param latent_vector: the latent vector obtained from self.shared_latent_layer.
        :param tau: tau is the temperature parameter in Gumbel-Softmax function.
        :return: the generated tabular data in this order: (origin, destination, micro-category, price, crypto price).
        """
        x = self.tabular_layer1(latent_vector)
        x = self.tabular_layer2(x)
        x = self.tabular_layer3(x)

        origin_logits = self.origin_head(x) # (N, origin_dim)
        destination_logits = self.destination_head(x) # (N, destination_dim)
        micro_category_logits = self.micro_category_head(x) # (N, micro_category_dim)
        price_value = self.price_head(x) # (N, 1)
        crypto_price_value = self.crypto_price_head(x) # (N, 1)

        origin_one_hot = F.gumbel_softmax(origin_logits, tau=tau, hard=False)
        destination_one_hot = F.gumbel_softmax(destination_logits, tau=tau, hard=False)
        micro_category_one_hot = F.gumbel_softmax(micro_category_logits, tau=tau, hard=False)

        return origin_one_hot, destination_one_hot, micro_category_one_hot, price_value, crypto_price_value

    def img_forward(self, latent_vector):
        """
        Generate image.

        :param latent_vector: the latent vector obtained from self.shared_latent_layer
        :return: the generated image.
        """
        # Reshape latent vector into (N, C, H, W) as an image.
        img_input = latent_vector.unsqueeze(2).unsqueeze(3) # (N, 1024) --> (N, 1024, 1, 1)

        x = self.img_layer1(img_input)
        x = self.img_layer2(x)
        x = self.img_layer3(x)
        x = self.img_layer4(x)
        x = self.img_layer5(x)
        x = self.img_layer6(x)
        x = self.last_img_layer(x) # (N, 3, 256, 256)

        # Reshape the output image: (N, 3, 256, 256) --> (N, 3, self.img_size, self.img_size)
        return F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False) 
    
    def txt_forward(self, latent_vector):
        """
        Generate text.

        :param latent_vector: the latent vector obtained from self.shared_latent_layer
        :return: a list of token-IDs.
        """
        # Projecting latent vector in d_h dimension
        x = self.text_projection(latent_vector) # (N, 1024) --> (N, d_h)

        # Add the sequence dimension
        x = x.unsqueeze(1) # (N, d_h) --> (N, 1, d_h)

        # Define the casual mask
        decoder_mask = causal_mask(x.size(1)).to(self.device)

        # List to store generated token-IDs at each step
        predicted_tokens = list()

        for _ in range(self.seq_len):
            # Get decoder output (hidden state)
            decoder_output = self.decoder_only.generate_text(x, decoder_mask) # (N, seq_len, d_h)

            # Project hidden state in the vocabulary space
            decoder_projection = self.decoder_only.project(decoder_output) # (N, seq_len, vocab_size)

            # Get last position in the sequence
            last_step_logits = decoder_projection[:, -1, :] # (N, vocab_size)

            # Get the predicted token-ID per batch element. Token with highest probability
            __, next_token = torch.max(last_step_logits, dim=-1) # (N,)

            # Add sequence dimension
            next_token = next_token.unsqueeze(1) # (N, 1)

            # Store prediceted token
            predicted_tokens.append(next_token)

            # Transform new token in its corresponding embedding (for each element in the batch)
            token_embedding = self.token_embedding_layer(next_token) # (N, 1) --> (N, 1, d_h)

            # Add new token to the sequence
            x = torch.cat([x, token_embedding], dim=1) # Increases sequence length by 1

            # Update the casual mask
            decoder_mask = causal_mask(x.size(1)).to(self.device)

        # Concatenate all tokens along sequence dimension
        predicted_tokens = torch.cat(predicted_tokens, dim=1) # (N, seq_len)

        return predicted_tokens

    def forward(self, z_vector, labels, tau):
        """
        Generate new samples based on input labels.

        :param z_vector: the noisy z vector.
        :param labels: labels of the generated samples.
        :param tau: tau is the temperature parameter in Gumbel-Softmax function.
        :return: generated samples made of: image, text, and tabular data.
        """
        # Create an embedding for each label. 
        label_embedding = self.label_embedding_layer(labels) # (N, label_embedding_size)

        # Concatenate noisy z vector with label embedding
        z_vector = torch.cat([z_vector, label_embedding], dim=1) # (N, z_dim + label_embedding_size)

        latent_vector = self.shared_latent_layer(z_vector) # (N, 1024)

        # TABULAR GENERATION
        origin, destination, micro_category, price, crypto_price = self.tabular_forward(latent_vector, tau)

        # IMAGE GENERATION
        img_output = self.img_forward(latent_vector) # (N, 3, self.img_size, self.img_size)

        # TEXT GENERATION
        text_output = self.txt_forward(latent_vector) # (N, seq_len)

        return img_output, text_output, origin, destination, micro_category, price, crypto_price

import matplotlib.pyplot as plt
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise = torch.randn(1, 128).to(device) # (1, 128)
    label = torch.tensor(0, dtype=torch.int64).unsqueeze(0).to(device) # (1, 1)

    tau = 1.0

    gen = Generator(device, num_classes=7, z_dim=128, img_channels=3, img_size=224, feature_map=8, tabular_dim=[31, 18, 19], vocab_size=30581, seq_len=510, d_h=512).to(device)
    gen.eval()

    
    img_output, text_output, origin, destination, micro_category, price, crypto_price = gen(noise, label, tau)

    assert img_output.size(1) == 3
    assert img_output.size(2) == 224
    assert img_output.size(3) == 224

    assert text_output.size(1) == 510
    
    assert origin.size(1) == 31
    assert destination.size(1) == 18
    assert micro_category.size(1) == 19

    image_tensor = img_output[0] # Shape: (C, H, W)

    # Rescale the tensor to [0, 1]
    image_tensor = (image_tensor + 1) / 2

    # Convert to NumPy array
    image_np = image_tensor.permute(1, 2, 0).detach().cpu().numpy()  # Shape: (H, W, C)

    # Display the image
    plt.imshow(image_np)
    plt.axis('off')  # Turn off axis
    plt.show()

    print(f"Text:\n{text_output}")

    print(f"Tabular origin:\n{origin}")
    print(f"Tabular destination:\n{destination}")
    print(f"Tabular micro:\n{micro_category}")
    print(f"Tabular price:\n{price}")
    print(f"Tabular crypto:\n{crypto_price}")

if __name__ == "__main__":
    test()