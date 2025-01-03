import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    This class represent the Generator in the WGAN.
    """

    def __init__(self, device, num_classes:int, z_dim:int, img_channels:int, img_size:int, feature_map:int, tabular_dim:list, label_embedding_size:int=128):
        """
        :param device: device on which move the calculation.
        :param num_classes: the number of classes in the dataset.
        :param z_dim: dimension of noisy z vector.
        :param img_channels: number of channels of input images.
        :param img_size: the image size.
        :param feature_map: number of feature map in Convolutional layer.
        :param tabular_dim: a list containing the dimension of categorical features.
        :param label_embedding_size: the label embedding size.
        """
        super(Generator, self).__init__()

        self.device = device
        self.img_size = img_size

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

        return img_output, origin, destination, micro_category, price, crypto_price

import matplotlib.pyplot as plt
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    noise = torch.randn(1, 128).to(device) # (1, 128)
    label = torch.tensor(0, dtype=torch.int64).unsqueeze(0).to(device) # (1, 1)

    tau = 1.0

    gen = Generator(device, num_classes=7, z_dim=128, img_channels=3, img_size=224, feature_map=8, tabular_dim=[31, 18, 19]).to(device) #vocab_size=30581, seq_len=510, d_h=512
    gen.eval()

    
    img_output, origin, destination, micro_category, price, crypto_price = gen(noise, label, tau)

    assert img_output.size(1) == 3
    assert img_output.size(2) == 224
    assert img_output.size(3) == 224

    #assert text_output.size(1) == 510
    #assert text_output.size(2) == 512
    
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

    #print(f"Text:\n{text_output}")

    print(f"Tabular origin:\n{origin}")
    print(f"Tabular destination:\n{destination}")
    print(f"Tabular micro:\n{micro_category}")
    print(f"Tabular price:\n{price}")
    print(f"Tabular crypto:\n{crypto_price}")

if __name__ == "__main__":
    test()