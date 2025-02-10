import torch
import torch.nn as nn

# Local import
from .critic import Critic

def initialize_weights(model:nn.Module):
    """
    Initialize the model weights for nn.Conv2d, nn.ConvTranspose2d, and nn.BatchNorm2d layers.

    :param model: a nn.Module instance.
    """
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def gradient_penalty(critic:Critic, real_data, fake_img:torch.Tensor, fake_tabular:torch.Tensor, label:torch.Tensor, device):
    """
    Apply the gradient penalty to the critic.

    :param critic: a Critic instance.
    :param real_data: a batch of real samples.
    :param fake_img: a batch of generated images.
    :param fake_tabular: a batch of generated tabular data.
    :param label: a batch of corresponding label.
    :param device: a device on which run data.?
    :return: the gradient penalty value.
    """
    # Get input dimension
    batch_size, channel, height, width = real_data["image"].shape # (N, C, H, W)
    _, tabular_dim = real_data["tabular"].shape # (N, tabular_dim)


    # Define epsilon value for each modalitys
    img_epsilon = torch.rand((batch_size, 1, 1, 1)).repeat(1, channel, height, width).to(device)
    tab_epsilon = torch.rand((batch_size, 1)).repeat(1, tabular_dim).to(device)

    # Interpolation
    img_interpolation = real_data["image"] * img_epsilon + fake_img * (1 - img_epsilon)
    tab_interpolation = real_data["tabular"] * tab_epsilon + fake_tabular * (1 - tab_epsilon)

    # Calculate critic scores
    mixed_scores = critic(img_interpolation, tab_interpolation, label)

    # Compute gradient per input
    gradients = torch.autograd.grad(
        inputs=[img_interpolation, tab_interpolation],
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores, device=device),
        create_graph=True,
        retain_graph=True
    )

    # Get the gradient for each input
    img_gradient = gradients[0].view(batch_size, -1)
    tab_gradient = gradients[1].view(batch_size, -1)

    # Concatenate all gradients
    all_gradients = torch.cat([img_gradient, tab_gradient], dim=1)

    gradient_norm = all_gradients.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty