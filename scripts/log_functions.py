import torch
import torchvision
from transformers import BertTokenizer # For tokenization
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import skew, kurtosis

# Local import
from config import Config
from metrics_function import get_categorical_mapping, compute_pixel_variance, compute_pairwise_l2_distance, compute_distinct_n, compute_self_bleu

def log_categorical(
  config:Config,
  real_tabular:torch.Tensor,
  fake_origin:torch.Tensor,
  fake_destination:torch.Tensor,
  fake_micro_category:torch.Tensor,
  writer_real:SummaryWriter,
  writer_fake:SummaryWriter,
  step:int
) -> None:
    """
    Log categorical features.

    :param config: a Config instance.
    :param real_tabular: a batch of real tabular data.
    :param fake_origin: a batch of generated origin.
    :param fake_destination: a batch of generated destination.
    :param fake_micro_category: a batch of generated micro-category.
    :param writer_real: a SummaryWriter to log metrics for real scalar features.
    :param writer_fake: a SummaryWriter to log metrics for generated scalar features.
    :param step: the current step.
    """
    # 1) Get the ONE-HOT ENCODING MAPPING for each categorical feature
    origin_mapping = config.get_origin_mapping()
    destination_mapping = config.get_destination_mapping()
    micro_category_mapping = config.get_micro_category_mapping()

    # 2) Convert generated one-hot vectors in their corresponding categorical value
    fake_origin = get_categorical_mapping(fake_origin, origin_mapping)
    fake_destination = get_categorical_mapping(fake_destination, destination_mapping)
    fake_micro_category = get_categorical_mapping(fake_micro_category, micro_category_mapping)

    # 3) Log generated categorical features
    log_text = "Origin\n"
    for category in fake_origin:
        log_text += f"{category}\n"
    writer_fake.add_text("Fake_Tabular/Origin", log_text, global_step=step)

    log_text = "Destination\n"
    for category in fake_destination:
        log_text += f"{category}\n"
    writer_fake.add_text("Fake_Tabular/Destination", log_text, global_step=step)

    log_text = "Micro-category\n"
    for category in fake_micro_category:
        log_text += f"{category}\n"
    writer_fake.add_text("Fake_Tabular/Micro_Category", log_text, global_step=step)

    # REAL DATA
    # Extract categorical features
    origin_info_end = 31
    destination_info_end = origin_info_end + 18
    micro_category_info_end = destination_info_end + 19

    real_origin_info = real_tabular[:, :origin_info_end]  # Shape: (N, 31)
    real_destination_info = real_tabular[:, origin_info_end:destination_info_end]  # Shape: (N, 18)
    real_micro_category_info = real_tabular[:, destination_info_end:micro_category_info_end]  # Shape: (N, 19)

    # 2) Convert one-hot vectors
    real_origin = get_categorical_mapping(real_origin_info, origin_mapping)
    real_destination = get_categorical_mapping(real_destination_info, destination_mapping)
    real_micro_category = get_categorical_mapping(real_micro_category_info, micro_category_mapping)

    # 3) Log real categorical features
    log_text = "Origin\n"
    for category in real_origin:
        log_text += f"{category}\n"
    writer_real.add_text("Real_Tabular/Origin", log_text, global_step=step)

    log_text = "Destination\n"
    for category in real_destination:
        log_text += f"{category}\n"
    writer_real.add_text("Real_Tabular/Destination", log_text, global_step=step)

    log_text = "Micro-category\n"
    for category in real_micro_category:
        log_text += f"{category}\n"
    writer_real.add_text("Real_Tabular/Micro_Category", log_text, global_step=step)

def log_scalar_metrics(tabular_data:torch.Tensor, fake_price:torch.Tensor, fake_crypto_price:torch.Tensor, writer_real:SummaryWriter, writer_fake:SummaryWriter, step:int) -> None:
    """
    Log metrics for scalar features.

    :param tabular_data: a batch of real tabular data.
    :param fake_price: a batch of generated price.
    :param fake_crypto_price: a batch of generated crypto price.
    :param writer_real: a SummaryWriter to log metrics for real scalar features.
    :param writer_fake: a SummaryWriter to log metrics for generated scalar features.
    :param step: the current step.
    """
    # 1) Log histogram
    writer_fake.add_histogram("Fake_Price/Distribution", fake_price, global_step=step)
    writer_fake.add_histogram("Fake_Crypto_Price/Distribution", fake_crypto_price, global_step=step)

    # 2) Compute skew and kurtosis
    price_flat = fake_price.squeeze() # (N, 1) --> (N,)
    price_np = price_flat.numpy()
    price_skew = skew(price_np)
    price_kurtosis = kurtosis(price_np)
    writer_fake.add_scalar("Fake_Price/Skewness", price_skew, global_step=step)
    writer_fake.add_scalar("Fake_Price/Kurtosis", price_kurtosis, global_step=step)

    crypto_price_flat = fake_crypto_price.squeeze() # (N, 1) --> (N,)
    crypto_price_np = crypto_price_flat.numpy()
    crypto_price_skew = skew(crypto_price_np)
    crypto_price_kurtosis = kurtosis(crypto_price_np)
    writer_fake.add_scalar("Fake_Crypto_Price/Skewness", crypto_price_skew, global_step=step)
    writer_fake.add_scalar("Fake_Crypto_Price/Kurtosis", crypto_price_kurtosis, global_step=step)

    # REAL DATA
    # 1) Extract features
    origin_info_end = 31
    destination_info_end = origin_info_end + 18
    micro_category_info_end = destination_info_end + 19
    price_info_end = micro_category_info_end + 1

    real_price_info = tabular_data[:, micro_category_info_end:price_info_end]  # Shape: (N, 1)
    real_crypto_price_info = tabular_data[:, price_info_end:]  # Shape: (N, 1)

    writer_real.add_histogram("Real_Price/Distribution", real_price_info, global_step=step)
    writer_real.add_histogram("Real_Crypto_Price/Distribution", real_crypto_price_info, global_step=step)

    price_flat = real_price_info.squeeze() # (N, 1) --> (N,)
    price_np = price_flat.numpy()
    price_skew = skew(price_np)
    price_kurtosis = kurtosis(price_np)
    writer_real.add_scalar("Real_Price/Skewness", price_skew, global_step=step)
    writer_real.add_scalar("Real_Price/Kurtosis", price_kurtosis, global_step=step)

    crypto_price_flat = real_crypto_price_info.squeeze() # (N, 1) --> (N,)
    crypto_price_np = crypto_price_flat.numpy()
    crypto_price_skew = skew(crypto_price_np)
    crypto_price_kurtosis = kurtosis(crypto_price_np)
    writer_real.add_scalar("Real_Crypto_Price/Skewness", crypto_price_skew, global_step=step)
    writer_real.add_scalar("Real_Crypto_Price/Kurtosis", crypto_price_kurtosis, global_step=step)

def log_img_metrics(real_img, fake_img, writer_real:SummaryWriter, writer_fake:SummaryWriter, step:int) -> None:
    """
    Log image metrics

    :param real_img: a batch of real images.
    :param fake_img: a batch of generated images.
    :param writer_real: a SummaryWriter to log metrics for real images.
    :param writer_fake: a SummaryWriter to log metrics for generated images.
    :param step: the current step.
    """
    # 1) Log a grid of images
    img_grid_real = torchvision.utils.make_grid(real_img, normalize=True)
    img_grid_fake = torchvision.utils.make_grid(fake_img, normalize=True)

    writer_real.add_image("Real_Image/Grid", img_grid_real, global_step=step)
    writer_fake.add_image("Fake_Image/Grid", img_grid_fake, global_step=step)

    # 2) Compute pixel variance
    real_pixel_variance = compute_pixel_variance(real_img)
    fake_pixel_variance = compute_pixel_variance(fake_img)

    writer_real.add_scalar("Real_Image/Pixel_Variance", real_pixel_variance, global_step=step)
    writer_fake.add_scalar("Fake_Image/Pixel_Variance", fake_pixel_variance, global_step=step)

    # 3) Compute pairwise L2 distance
    real_pairwise_l2_distance = compute_pairwise_l2_distance(real_img)
    fake_pairwise_l2_distance = compute_pairwise_l2_distance(fake_img)

    writer_real.add_scalar("Real_Image/Pairwise_L2_Distance", real_pairwise_l2_distance, global_step=step)
    writer_fake.add_scalar("Fake_Image/Pairwise_L2_Distance", fake_pairwise_l2_distance, global_step=step)

def log_txt_metrics(real_text:torch.Tensor, fake_token_distribution:torch.Tensor, tokenizer:BertTokenizer, writer_real:SummaryWriter, writer_fake:SummaryWriter, step:int) -> None:
    """
    Log text metrics

    :param real_text: a batch of real sequence of token-IDs.
    :param fake_token_distribution: the generated probability distribution over vocabulary (N, seq_len, vocab_size)
    :param tokenizer: a BertTokenizer instance.
    :param writer_real: a SummaryWriter to log metrics for real text.
    :param writer_fake: a SummaryWriter to log metrics for generated text.
    :param step: the current step.
    """
    # Log FAKE TEXT metrics

    # 1) Select the generated token-IDs with the highest probability
    predicted_ids = torch.argmax(fake_token_distribution, dim=-1)  # (N, seq_len)
    predicted_tokens = [
        tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens=True) for sequence in predicted_ids
    ]

    # 2) Compute  DISTINCT-1
    distinct_1 = compute_distinct_n(predicted_tokens, n=1)
    writer_fake.add_scalar("Fake_Text/Distinct-1", distinct_1, global_step=step)
    # 3) Compute  DISTINCT-2
    distinct_2 = compute_distinct_n(predicted_tokens, n=2)
    writer_fake.add_scalar("Fake_Text/Distinct-2", distinct_2, global_step=step)

    # 4) Self-BLUE
    self_bleu_value = compute_self_bleu(predicted_tokens)
    writer_fake.add_scalar("Fake_Text/Self-BLUE", self_bleu_value, global_step=step)

    # 5) Compute ENTROPY
    # 5.1) Compute entropy per token
    entropy_per_token = -torch.sum(fake_token_distribution * torch.log(fake_token_distribution + 1e-10), dim=-1) # (N, seq_len)
    # 5.2) Compute mean entropy per sequence
    mean_entropy_per_sequence = torch.mean(entropy_per_token, dim=1)  # Shape: (N,)
    # 5.3) Compute average entropy over batch
    mean_entropy_batch = torch.mean(mean_entropy_per_sequence)
    writer_fake.add_scalar("Fake_Text/Entropy", mean_entropy_batch, global_step=step)

    # Log REAL TEXT
    # 1) Transform each sequence of token-IDs in a sequence of tokens
    real_tokens = [
        tokenizer.convert_ids_to_tokens(sequence, skip_special_tokens=True) for sequence in real_text
    ]

    # Distinct-1 and Distinct-2
    distinct_1 = compute_distinct_n(real_tokens, n=1)
    writer_real.add_scalar("Real_Text/Distinct-1", distinct_1, global_step=step)
    distinct_2 = compute_distinct_n(real_tokens, n=2)
    writer_real.add_scalar("Real_Text/Distinct-2", distinct_2, global_step=step)

    # Self-BLUE
    self_bleu_value = compute_self_bleu(real_tokens)
    writer_real.add_scalar("Real_Text/Self-BLUE", self_bleu_value, global_step=step)