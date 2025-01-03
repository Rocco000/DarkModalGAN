import torch
from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def get_categorical_mapping(batch:torch.Tensor, mapping:dict) -> list:
    """
    Convert the input batch of one-hot vectors into their corresponding categorical values.

    :param batch: a batch of generated categorical feature.
    :param mapping: the mapping dictionary containing the mapping between the one-hot vector and the corresponding categorical value.
    :return: a list containing the corresponding categorical values.
    """
    categorical_values = list()
    for sample in batch:
        # 1) Get the index of the maximum value
        idx = torch.argmax(sample)
        # 2) Construct the one-hot vector
        one_hot_vector = [1 if i == idx else 0 for i in range(batch.size(1))]
        # 3) Get the corresponding categorical value
        categorical_value = mapping.get(tuple(one_hot_vector), "Unknown")

        if categorical_value == "Unknown":
            print(f"One-hot vector {one_hot_vector} not found in the mapping.")

        categorical_values.append(categorical_value)
    
    return categorical_values

# IMAGE
def compute_pixel_variance(img_batch:torch.Tensor) -> float:
    """
    Compute the pixel variance.

    :param img_batch: a batch of images.
    :return: the pixel variance
    """
    # 1) Compute variance along the batch dimension
    pixel_variance = torch.var(img_batch, dim=0)
    # 2) Compute average variance across all pixels
    return pixel_variance.mean().item()

def compute_pairwise_l2_distance(img_batch:torch.Tensor) -> float:
    """
    Compute the pairwise L2 distance.

    :param img_batch: a batch of images.
    :return: the pairwise L2 distance
    """
    # 1) Flat images: (N, C, H, W) --> (N, C*H*W)
    flattened_images = img_batch.view(img_batch.size(0), -1)
    # 2) Pairwise L2 distance
    pairwise_distances = torch.cdist(flattened_images, flattened_images)
    return pairwise_distances.mean().item()

# TEXT
def compute_distinct_n(batch:list, n:int=1) -> float:
    """
    Compute the Distinct-N score.

    :param batch: a batch of tokenized senteces.
    :param n: the n-gram size.
    :return: the Distinct-N score.
    """
    all_ngrams = list() # To store all N-grams of the batch
    total_ngrams = 0 # Total number of N-grams across all senteces

    for sentence in batch:
        # 1) Generate N-grams for the sentence
        ngram_list = list(ngrams(sentence, n))

        all_ngrams.extend(ngram_list)
        total_ngrams += len(ngram_list)

    # 2) Compute unique N-grams
    unique_ngrams = set(all_ngrams)

    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0

def compute_self_bleu(batch:list) -> float:
    """
    Compute Self-BLUE.

    :param batch: a batch of tokenized senteces.
    :return: the Self-BLUE score.
    """
    smoothing = SmoothingFunction().method1  # Smoothing for short sentences
    total_bleu = 0.0

    for i, reference in enumerate(batch):
        # Treat the current sentence as the reference
        candidates = [candidate for j, candidate in enumerate(batch) if j != i]

        # Compute BLEU for the reference against each candidate
        candidate_bleu = [sentence_bleu([reference], candidate, smoothing_function=smoothing) for candidate in candidates]
        total_bleu += sum(candidate_bleu) / len(candidate_bleu)

    return total_bleu / len(batch) if len(batch) > 0 else 0.0