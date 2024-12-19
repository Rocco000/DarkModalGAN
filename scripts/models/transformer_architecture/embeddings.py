import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    This class calculate the embedding of input tokens.
    """

    def __init__(self, d_h:int, vocab_size:int):
        """
        :param d_h: embedding size.
        :param vocab_size: vocabulary size.
        """
        super(InputEmbedding, self).__init__()
        self.d_h = d_h
        self.vocab_size = vocab_size

        # To obtain the corresponding embedding for each word.
        self.embedding = nn.Embedding(vocab_size, d_h)

    def forward(self, x):
        """
        Create an embedding of the input sequence.

        :param x: the sequence of tokens.
        :return: an embedding representing the input sequence.
        """
        # Input: (N, seq_len) --> (N, seq_len, d_h)
        # Multiply by sqrt(d_h) to scale the embeddings according to the paper (Attention is all you need)
        return self.embedding(x) * math.sqrt(self.d_h)
    
class PositionalEncoding(nn.Module):
    """
    This class calculate the positional encoding to add to the input embeddings to also provide to the model position information.
    """

    def __init__(self, d_h:int, seq_len:int, dropout:float):
        """
        :param d_h: embedding size.
        :param seq_len: maximum token length.
        :param dropout: dropout value.
        """
        super(PositionalEncoding, self).__init__()
        self.d_h = d_h
        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_h)
        positional_encoding_matrix = torch.zeros(seq_len, d_h)

        # Create a vector of shape (seq_len). It represents the position inside the input sequence, 'pos' in the paper formula.
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        # Create a vector of shape (d_h) to perform the positional values
        div_term = torch.exp(torch.arange(0, d_h, 2).float() * (-math.log(10000.0) / d_h)) # (d_h / 2)

        # Apply sine to even indices
        positional_encoding_matrix[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_h))

        # Apply cosine to odd indices
        positional_encoding_matrix[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_h))

        # Add a batch dimension to the positional encoding
        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(0) # (1, seq_len, d_h)

        # Register the positional encoding as a buffer. In this way, it is not a learnable parameter and it will be stored along with the state of the model
        self.register_buffer('positional_encoding_matrix', positional_encoding_matrix)

    def forward(self, x):
        """
        Add the position information to the input embedding.

        :param x: the input embedding.
        :return: the input emebedding with position information.
        """
        # Add the positional embedding to the input embedding. requires_grad() to not learn this emebdding because it is fixed.
        x = x + (self.positional_encoding_matrix[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_h)
        return self.dropout(x)