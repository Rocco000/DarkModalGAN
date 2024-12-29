import torch
import torch.nn as nn
import math

class LayerNormalization(nn.Module):
    """
    This class represent the Add & Norm layer in the paper (Attention is all you need).
    It calculate the mean and standard deviation for each sample in the input batch and normalize it.
    """

    def __init__(self, embedding_size:int, eps:float=10**-6):
        """
        :param embedding_size: the embedding size.
        :param eps: it represents the epsilon in the paper formula.
        """
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(embedding_size)) # gamma is a learnable parameter
        self.beta = nn.Parameter(torch.zeros(embedding_size)) # beta is a learnable parameter

    def forward(self, x):
        """
        Normalize the features of each input across the embedding dimension.

        :param x: the input embedding.
        :return: the input x normalized.
        """
        # Input: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)

        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)

        # eps is to prevent dividing by zero or when std is very small
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    

class FeedForwardBlock(nn.Module):
    """
    This class represent the FeedForward layer of the paper (Attention is all you need).
    """

    def __init__(self, d_h:int, d_ff:int, dropout:float):
        """
        :param d_h: the embedding size.
        :param d_ff: output dimension of the first nn.Linear layer.
        :param dropout: dropout value.
        """
        super(FeedForwardBlock, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(d_h, d_ff), # w1 and b1
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fc2 = nn.Linear(d_ff, d_h) # w2 and b2

    def forward(self, x):
        """
        Apply the two fully-connected layer of FeeForward block.

        :param x: the input hidden embedding.
        :return: the processed embedding.
        """
        # Input: (batch, seq_len, d_h)
         
        x= self.fc1(x) # (batch, seq_len, d_ff)
        
        x=self.fc2(x) # (batch, seq_len, d_h)
        return x
    
class ResidualConnection(nn.Module):
    """
    This class represent the skip connection in the Transformer architecture.
    """
    def __init__(self, embedding_size:int, dropout:float):
        """
        :param embedding_size: the embedding size.
        :param dropout: the dropout value.
        """
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(embedding_size)

    def forward(self, x, sublayer):
        """
        Apply the residual connection mechanism, namely add input x to the output of previous_layer(normalization(x)).

        :param x: the input x.
        :param sublayer: the layer before the Add & Norm layer.
        :return: input x + previous_layer(normalization(x)).
        """
        y = self.norm(x)
        y = sublayer(y)
        y = self.dropout(y)
        return x + y

class MultiHeadAttentionBlock(nn.Module):
    """
    This class represent the Multi-Head Attention layer in the paper (Attention is all you need).
    """

    def __init__(self, d_h:int, h:int, dropout:float):
        """
        :param d_h: the embedding size.
        :param h: the number of heads.
        :param dropout: the dropout value.
        """
        super(MultiHeadAttentionBlock, self).__init__()
        self.d_h = d_h # Embedding vector size
        self.h = h # Number of heads

        # Make sure d_h is divisible by h
        assert d_h % h == 0, "d_h is not divisible by h"

        self.d_k = d_h // h # Dimension of vector seen by each head

        self.query_layer = nn.Linear(d_h, d_h, bias=False) # QUERY weight matrix

        self.key_layer = nn.Linear(d_h, d_h, bias=False) # KEY weight matrix

        self.value_layer = nn.Linear(d_h, d_h, bias=False) # VALUE weight matrix

        self.fc1 = nn.Linear(d_h, d_h, bias=False) # Last matrix W_o

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        """
        Apply the attention mechanism, namely: softmax(Q*K_transpose/ sqrt(d_k)) * V

        :param query: the output of multiplication between input embeddings and query matrix.
        :param key: the output of multiplication between input embeddings and query matrix.
        :param value: the output of multiplication between input embeddings and query matrix.
        :param mask: the mask to hide some words.
        :param dropout: a dropout layer.
        """
        # Get head dimension
        d_k = query.shape[-1] 

        # Apply the first part of the formula: Q*K_transpose/ sqrt(d_k)
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply mask to hide some words
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0 therefore for words to hide (such as future words for decoder or padding for encoder)
            attention_scores.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) 

        # Apply dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        

        # Apply the last part of the formula: softmax() * V. Dim: (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores # return also the attention scores which can be used for visualization

    def forward(self, q, k, v, mask):
        """
        Apply the multi-head attention.

        :param q: input embeddings for query matrix.
        :param k: input embeddings for key matrix.
        :param v: input embeddings for value matrix.
        :param mask: to mask some words (their attention score) to not interact with other words.
        """
        # Apply multiplication between embeddings and matrices
        query = self.query_layer(q) # (batch, seq_len, d_h) --> (batch, seq_len, d_h)
        key = self.key_layer(k) # (batch, seq_len, d_h) --> (batch, seq_len, d_h)
        value = self.value_layer(v) # (batch, seq_len, d_h) --> (batch, seq_len, d_h)

        # Get d_k vectors from each matrix (d_k vectors to provide to heads)
        # (batch, seq_len, d_h) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout) # Dim: (batch, h, seq_len, d_k)
        
        # Combine all heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_h)
        # contiguous() to put the in coniguous memory space and to modify it in place
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) 

        # Multiply by W_o
        # (batch, seq_len, d_h) --> (batch, seq_len, d_h)  
        return self.fc1(x)
    
class EncoderBlock(nn.Module):
    """
    This class represent an Encoder block. It is made of: residual connection, Multi-Head Attention layer, Add & Norm layer, residual connection, FeedForward layer, Add & Norm layer.
    """

    def __init__(self, embedding_size:int, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float):
        """
        :param embedding_size: the embedding size.
        :param self_attention_block: a MultiHeadAttentionBlock instance.
        :param feed_forward_block: a FeedForwardBlock instance.
        :param dropout: dropout value.
        """
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(embedding_size, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        """
        Apply encoder layers.

        :param x: input embeddings.
        :param src_mask: a mask to hide padding tokens.
        :return: encoder output.
        """
        # Apply the first residual connection (x and Multi-Head Attention layer, Add & Norm)
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))

        # Apply the second residual connection (x and FeedForward layer, Add & Norm)
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class GANDecoderBlock(nn.Module):
    """
    This class represent the Decoder block for GAN purpose. It is made of: residual connection, Masked-Multi-Head Attention layer, Add & Norm layer, residual connection, FeedForward layer, Add & Norm layer.
    There is no Cross-Multi-Head Attention since there is no encoder part.
    """

    def __init__(self, embedding_size:int, self_attention_block:MultiHeadAttentionBlock, feed_forward_block:FeedForwardBlock, dropout:float):
        """
        :param embedding_size: the embedding size.
        :param self_attention_block: a MultiHeadAttentionBlock instance.
        :param feed_forward_block: a FeedForwardBlock instance.
        :param dropout: the dropout value.
        """
        super(GANDecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(embedding_size, dropout) for _ in range(2)])

    def forward(self, x, tgt_mask):
        """
        Apply decoder layers.

        :param x: the input embeddings. At the beginnig is the (z + label) embedding, then the generated tokens.
        :param tgt_mask: a mask to hide future tokens.
        :return: decoder output.
        """
        # Apply the first residual connection and Multi-Head Attention layer - Add & Norm block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))

        # Apply the second residual connection and FeedForward layer - Add & Norm block
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class ProjectionLayer(nn.Module):
    """
    This class represents the Linear layer at the end of the Transformer architecture. It projects the input embedding to vocabulary.
    """

    def __init__(self, d_h:int, vocab_size:int):
        """
        :param d_h: the embedding size.
        :param vocab_size: the vocabulary size.
        """
        super(ProjectionLayer, self).__init__()

        self.proj = nn.Linear(d_h, vocab_size)

    def forward(self, x):
        """
        Project the input embedding to vocabulary.

        :param x: the decoder output.
        :return: the projection of the input x into vocabulary.
        """
        # Input: (batch, seq_len, d_h) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    

if __name__ == "__main__":
    x = nn.Parameter(torch.ones(1))

    print(x)

    print(x.shape)