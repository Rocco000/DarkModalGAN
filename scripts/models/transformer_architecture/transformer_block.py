import torch.nn as nn
from transformer_architecture import InputEmbedding, PositionalEncoding
from transformer_architecture import LayerNormalization, ProjectionLayer

class Decoder(nn.Module):
    """
    This class represents the Decoder in Transformer architecture.
    """

    def __init__(self, embedding_size:int, layers:nn.ModuleList):
        """
        :param embedding_size: the embedding size.
        :param layers: a list of GANDecoderBlock.
        """
        super(Decoder, self).__init__()

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(embedding_size)

    def forward(self, x, tgt_mask):
        """
        Apply all decoder blocks.

        :param x: the input embeddings. At the beginnig is the (z + label) embedding, then the generated tokens.
        :param tgt_mask: a mask to hide future tokens.
        :return: a hidden state that should be project in vocabulary dimension to get the next token of the sequence.
        """
        for layer in self.layers:
            x = layer(x, tgt_mask)
        
        return self.norm(x)
    
class DecoderOnly(nn.Module):
    """
    This class represents the Decoder-only architecture.
    """

    def __init__(self, decoder:Decoder, tgt_pos:PositionalEncoding, projection_layer:ProjectionLayer):
        """
        :param decoder: a Decoder instance.
        :param tgt_pos: a PositionalEncoding instance to add position information to embeddings.
        :param projection_layer: a ProjectionLayer instance to convert the final embeddings in the corresponding words in vocabulary.
        """
        super(DecoderOnly, self).__init__()

        self.decoder = decoder
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def generate_text(self, x, tgt_mask):
        """
        Generate text.

        :param x: the input embeddings. At the beginnig is the (z + label) embedding, then (z + label) + the generated tokens.
        :param tgt_mask: the mask to hide futere tokens.
        :return: the hidden state of the decoder that should be project in vocabulary dimension to get the next token of the sequence..
        """
    
        # Apply the positional encoding to add the position information
        x = self.tgt_pos(x)

        return self.decoder(x, tgt_mask)

    def project(self, x):
        """
        Project the input embedding in the vocabulary space.

        :param x: the input embeddings.
        :return: the projection of the input embedding in vocabulary dimension to get its corresponding words.
        """
        # (batch, seq_len, d_h) --> (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    
class Encoder(nn.Module):
    """
    This class represents the Encoder in Transformer architecture.
    """

    def __init__(self, embedding_size:int, layers:nn.ModuleList):
        """
        :param embedding_size: the embedding size.
        :param layers: a list of EncoderBlock.
        """
        super(Encoder, self).__init__()

        self.layers = layers
        self.norm = LayerNormalization(embedding_size)

    def forward(self, x, src_mask):
        """
        :param x: input embeddings.
        :param src_mask: a mask to hide pagging tokens.
        :return: encoder output.
        """

        for layer in self.layers:
            x = layer(x, src_mask)

        return self.norm(x)
    
class EncoderOnly(nn.Module):
    """
    This class represents the Encoder-Only architecture.
    """

    def __init__(self, encoder:Encoder, positional_layer:PositionalEncoding):
        """
        :param encoder: a Encoder instance.
        :param positional_layer: a PositionalEncoding instance.
        """
        super(EncoderOnly, self).__init__()

        self.encoder = encoder
        self.positional_layer = positional_layer

    def forward(self, x, src_mask):
        """
        Process input embeddings.
        
        :param x: the input embeddings.
        :param src_mask: a mask to hide the padding tokens.
        :return:
        """
        # Add position information
        x = self.positional_layer(x)

        return self.encoder(x, src_mask)