import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ParserModel(nn.Module):
    
    def __init__(self, embeddings, n_features=36,
        hidden_size=200, n_classes=3, dropout_prob=0.5):

        """ 
        @param embeddings (ndarray): word embeddings (num_words, embedding_size)
        @param n_features (int): number of input features
        @param hidden_size (int): number of hidden units
        @param n_classes (int): number of output classes
        @param dropout_prob (float): dropout probability
        """
        super(ParserModel, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.dropout_prob = dropout_prob
        self.embed_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.embeddings = nn.Parameter(torch.tensor(embeddings))

        W = nn.Parameter(torch.empty(self.n_features * self.embed_size, hidden_size))
        b1 = nn.Parameter(torch.empty(hidden_size))
        self.embed_to_hidden_weight = nn.init.xavier_uniform_(W)
        self.embed_to_hidden_bias = nn.init.uniform_(b1)

        self.dropout = nn.Dropout(p = dropout_prob)

        U = nn.Parameter(torch.empty(hidden_size, n_classes))
        b2 = nn.Parameter(torch.empty(n_classes))
        self.embed_to_logits_weight = nn.init.xavier_uniform_(U)
        self.embed_to_logits_bias = nn.init.uniform_(b2)

        ### END YOUR CODE

    def embedding_lookup(self, w):
        """ 
            @param w (Tensor): input tensor of word indices (batch_size, n_features)

            @return x (Tensor): tensor of embeddings for words represented in w
                                (batch_size, n_features * embed_size)
        """
        x = torch.index_select(self.embeddings, 0, w.reshape(1, -1)[0])
        x = x.reshape(w.shape[0], w.shape[1] * self.embed_size)

        return x


    def forward(self, w):
        """ 
        @param w (Tensor): input tensor of tokens (batch_size, n_features)

        @return logits (Tensor): tensor of predictions (output after applying the layers of the network)
                                 without applying softmax (batch_size, n_classes)
        """
        x = self.embedding_lookup(w)
        h = F.relu(torch.matmul(x, self.embed_to_hidden_weight) + self.embed_to_hidden_bias)
        logits = torch.matmul(h, self.hidden_to_logits_weight) + self.hidden_to_logits_bias

        return logits

    
