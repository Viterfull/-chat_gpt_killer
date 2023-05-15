import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

def create_masks(eng_batch, kn_batch):

    max_sequence_length = 200
    NEG_INFTY = -1e9
    num_sentences = len(eng_batch)

    look_ahead_mask = torch.full([max_sequence_length, max_sequence_length] , True)
    look_ahead_mask = torch.triu(look_ahead_mask, diagonal=1)
    
    encoder_padding_mask = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_self_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)
    decoder_padding_mask_cross_attention = torch.full([num_sentences, max_sequence_length, max_sequence_length] , False)

    for idx in range(num_sentences):
      
      eng_sentence_length, kn_sentence_length = len(eng_batch[idx]), len(kn_batch[idx])
      eng_chars_to_padding_mask = np.arange(eng_sentence_length + 1, max_sequence_length)
      kn_chars_to_padding_mask = np.arange(kn_sentence_length + 1, max_sequence_length)
      
      encoder_padding_mask[idx, :, eng_chars_to_padding_mask] = True
      encoder_padding_mask[idx, eng_chars_to_padding_mask, :] = True

      decoder_padding_mask_self_attention[idx, :, kn_chars_to_padding_mask] = True
      decoder_padding_mask_self_attention[idx, kn_chars_to_padding_mask, :] = True
      decoder_padding_mask_cross_attention[idx, :, eng_chars_to_padding_mask] = True
      decoder_padding_mask_cross_attention[idx, kn_chars_to_padding_mask, :] = True

    encoder_self_attention_mask = torch.where(encoder_padding_mask, NEG_INFTY, 0)
    decoder_self_attention_mask =  torch.where(look_ahead_mask + decoder_padding_mask_self_attention, NEG_INFTY, 0)
    decoder_cross_attention_mask = torch.where(decoder_padding_mask_cross_attention, NEG_INFTY, 0)

    return encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask = None):
    
    d_k = q.size()[-1]

    scaled = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)

    if mask is not None:

        # print(f"MASK IN NOT NONE, ITS SHAPE: {mask.size()}")

        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)

    attention = F.softmax(scaled, dim=-1)
    out = torch.matmul(attention, v)

    return out, attention

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_sequence_length):

        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):

        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length).reshape(self.max_sequence_length, 1))
        
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        return PE

class SentenceEmbedding(nn.Module):

    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN
    
    def batch_tokenize(self, batch, start_token, end_token):

        def tokenize(sentence, start_token, end_token):

            # print(f"sentence to tokenize: {sentence}")

            # for token in list(sentence):
            #     if (token == 'ｓ'):
            #         print("PLOHOY TOKEN NAYDEN")

            sentence_word_indicies = [self.language_to_index[token] for token in list(sentence)]

            if start_token:
                sentence_word_indicies.insert(0, self.language_to_index[self.START_TOKEN])
            
            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.END_TOKEN])
            
            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(self.language_to_index[self.PADDING_TOKEN])

            return torch.tensor(sentence_word_indicies)

        tokenized = []
        for sentence_num in range(len(batch)):
           tokenized.append( tokenize(batch[sentence_num], start_token, end_token) )
        
        # print(f"TOKENIZED: {tokenized}")
        tokenized = torch.stack(tokenized)

        return tokenized.to(get_device())
    
    def forward(self, x, start_token, end_token): # sentence

        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)
        
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.ln = nn.LayerNorm(d_model, d_model)

    def forward(self, x, mask):

        B, T, C = x.size()
        
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(B, T, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)

        q, k, v = qkv.chunk(3, dim=-1)
        values, attention = scaled_dot_product(q, k, v, mask)

        values = values.permute(0, 2, 1, 3).reshape(B, T, self.num_heads * self.head_dim)
        out = self.ln(values)

        return out

class FeedFoward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.2):

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_model),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):

        return self.net(x)
    
class EncoderBLock(nn.Module):

    def __init__(self):
        
        super().__init__()

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):
    
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = FeedFoward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    
    def forward(self, x, self_attention_mask):

        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)
        residual_x = x.clone()
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)

        return x
    
class SequentialEncoder(nn.Sequential):

    def forward(self, *inputs):

        x, self_attention_mask  = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)

        return x
    

class Encoder(nn.Module):
    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialEncoder(*[EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                                      for _ in range(num_layers)])

    def forward(self, x, self_attention_mask, start_token, end_token):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)
        return x
    
class MultiHeadCrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.kv_layer = nn.Linear(d_model , 2 * d_model)
        self.q_layer = nn.Linear(d_model , d_model)
        self.linear_layer = nn.Linear(d_model, d_model)
    
    def forward(self, x, y, mask):

        batch_size, sequence_length, d_model = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)

        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, d_model)
        
        out = self.linear_layer(values)

        return out

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads, drop_prob):

        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=num_heads)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = FeedFoward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, x, y, self_attention_mask, cross_attention_mask):

        _y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.dropout1(y)
        y = self.layer_norm1(y + _y)

        _y = y.clone()
        y = self.encoder_decoder_attention(x, y, mask=cross_attention_mask)
        y = self.dropout2(y)
        y = self.layer_norm2(y + _y)

        _y = y.clone()
        y = self.ffn(y)
        y = self.dropout3(y)
        y = self.layer_norm3(y + _y)

        return y

class SequentialDecoder(nn.Sequential):

    def forward(self, *inputs):
    
        x, y, self_attention_mask, cross_attention_mask = inputs

        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)
        
        return y
    
class Decoder(nn.Module):

    def __init__(self, 
                 d_model, 
                 ffn_hidden, 
                 num_heads, 
                 drop_prob, 
                 num_layers,
                 max_sequence_length,
                 language_to_index,
                 START_TOKEN,
                 END_TOKEN, 
                 PADDING_TOKEN):
        
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.layers = SequentialDecoder(*[DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob) for _ in range(num_layers)])

    def forward(self, x, y, self_attention_mask, cross_attention_mask, start_token, end_token):

        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)
        
        return y
    
class Transformer(nn.Module):

    def __init__(self, 
                d_model, 
                ffn_hidden, 
                num_heads, 
                drop_prob, 
                num_layers,
                max_sequence_length, 
                trg_vocab_size,
                src_to_index,
                trg_to_index,
                index_to_trg,
                START_TOKEN, 
                END_TOKEN, 
                PADDING_TOKEN
                ):
        
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.trg_vocab_size = trg_vocab_size

        self.trg_to_index = trg_to_index
        self.index_to_trg = index_to_trg

        self.PADDING_TOKEN = PADDING_TOKEN
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        
        self.encoder = Encoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, src_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.decoder = Decoder(d_model, ffn_hidden, num_heads, drop_prob, num_layers, max_sequence_length, trg_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN)
        self.linear = nn.Linear(d_model, trg_vocab_size)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def forward(self, 
                x, 
                y, 
                encoder_self_attention_mask=None, 
                decoder_self_attention_mask=None, 
                decoder_cross_attention_mask=None,
                enc_start_token=False,
                enc_end_token=False,
                dec_start_token=False, 
                dec_end_token=False): 
        
        x = self.encoder(x, encoder_self_attention_mask, start_token=enc_start_token, end_token=enc_end_token)
        out = self.decoder(x, y, decoder_self_attention_mask, decoder_cross_attention_mask, start_token=dec_start_token, end_token=dec_end_token)
        out = self.linear(out)

        return out
    
    def translate(self, src_sentence):
        
        self.eval()
        max_sequence_length = self.max_sequence_length
        device = get_device()
        src_sentence = (src_sentence,)
        trg_sentence = ("",)
        
        for word_counter in range(max_sequence_length):

            encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask=create_masks(src_sentence, trg_sentence)
            predictions = self(src_sentence,
                                    trg_sentence,
                                    encoder_self_attention_mask.to(device), 
                                    decoder_self_attention_mask.to(device), 
                                    decoder_cross_attention_mask.to(device),
                                    enc_start_token=False,
                                    enc_end_token=False,
                                    dec_start_token=True,
                                    dec_end_token=False)
            
            next_token_prob_distribution = predictions[0][word_counter]
            next_token_index = torch.argmax(next_token_prob_distribution).item()
            next_token = self.index_to_trg[next_token_index]
            
            if next_token == self.END_TOKEN:
                break

            trg_sentence = (trg_sentence[0] + next_token, )
        return trg_sentence[0]

