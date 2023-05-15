import torch
from transformer import Transformer

START_TOKEN = '<START>'
PADDING_TOKEN = '<PADDING>'
END_TOKEN = '<END>'

trg_vocab = list(set(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890.,#!@"$;:^|\'-=+_—?<>'))
src_vocab = list(set(' abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZабвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ1234567890.,#!@"$;:^|\'-=+_—?<>'))

trg_vocab.append(START_TOKEN)
trg_vocab.append(END_TOKEN)
trg_vocab.append(PADDING_TOKEN)

src_vocab.append(START_TOKEN)
src_vocab.append(END_TOKEN)
src_vocab.append(PADDING_TOKEN)

index_to_trg = {k:v for k,v in enumerate(trg_vocab)}
trg_to_index = {v:k for k,v in enumerate(trg_vocab)}
index_to_src = {k:v for k,v in enumerate(src_vocab)}
src_to_index = {v:k for k,v in enumerate(src_vocab)}

d_model = 512
batch_size = 50
ffn_hidden = 2048
num_heads = 8
drop_prob = 0.2
num_layers = 2
max_sequence_length = 200
trg_vocab_size = len(trg_vocab)

ru_eng_translator = Transformer(d_model, 
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
                            PADDING_TOKEN)

eng_ru_translator = Transformer(d_model, 
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
                            PADDING_TOKEN)

ru_eng_translator.load_state_dict(torch.load('models/ru_eng_translator.pth'))
eng_ru_translator.load_state_dict(torch.load('models/eng_ru_translator.pth'))