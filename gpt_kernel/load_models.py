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
num_layers = 1
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

ru_eng_translator.load_state_dict(torch.load('/home/grigoriy/killer/chat_gpt_killer/gpt_kernel/models/ru_eng_trans.pth', map_location=torch.device('cpu')))
eng_ru_translator.load_state_dict(torch.load('/home/grigoriy/killer/chat_gpt_killer/gpt_kernel/models/eng_ru_trans.pth', map_location=torch.device('cpu')))

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

device = get_device()

ru_eng_translator.to(device)
eng_ru_translator.to(device)

ru_eng_translator.eval()
eng_ru_translator.eval()

dict_for_models = {'ru->en': ru_eng_translator, 'en->ru': eng_ru_translator}

def translate(s, lend_from, lend_to):
    if not lend_to or not lend_from:
        return "Пожалуйста, выберите язык"
    if lend_from == lend_to:
        return s
    return dict_for_models[lend_from+'->'+lend_to].translate(s)