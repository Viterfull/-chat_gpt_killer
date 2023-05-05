import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformer import Transformer

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


class TextDataset(Dataset):

    def __init__(self, english_sentences, kannada_sentences):

        self.english_sentences = english_sentences
        self.kannada_sentences = kannada_sentences

    def __len__(self):
        
        return len(self.english_sentences)

    def __getitem__(self, idx):
        
        return self.english_sentences[idx], self.kannada_sentences[idx]

def is_valid_tokens(sentence, vocab):

    for token in list(set(sentence)):
        
        if token not in vocab:
            return False
        
    return True

def is_valid_length(sentence, max_sequence_length):
    return len(list(sentence)) < (max_sequence_length - 1)

class ModelTrainer():

    def __init__(self):
        
        pass
    
    def train(self, model, train_loader):
        
        criterian = nn.CrossEntropyLoss(ignore_index=trg_to_index[model.PADDING_TOKEN],
                            reduction='none')

        # When computing the loss, we are ignoring cases when the label is the padding token
        for params in transformer.parameters():
            if params.dim() > 1:
                nn.init.xavier_uniform_(params)

        optim = torch.optim.Adam(transformer.parameters(), lr=1e-4)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        model.train()
        model.to(device)
        num_epochs = 10

        for epoch in range(num_epochs):

            print(f"Epoch {epoch}")
            iterator = iter(train_loader)

            for batch_num, batch in enumerate(iterator):

                model.train()
                src_batch, trg_batch = batch
                encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(src_batch, trg_batch)
                optim.zero_grad()

                trg_predictions = model(src_batch,
                                        trg_batch,
                                        encoder_self_attention_mask.to(device), 
                                        decoder_self_attention_mask.to(device), 
                                        decoder_cross_attention_mask.to(device),
                                        enc_start_token=False,
                                        enc_end_token=False,
                                        dec_start_token=True,
                                        dec_end_token=True)
                
                labels = model.decoder.sentence_embedding.batch_tokenize(trg_batch, start_token=False, end_token=True)
                
                loss = criterian(
                    trg_predictions.view(-1, model.trg_vocab_size).to(device),
                    labels.view(-1).to(device)
                ).to(device)
                
                valid_indicies = torch.where(labels.view(-1) == trg_to_index[model.PADDING_TOKEN], False, True)
                loss = loss.sum() / valid_indicies.sum()
                loss.backward()
                optim.step()
                
                #train_losses.append(loss.item())
                
                if batch_num % 100 == 0:
                    
                    print(f"Iteration {batch_num} : {loss.item()}")
                    print(f"English: {src_batch[0]}")
                    print(f"Kannada Translation: {trg_batch[0]}")

                    trg_sentence_predicted = torch.argmax(trg_predictions[0], axis=1)
                    predicted_sentence = ""

                    for idx in trg_sentence_predicted:
                        
                        if idx == trg_to_index[model.END_TOKEN]:
                            break
                        predicted_sentence += index_to_trg[idx.item()]
                        
                    print(f"Target Prediction: {predicted_sentence}")


                    model.eval()
                    trg_sentence = ("",)
                    src_sentence = ("should we go to the mall?",)

                    for word_counter in range(model.max_sequence_length):
                        
                        encoder_self_attention_mask, decoder_self_attention_mask, decoder_cross_attention_mask = create_masks(src_sentence, trg_sentence)
                        predictions = model(src_sentence,
                                                trg_sentence,
                                                encoder_self_attention_mask.to(device), 
                                                decoder_self_attention_mask.to(device), 
                                                decoder_cross_attention_mask.to(device),
                                                enc_start_token=False,
                                                enc_end_token=False,
                                                dec_start_token=True,
                                                dec_end_token=False)
                        next_token_prob_distribution = predictions[0][word_counter] # not actual probs
                        next_token_index = torch.argmax(next_token_prob_distribution).item()
                        next_token = index_to_trg[next_token_index]
                        trg_sentence = (trg_sentence[0] + next_token, )
                        if next_token == model.END_TOKEN:
                            break
                    
                    print(f"Evaluation translation (should we go to the mall?) : {trg_sentence}")
                    print("-------------------------------------------")

        return

if __name__ == "__main__":
    max_sequence_length = 200

    START_TOKEN = '<START>'
    PADDING_TOKEN = '<PADDING>'
    END_TOKEN = '<END>'

    src_file = "english.txt"
    trg_file = "kannada.txt"

    with open(src_file, 'r') as file:
        src_sentences = file.readlines()
    with open(trg_file, 'r') as file:
        trg_sentences = file.readlines()

    trg_vocab = set()
    src_vocab = set()

    for trg_sentence in trg_sentences:
        trg_vocab.update(trg_sentence)

    for src_sentence in src_sentences:
        src_vocab.update(src_sentence)

    trg_vocab = list(trg_vocab)
    src_vocab = list(src_vocab)

    trg_vocab.append(START_TOKEN)
    trg_vocab.append(END_TOKEN)
    trg_vocab.append(PADDING_TOKEN)

    src_vocab.append(START_TOKEN)
    src_vocab.append(END_TOKEN)
    # src_vocab.append(PADDING_TOKEN)

    index_to_trg = {k:v for k,v in enumerate(trg_vocab)}
    trg_to_index = {v:k for k,v in enumerate(trg_vocab)}
    index_to_src = {k:v for k,v in enumerate(src_vocab)}
    src_to_index = {v:k for k,v in enumerate(src_vocab)}

    TOTAL_SENTENCES = 200000
    src_sentences = src_sentences[:TOTAL_SENTENCES]
    trg_sentences = trg_sentences[:TOTAL_SENTENCES]
    src_sentences = [sentence.rstrip('\n').lower() for sentence in src_sentences]
    trg_sentences = [sentence.rstrip('\n') for sentence in trg_sentences]

    valid_sentence_indicies = []

    for index in range(len(trg_sentences)):

        trg_sentence, src_sentence = trg_sentences[index], src_sentences[index]
        if is_valid_length(trg_sentence, max_sequence_length) \
        and is_valid_length(src_sentence, max_sequence_length) \
        and is_valid_tokens(trg_sentence, trg_vocab):
            valid_sentence_indicies.append(index)

    trg_sentences = [trg_sentences[i] for i in valid_sentence_indicies]
    src_sentences = [src_sentences[i] for i in valid_sentence_indicies]

    d_model = 512
    batch_size = 30
    ffn_hidden = 2048
    num_heads = 8
    drop_prob = 0.1
    num_layers = 1
    max_sequence_length = 200
    kn_vocab_size = len(trg_vocab)

    transformer = Transformer(d_model, 
                            ffn_hidden,
                            num_heads, 
                            drop_prob, 
                            num_layers, 
                            max_sequence_length,
                            kn_vocab_size,
                            src_to_index,
                            trg_to_index,
                            START_TOKEN, 
                            END_TOKEN, 
                            PADDING_TOKEN)

    dataset = TextDataset(src_sentences, trg_sentences)

    train_loader = DataLoader(dataset, batch_size)
    iterator = iter(train_loader)

    for batch_num, batch in enumerate(iterator):
        print(batch)
        if batch_num > 3:
            break

    trainer = ModelTrainer()

    trainer.train(transformer, train_loader)