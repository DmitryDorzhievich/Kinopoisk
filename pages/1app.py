import torch
import nltk
nltk.download('stopwords')

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import re
import string
from collections import Counter

from nltk.corpus import stopwords

import streamlit as st

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torchutils as tu

from dataclasses import dataclass
russian_stopwords = stopwords.words("russian")

from dataclasses import dataclass
import json

with open('vocab_to_int.json', 'r') as json_file:
    json_data = json_file.read()

vocab_to_int = json.loads(json_data)

def data_preprocessing(text: str) -> str:
    """preprocessing string: lowercase, removing html-tags, punctuation and stopwords

    Args:
        text (str): input string for preprocessing

    Returns:
        str: preprocessed string
    """

    text = text.lower()
    text = re.sub("<.*?>", "", text)  # html tags
    text = "".join([c for c in text if c not in string.punctuation])
    splitted_text = [word for word in text.split() if word not in russian_stopwords]
    text = " ".join(splitted_text)
    return text

def padding(review_int: list, seq_len: int) -> np.array:
    """Make left-sided padding for input list of tokens

    Args:
        review_int (list): input list of tokens
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros

    Returns:
        np.array: padded sequences
    """
    features = np.zeros((36591, seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[:seq_len]
        features[i, :] = np.array(new)

    return features

def preprocess_single_string(
        input_string: str, 
        seq_len: int, 
        vocab_to_int: dict = vocab_to_int
        ) -> torch.Tensor:
    """Function for all preprocessing steps on a single string

    Args:
        input_string (str): input single string for preprocessing
        seq_len (int): max length of sequence, it len(review_int[i]) > seq_len it will be trimmed, else it will be padded by zeros
        vocab_to_int (dict, optional): word corpus {'word' : int index}. Defaults to vocab_to_int.

    Returns:
        list: preprocessed string
    """    

    preprocessed_string = data_preprocessing(input_string)
    result_list = []
    for word in preprocessed_string.split():
        try: 
            result_list.append(vocab_to_int[word])
        except KeyError as e:
            print(f'{e}: not in dictionary!')
    result_padded = padding([result_list], seq_len)[0]

    return torch.Tensor(result_padded)

class RNNNet(nn.Module):    
    '''
    vocab_size: int, размер словаря (аргумент embedding-слоя)
    emb_size:   int, размер вектора для описания каждого элемента последовательности
    hidden_dim: int, размер вектора скрытого состояния, default 0
    batch_size: int, размер batch'а

    '''
    
    def __init__(
            self, 
            vocab_size: int, 
            emb_size: int,
            hidden_dim: int, 
            seq_len: int, 
            n_layers: int = 1
                ) -> None:
        super().__init__()
        
        self.seq_len  = seq_len 
        self.emb_size = emb_size 
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.rnn_cell  = nn.RNN(
            input_size  = self.emb_size, 
            hidden_size = self.hidden_dim, 
            batch_first = True, 
            num_layers  = n_layers
            )
        self.linear    = nn.Sequential(

            nn.Linear(self.hidden_dim * self.seq_len, 16),
            nn.Tanh(),
            nn.Linear(16, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self.input = x.size(0)
        x = self.embedding(x.to(rnn_conf.device))
        output, _ = self.rnn_cell(x) 
        # print(f'RNN output: {output.shape}')
        output = output.contiguous().view(output.size(0), -1)
        # print(f'Flatten output: {output.shape}')
        out = self.linear(output.squeeze(0))
        return out

SEQ_LEN = 100

@dataclass
class ConfigRNN:
    vocab_size: int
    device : str
    n_layers : int
    embedding_dim : int
    hidden_size : int
    seq_len : int

rnn_conf = ConfigRNN(
    vocab_size = len(vocab_to_int)+1,
    device='cpu',
    n_layers=1,
    embedding_dim=8, 
    hidden_size=16,
    seq_len = SEQ_LEN
)

rnn_model = RNNNet(
    vocab_size=rnn_conf.vocab_size,
    emb_size=rnn_conf.embedding_dim,
    hidden_dim=rnn_conf.hidden_size,
    seq_len=rnn_conf.seq_len,
    n_layers=rnn_conf.n_layers

)

rnn_model.load_state_dict(torch.load("weights.pt"))


result = {1: "Нейтральный", 2: "Положительный", 0: "Отрицательный" }

rnn_model.eval()
probability = rnn_model(preprocess_single_string('Сказать, что я разочарован — ничего не сказать. Сценаристу за адаптацию шедевральной книги Р. Д. Уоллера надо что-нибудь оторвать. Нелогичные поступки, из-за невозможности перенести на экран мысли людей (не можешь не берись, на самом деле). Важные мысли и сцены из книги убраны, добавлены новые, ни к селу. Все хорошее в сценарии — из книги. Все нелепое — от сценариста. Да и затянуть до 2,15 часа короткую книгу тоже не лучший ход.\n\nК кастингу тоже вопросы. Мэрис Стрип прекрасная актриса, но Франческа — итальянка, и Стрип пришлось прибегать к ужимкам в стиле Маргариты Тереховой, что меня просто коробило. С Иствудом отдельная тема. При прочтении именно Иствуд («Роберт был высокий, худой и сильный, а двигался, как трава под ветром, плавно, без усилий. Серебристо-седые волосы прикрывали уши и шею, и, надо сказать, выглядел он всегда слегка растрепанным, как будто только что сошел на землю после путешествия по бурному морю и пытался ладонью привести волосы в порядок. Узкое лицо, высокие скулы и лоб, наполовину прикрытый волосами, на фоне которых голубые глаза смотрелись особенно ярко.») выглядел идеальным актером на роль Кинкейда. Но вот беда — Иствуд постарел. Играть в 65 пятидесятилетнего мужчину нелегко. Лет десять назад было бы намного лучше.\n\nНу и режиссура. Слабо, к сожалению. Очень поверхностно. Получилась простенькая мелодрама. А жаль, книга более чем достойная.', seq_len=SEQ_LEN).unsqueeze(0).long().to(rnn_conf.device)).sigmoid()
print(probability)
print(f'{result[torch.argmax(probability).item()]} Вероятность: {probability.max():.3f}')

def main():
    # Заголовок приложения
    st.title("Модель обработки текста")

    # Ввод предложения от пользователя
    input_text = st.text_input("Введите предложение:", "")

    # Обработка входных данных через модель
    rnn_model.eval()
    probability = rnn_model(preprocess_single_string(input_text, seq_len=SEQ_LEN).unsqueeze(0).long().to(rnn_conf.device)).sigmoid()
    # Вывод результатов
    st.write(f'{result[torch.argmax(probability).item()]} Вероятность: {probability.max():.3f}')
    print(f'{result[torch.argmax(probability).item()]} Вероятность: {probability.max():.3f}')

if __name__ == '__main__':
    main()