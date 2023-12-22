import gdown
import pandas as pd
import os
import re
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

def download():
    gdown.download(id="1hUqu1mbFeTEfBvl-7fc56fHFfCSzIktD")
    gdown.extractall("ml1m.zip", "ml1m")

def load_data_frames():
    movies_train = pd.read_csv('ml1m/content/dataset/movies_train.dat', engine='python',
                            sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
    movies_test = pd.read_csv('ml1m/content/dataset/movies_test.dat', engine='python',
                            sep='::', names=['movieid', 'title', 'genre'], encoding='latin-1', index_col=False).set_index('movieid')
    movies_train['genre'] = movies_train.genre.str.split('|')
    movies_test['genre'] = movies_test.genre.str.split('|')

    folder_img_path = 'ml1m/content/dataset/ml1m-images'
    # Train data
    movies_train['id'] = movies_train.index
    movies_train.reset_index(inplace=True)
    movies_train['img_path'] = movies_train.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)
    # Test data
    movies_test['id'] = movies_test.index
    movies_test.reset_index(inplace=True)
    movies_test['img_path'] = movies_test.apply(lambda row: os.path.join(folder_img_path, f'{row.id}.jpg'), axis = 1)

    # Remove entries without images
    movies_train = movies_train[movies_train.img_path.map(lambda x: os.path.exists(x))].copy()
    movies_test = movies_test[movies_test.img_path.map(lambda x: os.path.exists(x))].copy()
    
    # Remove years
    def remove_years(title):
        parts = title.split(' ')
        if parts[-1].isnumeric():
            return ' '.join(parts[:-1])
        return title

    # Move articles to the beginning
    regex = r"(.*), ([^\s\(\)]+)$"
    articles = ['a', 'an', 'the']
    def normalize_articles(title):
        title = remove_years(title)
        match = re.match(regex, title)
        if match is None or match.group(2).lower() not in articles:
            return title
        return match.group(2) + " " + match.group(1)

    movies_train.title = movies_train.title.map(normalize_articles)
    movies_test.title = movies_test.title.map(normalize_articles)
    return movies_train, movies_test

def load_genres():
    with open('ml1m/content/dataset/genres.txt', 'r') as f:
        genres = f.readlines()
        genres = [x.replace('\n','') for x in genres]
    return genres

class MovieLensDataset(Dataset):
    def __init__(self, data, genres, use_title=True, tokenizer=None, transform=None):
        self.data = data
        self.genre2idx = {genre:idx for idx, genre in enumerate(genres)}
        self.use_title = use_title
        self.transform = transform
        if use_title and tokenizer is not None:
            self.tokenizer = tokenizer

    def __getitem__(self, index):
        # Image
        image = cv2.imread(self.data.iloc[index].img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        # Genre
        genre = self.data.iloc[index].genre
        label = self.one_hot_genre(genre)
        if not self.use_title:
            return image, label
        # Title
        encoding = self.tokenizer.encode_plus(
            self.data.iloc[index].title,
            truncation=True,
            add_special_tokens=True,
            max_length=64,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].view(-1)
        attention_mask = encoding['attention_mask'].view(-1)
        return input_ids, attention_mask, image, label

    def one_hot_genre(self, genre):
        genre_vector = np.zeros(len(self.genre2idx))
        for g in genre:
            genre_vector[self.genre2idx[g]] = 1
        genre_tensor = torch.from_numpy(genre_vector).float()
        return genre_tensor

    def __len__(self):
        return len(self.data)
