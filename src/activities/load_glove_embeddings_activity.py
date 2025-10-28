import numpy as np
import json
import os
from temporalio import activity
from dataclasses import dataclass

@dataclass
class LoadGloveEmbeddingsIn:
    glove_file_path: str
    output_path: str
    word_index_path: str
    max_words: int

@dataclass
class LoadGloveEmbeddingsOut:
    embedding_matrix_path: str
    word_index_path: str
    embedding_dim: int
    num_words: int

@activity.defn
async def load_glove_embeddings_activity(data: LoadGloveEmbeddingsIn) -> LoadGloveEmbeddingsOut:
    os.makedirs(os.path.dirname(data.output_path), exist_ok=True)
    
    embedding_matrix_path = data.output_path

    if os.path.exists(embedding_matrix_path) and os.path.exists(data.word_index_path):
        embedding_matrix = np.load(embedding_matrix_path)
        num_words, embedding_dim = embedding_matrix.shape
        return LoadGloveEmbeddingsOut(
            embedding_matrix_path=embedding_matrix_path,
            word_index_path=data.word_index_path,
            num_words=num_words,
            embedding_dim=embedding_dim
        )

    embeddings_index = {}
    with open(data.glove_file_path, encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_dim = len(next(iter(embeddings_index.values())))

    if not os.path.exists(data.word_index_path):
        raise FileNotFoundError(f'did not find required word_index: {data.word_index_path}')
    with open(data.word_index_path, 'r') as f:
        word_index = json.load(f)

    max_words = data.max_words
    num_words = min(max_words, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, embedding_dim), dtype=np.float32)

    for word, i in word_index.items():
        if i >= max_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    np.save(embedding_matrix_path, embedding_matrix)

    return LoadGloveEmbeddingsOut(
        embedding_matrix_path=embedding_matrix_path,
        word_index_path=data.word_index_path,
        num_words=num_words,
        embedding_dim=embedding_dim
    )
