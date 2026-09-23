import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# with open("corpus.json", 'r', encoding='utf-8') as f:
#     data = json.load(f)

# textes = [item['texte'] for item in data]

# print(textes)

# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# embeddings = model.encode(textes)

# print(embeddings.shape)

# np.save('embeddings_sbert.py', embeddings)

print(np.load('embeddings_sbert.npy'))
contenu_sbert = np.load('embeddings_sbert.npy')

# tensor_model = model.similarity(embeddings1=embeddings, embeddings2=embeddings)

# df = pd.DataFrame(tensor_model.numpy())

# print(df)

# equal = np.dot(embeddings[1], embeddings[2]) / (np.linalg.norm(embeddings[1]) * np.linalg.norm(embeddings[2]))

# print(equal)