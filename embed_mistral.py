from dotenv import load_dotenv
import os
from mistralai.client import Mistral
import sys
import json
import numpy as np

with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

textes = [item['texte'] for item in data ]

load_dotenv()

key = os.getenv("API_KEY")

if not key:
    sys.exit("Aucune cle retrouve")

client = Mistral(api_key=key)

# embeddings_batch_response = client.embeddings.create(
#     model='mistral-embed',
#     inputs=textes
# ) 

# print(embeddings_batch_response)

# np.save('data_embeddings.npy', embeddings_batch_response)

contenu_data_embeddings = np.load('data_embeddings.npy', allow_pickle=True)

# print(contenu_data_embeddings)
# print(len(data))


reponse = contenu_data_embeddings.item()

# print(reponse)

extract = [item.embedding for item in reponse.data]

print(np.array(extract).shape)

np.save('embeddings_mistral.npy', np.array(extract))

contenu_mistral = np.load('embeddings_mistral.npy')

print(contenu_mistral)