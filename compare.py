from sentence_transformers import util
import numpy as np
import pandas as pd

data_sbert = np.load('embeddings_sbert.npy')
data_mistral = np.load('embeddings_mistral.npy')

print(data_sbert.shape)
print(data_mistral.shape)

cosine_scores_sbert = util.cos_sim(data_sbert, data_sbert)
cosine_scores_mistral = util.cos_sim(data_mistral, data_mistral)

print(pd.DataFrame(cosine_scores_sbert.numpy().round(2)))
print(pd.DataFrame(cosine_scores_mistral.numpy().round(2)))