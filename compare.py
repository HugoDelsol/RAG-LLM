from sentence_transformers import util
import numpy as np
import pandas as pd

data_sbert = np.load('embeddings_sbert.npy')
data_mistral = np.load('embeddings_mistral.npy')

print(data_sbert.shape)
print(data_mistral.shape)

cosine_scores_sbert = util.cos_sim(data_sbert, data_sbert)
cosine_scores_mistral = util.cos_sim(data_mistral, data_mistral)

sim_sbert = pd.DataFrame(cosine_scores_sbert.numpy().round(3))
sim_mistral = pd.DataFrame(cosine_scores_mistral.numpy().round(3))

paires = [
    {'label': 'permis construire / conduire', 'type': 'A', 'indices': (0, 1)},
    {'label': 'carte identité / cantine',     'type': 'A', 'indices': (2, 3)},
    {'label': 'ordures / collecte',           'type': 'B', 'indices': (4, 5)},
    {'label': 'horaires / samedi',            'type': 'B', 'indices': (6, 7)},
    {'label': 'permis construire / PLU',      'type': 'C', 'indices': (0, 8)},
    {'label': 'carte cantine / école',        'type': 'C', 'indices': (3, 9)},
]

# A : mêmes mots, sens différent. On attend un score bas.
# B : même sens, mots différents. On attend un score haut.
# C : même thème, sans piège. On attend un score haut.

sbert = np.array(sim_sbert)
mistral = np.array(sim_mistral)

resultats = []

for paire in paires:
    i, j = paire['indices']
    resultats.append({
        'type': paire['type'],
        'label': paire['label'],
        'sbert': sbert[i, j],
        'mistral': mistral[i, j],
    })

print(pd.DataFrame(resultats))
print(pd.DataFrame(sbert))
print(pd.DataFrame(mistral))

print(np.argmax(mistral))

# for i in range(len(sim_sbert)):
#     for j in range(len(sim_mistral[i])):
