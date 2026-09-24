from sentence_transformers import util
import numpy as np
import pandas as pd
import json

with open("corpus.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

themes = [item['theme'] for item in data]

data_sbert = np.load('embeddings_sbert.npy')
data_mistral = np.load('embeddings_mistral.npy')

print(data_sbert.shape)
print(data_mistral.shape)

cosine_scores_sbert = util.cos_sim(data_sbert, data_sbert)
cosine_scores_mistral = util.cos_sim(data_mistral, data_mistral)

sbert = np.array(cosine_scores_sbert.numpy())
mistral = np.array(cosine_scores_mistral.numpy())

# sim_sbert = pd.DataFrame(cosine_scores_sbert.numpy())
# sim_mistral = pd.DataFrame(cosine_scores_mistral.numpy())

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

print(pd.DataFrame(sbert).round(2))
print(pd.DataFrame(mistral).round(2))

def score_top_1(matrice, theme_list):
    result = 0

    for i in range(len(matrice)):
        meilleur_score = -1
        meilleur_j = None

        for j in range(len(matrice[i])):
            if j == i: 
                continue 

            if matrice[i, j] > meilleur_score:
                meilleur_score = matrice[i, j]
                meilleur_j = j

        if theme_list[i] == theme_list[meilleur_j]:
            result += 1
            
        print(f"LIGNE = {i} : COLONNE = {meilleur_j}")

    return result

score_sbert = score_top_1(matrice=sbert, theme_list=themes)
score_mistral = score_top_1(matrice=mistral, theme_list=themes)
        
print(f"SBERT : {score_sbert}/{len(themes)} | MISTRAL : {score_mistral}/{len(themes)}")
