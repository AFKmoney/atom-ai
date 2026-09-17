# ATOM — règles de non-dérive

Ce document est la référence opérationnelle du projet. ATOM n'est pas un
Transformer et ne doit pas être entraîné comme un Transformer.

## Pipeline canonique

```text
TOKEN
  → TOROIDAL ATOM
  → PERSISTENT TOROIDAL FIELD
  → RK4 DYNAMICS
  → LOCAL TOROIDAL INTERACTION
  → AGGREGATION
  → ABSTRACTION
  → CONSOLIDATION
  → PRODUCTION / NEXT-TOKEN LOGITS
```

Chaque token est un tick du système. Pour une séquence `x[0], ..., x[T-1]`,
la relation d'apprentissage est :

```text
x[t] → model(x[t]) → logits[t] → target x[t+1]
```

Le même objet modèle est conservé entre les ticks. Son champ, sa collection
d'atomes, ses abstractions et sa mémoire persistante ne sont pas remis à zéro
entre deux tokens, sauf si une nouvelle expérience demande explicitement un
nouvel état.

## Ce qui est interdit

1. **Aplatir une séquence** avec `reshape(-1)`, `flatten()` ou une opération
   équivalente avant l'appel au modèle.
2. Appeler `model(sequence_entière)` pour produire toutes les prédictions d'une
   séquence. Le chemin canonique appelle `model(row[t:t+1])` pour chaque tick.
3. Remplacer la dynamique RK4, le champ toroidal ou les interactions locales par
   un bloc Transformer, une attention token-token ou une autre architecture
   séquentielle classique.
4. Ajouter `nn.Transformer`, `TransformerEncoder`, `MultiheadAttention`, des
   projections Q/K/V ou une attention softmax dans `src/toroidal/`.
5. Réinitialiser l'état toroidal pour chaque token ou chaque mini-batch sans
   décision explicite de frontière d'épisode.
6. Utiliser un MLP/LSTM/GRU de pensée externe pour contourner la dynamique
   toroidale. L'encodeur appris et la tête de production sont autorisés comme
   traduction token→atome et état→logits; ils ne remplacent pas le pipeline.
7. Prétendre qu'une opération structurelle est exécutée lorsqu'elle n'est que
   prédite comme étiquette. Le chemin actuel crée un nouvel atome par tick;
   les opérations MODIFY/MERGE/SPLIT ne doivent pas être documentées comme
   actives tant qu'elles ne le sont pas réellement.
8. Réintroduire un tokenizer Hugging Face / GPT-2 ou une dépendance `transformers`
   sur le chemin par défaut. L'entrée canonique est l'Atomizer (bytes UTF-8).
   Une dépendance tokenizer ne justifierait jamais l'ajout d'attention au modèle.
9. Utiliser les anciens scripts de benchmark ou de debug supprimés du dépôt.
   Les comparaisons historiques ne définissent pas l'entraînement ATOM.

## Chemin d'entraînement autorisé

Le chemin supporté est :

- `src/io/data.py` pour les données et les séquences;
- `src/training/trainer.py` pour la boucle séquentielle;
- `src/toroidal/model.py` pour le pipeline d'un tick;
- `src/main.py` / `tools/run_atom_native.py` pour le point d'entrée atom-native.

Le trainer accumule les pertes des transitions séquentielles, exécute un seul
`backward()` par batch, puis effectue `optimizer.step()`. Le checkpoint doit
conserver les poids **et** l'état dynamique: champ, atomes, historique,
abstractions et mémoire persistante.

## Ce qu'ATOM est réellement

ATOM peut utiliser des paramètres appris pour transformer un token en propriétés
physiques d'un atome et pour décoder l'état courant en logits. Cela ne fait pas
d'ATOM un Transformer. La mémoire de calcul est la matière structurée
(`r, φ, ω, E, κ, M, τ, ρ`), son champ spectral et son évolution toroidale.
