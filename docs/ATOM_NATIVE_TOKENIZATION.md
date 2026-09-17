# ATOM-native atomisation — refonte `signal → atome`

Date de l'expérience : 2026-09-13.

## Intention

Le chemin GPT-2 existant était encore un chemin de tokenisation classique :

```text
texte → ID GPT-2 → embedding statique → mapping → atome
```

Ce chemin conserve un vocabulaire de 50 257 classes et traite l'ID comme
l'unité fondamentale. Il ne correspond pas à l'intention du document ATOM :

```text
T(x, c, S) → A
```

La refonte introduit donc un **atomizer**, et non un vocabulaire subword plus
petit :

```text
flux UTF-8 brut
  → observations de bytes et buffer local
  → paquet atomique réversible et contextuel
  → huit propriétés de ToroidalAtom
  → champ persistant / RK4 / interactions / agrégation / abstraction /
    consolidation
```

Le byte est une observation d'entrée, pas une entrée de table sémantique. Le
paquet atomique est l'unité qui entre dans le cœur toroidal.

## Implémentation

### `src/io/atomizer.py`

`Atomizer` segmente le flux en ligne, sans BPE, sans SentencePiece et sans
Hugging Face. Chaque `AtomPacket` contient :

- le `payload` UTF-8 brut, conservé en bytes pour garantir la réversibilité;
- les offsets `start` et `end`;
- la frontière qui a déclenché l'émission;
- un niveau structurel;
- phase et durée du fragment;
- un vecteur de 320 observations déterministes.

Les 320 observations sont :

```text
256 histogrammes de bytes
+ 16 groupes du premier nibble
+ 16 groupes du dernier nibble
+ 16 statistiques locales
+ 16 valeurs de contexte glissant
```

Les statistiques comprennent la longueur, la proportion de blancs, chiffres,
lettres, ponctuation et contrôle, l'entropie, les transitions de classes, la
phase, la longueur précédente et la similarité au contexte précédent.

Le même fragment de surface peut donc produire des observations différentes
selon le flux qui le précède. Le contexte est sérialisable avec l'atomizer.

Les frontières actuelles sont structurelles et déterministes :

- espaces et retours de ligne;
- ponctuation;
- transition lettre/chiffre;
- limite de durée `max_span_bytes`;
- fin de flux.

La limite de durée ne coupe pas un point de code UTF-8. La concaténation des
payloads reproduit exactement les bytes d'origine.

### `src/atom_native.py`

`AtomCompiler` reçoit les 320 observations et produit directement les huit
propriétés :

```text
(r, phi, omega, E, kappa, M, tau, rho)
```

Il n'y a pas de `Embedding(vocab_size=50257)` utilisé par ce chemin.

`AtomNativeModel` réutilise les modules toroidaux existants sans modifier
`src/toroidal/` dans cette refonte :

- `FractalSuperpositionState`;
- `RK4DynamicsEngine`;
- `ToroidalInteraction`;
- `AggregationEngine`;
- `AbstractionEngine`;
- `ConsolidationEngine`.

L'ancien encodeur et l'ancienne tête GPT-compatible sont conservés dans le
checkpoint du cœur pour compatibilité, mais gelés et non utilisés par
l'expérience atom-native.

### Sortie atomique

Pour pouvoir reconstruire la surface sans 50 257 logits, la tête atom-native
prévoit le prochain paquet avec :

```text
1..max_payload_bytes
+ 256 classes byte par position
```

La loss d'une transition est :

```text
loss = cross_entropy(longueur du prochain paquet)
     + cross_entropy(bytes du prochain paquet)
```

Dans l'expérience, `max_payload_bytes=16`. Une transition correspond donc à
un paquet atomique, et non à un token GPT-2.

## Préservation du cœur ATOM

Aucun Transformer, mécanisme d'attention, Q/K/V, séquence aplatie ou reset à
chaque tick n'a été ajouté. Le chemin d'un paquet est :

```text
AtomPacket
  → AtomCompiler
  → ToroidalAtom
  → champ persistant
  → RK4
  → interaction toroidale
  → agrégation / abstraction / consolidation
  → AtomSurfaceHead
```

Le reset utilisé dans le benchmark se produit seulement entre des épisodes
explicites de 64 paquets afin de borner la collection structurelle pendant la
mesure CPU. Ce n'est pas une réinitialisation entre les transitions d'un même
épisode.

## Tests ajoutés

`test/test_atom_native.py` couvre :

1. reconstruction UTF-8 exacte;
2. dimension et nature non-ID des observations;
3. dépendance du même fragment à son contexte précédent;
4. restauration exacte de l'état de l'atomizer;
5. forward, loss, backward et finitude du modèle;
6. sauvegarde/rechargement du checkpoint atom-native.

Le cœur existant est aussi vérifié par `test/core_invariants.py`.

## Limites connues

Cette première refonte est un chemin de recherche atom-native, pas encore une
preuve de qualité linguistique générale :

- les frontières sont déterministes, elles ne sont pas encore apprises par le
  champ;
- les données locales sont un extrait WikiText-2 raw de 3 070 caractères;
- la dimension de smoke test reste `d_model=8`, `n_modes=8`;
- le modèle de surface est entraîné sur des paquets de bytes bornés;
- la génération produite après 500 transitions reste expérimentale et
  partiellement incohérente;
- la collection explicite d'atomes doit encore être étudiée sur des flux
  beaucoup plus longs.

Les résultats complets du run sont dans
[`ATOM_NATIVE_RUN.md`](ATOM_NATIVE_RUN.md) et dans
`checkpoints/atom_native_500/`.
