# Checkpoints / Points de contrôle

## English

| Path | Size | Notes |
|------|------|-------|
| `atom_native_chat_persist/atom_native.pt` | ~8 MB | **Best chat sample** — ~1.05M steps, `--no-episode-reset`, dialogue corpus |
| `BEST_CHAT.txt` | tiny | Pointer + training metadata |

This is a **demo / research snapshot**, not a fluent assistant. Prefer training further (see `docs/TRAINING.md`) over shipping large weight dumps.

Config highlights: `d_model=64`, `n_modes=64`, `n_atoms_max=512`, `episode_length=512`, `episode_reset=false`, `atom_flush_every=256`.

On load, `energy_decay` is clamped to `[0.3, 0.95]` (see `docs/DECAY_FIX.md`).

## Français

| Chemin | Taille | Notes |
|--------|--------|-------|
| `atom_native_chat_persist/atom_native.pt` | ~8 Mo | **Meilleur échantillon chat** — ~1,05M pas, champ persistant |
| `BEST_CHAT.txt` | petit | Pointeur + méta d'entraînement |

Instantané de démo / recherche, **pas** un assistant fluide. Préférez reprendre l'entraînement (`docs/TRAINING.md`) plutôt que d'embarquer de gros poids.
