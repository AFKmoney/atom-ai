# ATOM speech cuts — measured log

Sandbox: d=16, byte-tick, CPU ~52–72 tick/s.
One mechanism per row. **No fluency claim.**

Pass rule: two prompts emit ≥4 Latin letters **and** the strings differ.

| step | cut | train loss | wrap generate (honest) | note |
|------|-----|------------|------------------------|------|
| 8–13k | span-16 | 6.1–6.9 | soup / `T T T` | contract wrong |
| 10k | byte-tick | 2.70 | space / newline | CE works, unigram |
| 20k | last-atom readout | **1.68** | `tis tis tis` | letters exist |
| 24k | + φ oscillatory binding | 5.09→4.67 | `te te te` | more letters, less diversity |
| 28k-r | + replay high-CE | 2.99→**0.008** | `tis`/`pisi` | **quarantine** — ring memorized |
| 28k-e | + efference from 24k | 2.99→4.82 | tail `Je` | first lexical Je |
| 32k | + dentate 2-gram | 2.41→5.37 | wrap all `Je` / `Je tenta` | Je attractor |
| 36k-cont | same cut continue | 2.95→5.54 | `Je Je Je` | worse — do not resume |
| 32k + n-gram anti-repeat | decode only | — | `Je Jestiste` / `Je pe te pe ?` | **best chat**; weights untouched |
| 36k-big | 32k + 223 kB combo | 2.95→**1.61** | `Je pe tisilisi` | best CE since last-atom |
| 39.5k-hf | 36k-big + CATIE FR 180 kB | 4.67→3.49 | `pe Je pe Jis te` | external chat domain |

## Resume

Speech tip for generate: **32k dentate** + n-gram anti-repeat in `generate_packets`.
Train tip for more data: **36k-big** or **39.5k-hf`.
Never resume **28k-replay** or **36k-cont**.

## Local artifacts (not in git)

`.pt` stays out of git (repo ban). On the sandbox:

- `checkpoints/byte_tick/atom_native_step_32000_dentate.pt`
- `checkpoints/byte_tick/atom_native_step_36000_big.pt`
- `checkpoints/byte_tick/atom_native_step_39500_hf.pt`

Corpus: `data/corpus_fr_hf.txt` (~2.2 MB, CATIE everyday-conversations FR).
