# How to train ATOM (byte-tick)

English. Numbers. **No fluency claim.**

Canonical trainer: `tools/run_atom_native.py`.
Not `src/main.py` (legacy GPT-era wrapper). Not `src/training/trainer.py`.

## What you are training

One tick = one UTF-8 byte = one atom.

```
packet_t  →  field α (RK4) + last atom  →  P(byte_{t+1})
```

Loss is next-byte cross-entropy. Payload-copy **off**. MERGE **off**
for this speech line. Replay **off** (it memorized a ring at CE 0.008).

## Data

UTF-8 text. Dialogue format that matches generate wrap:

```
Utilisateur: Qui es-tu ?
Assistant: Je suis Atom.
```

Put a `.txt` (or several shards) somewhere readable. A 2.2 MB French
everyday-chat dump lives locally as `data/corpus_fr_hf.txt` when you
have it. Git does not carry that file.

## Fresh d=16 speech run (CPU)

```bash
cd atom-ai
pip install -e .
export PYTHONPATH=.
export OMP_NUM_THREADS=4

python3 tools/run_atom_native.py \
  --stream --data data/corpus_fr_hf.txt --loop-shards \
  --chunk-bytes 65536 \
  --output-dir checkpoints/byte_tick \
  --checkpoint-name atom_native.pt \
  --steps 20000 \
  --d-model 16 --n-modes 16 --n-atoms-max 64 \
  --max-span-bytes 1 \
  --episode-length 256 \
  --atom-flush-every 64 \
  --field-max-rms 3.0 \
  --field-obligatory-hard \
  --printable-aux-weight 0.0 \
  --field-next-packet-weight 0.0 \
  --field-ignorance-weight 0.0 \
  --learning-rate 5e-5 \
  --surface-learning-rate 1.2e-4 \
  --log-every 1000 \
  --no-enable-merge --no-payload-copy --last-atom-readout --efference-every 10
```

After each save, **copy** to a unique name. Never overwrite the only
good snapshot in place:

```bash
cp checkpoints/byte_tick/atom_native.pt \
   checkpoints/byte_tick/atom_native_step_${STEP}.pt
```

## Resume

```bash
python3 tools/run_atom_native.py \
  --stream --data data/corpus_fr_hf.txt --loop-shards \
  --chunk-bytes 65536 \
  --resume checkpoints/byte_tick/atom_native_step_36000_big.pt \
  --output-dir checkpoints/byte_tick \
  --checkpoint-name atom_native.pt \
  --steps 20000 \
  --d-model 16 --n-modes 16 --n-atoms-max 64 \
  --max-span-bytes 1 \
  --field-obligatory-hard --no-enable-merge --no-payload-copy --last-atom-readout --efference-every 10 \
  --episode-reset --episode-length 2000 --field-contrast-weight 0 \
  --learning-rate 5e-5 --surface-learning-rate 1.2e-4
```

`--episode-reset` (2000 > ~850 saturation steps) trains the readout on
both transient and saturated field regimes — without it the α-MLP
overfits saturated α and cold probes go flat (ligne B++ mirage).
`--field-contrast-weight 0`: the contrast hinge sits satisfied at 0
(bit-identical outputs with/without); dropping it saves 3 surface
forwards per step.

`--d-model` / `--max-span-bytes` must match the checkpoint.

## Probe (honest)

```bash
python3 - <<'PY'
from src.atom_native import AtomNativeModel
m = AtomNativeModel(
    d_model=16, n_modes=16, n_atoms_max=64,
    max_payload_bytes=1, field_max_rms=3.0,
    field_obligatory_hard=True, enable_merge=False, last_atom_readout=True,
)
m.load("checkpoints/byte_tick/atom_native_step_32000_dentate.pt")
m.eval()
for p in ["Bonjour", "Qui es-tu ?", "Merci"]:
    raw = m.generate_packets(
        p, max_packets=48, deterministic=True,
        payload_copy=False, dialogue_wrap=True,
        speech_gate=False, merge_enabled=False,
    )
    print(p, "->", raw.decode("utf-8", "replace"))
PY
```

Pass rule (same as `docs/MEASURE_LOG.md`): two prompts emit ≥4 Latin
letters **and** the strings are not copies.

## Flags that stay off on this line

| flag | why |
|------|-----|
| payload copy | unigram / hist attractor |
| `--enable-merge` | collapses living atoms to 1 |
| replay (`model._replay_on`) | CE 0.008 ring memorization |
| printable-aux / L_ign | falsified as speech unlock |
| HF tokenizer | banned |

Efference (free-run every 10 ticks) is on in the sandbox trainer via
`--efference-every 10`. Leave it if you want train≈generate.

## Scale later

d=16 is the sandbox. Same contract at d=256 / n_modes=64 is a **new
run**, not a merge of the 3.5M payload-copy checkpoints. Those weights
are a different graph.

## Measured so far

`docs/MEASURE_LOG.md`. Best chat: 32k dentate + n-gram anti-repeat.
Best CE: 36k-big (1.61). Not fluent.
