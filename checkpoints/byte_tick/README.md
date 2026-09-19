# byte-tick speech checkpoints (d=16)

Not in this tree yet. The GitHub Contents hook used here is text-only;
putting a 1.5 MB pickle through it would corrupt the file.

Push from the box that has git + the files:

```bash
git add checkpoints/byte_tick/atom_native_step_32000_dentate.pt \
        checkpoints/byte_tick/atom_native_step_36000_big.pt \
        checkpoints/byte_tick/atom_native_step_39500_hf.pt
git commit -m "ckpt: d=16 byte-tick speech tips 32k/36k/39.5k"
git push origin main
```

| file | bytes | sha256 | role |
|------|------:|--------|------|
| `atom_native_step_32000_dentate.pt` | 1582447 | `0265da40d13011dc7a2987dc5622c064795fc309b4e6cd3b3f21ee32d11a1e63` | best chat + anti-repeat |
| `atom_native_step_36000_big.pt` | 1581743 | `cc818a4a8aa302a1135c6f706dd9a7e297cd272847203f39816975def3c8405f` | best CE 1.61 |
| `atom_native_step_39500_hf.pt` | 1619343 | `8bf4fb60c627d61017358bf8f3cfab02c4f5aa115e9a539f4894d44ba19a8e19` | CATIE FR domain |
| `atom_native_step_28000_efference.pt` | 1543199 | `cdf1490ef7b86528b545b09b3a3e31c5bcc04c39cb0b2a022cbb58ef6c7d636e` | first lexical Je |

Do **not** add `28000_replay` or `36000_cont`.
