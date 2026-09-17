"""Command-line entry point for Toroidal Fractal Intelligence."""
from __future__ import annotations

import argparse
import torch

from .toroidal.model import ToroidalFractalIntelligence
from .training.trainer import ToroidalTrainer
try:
    from .io.tokenizer import ToroidalTokenizer  # legacy; optional
except Exception:  # pragma: no cover
    ToroidalTokenizer = None  # type: ignore
try:
    from .io.data import load_wikitext
except Exception:  # pragma: no cover
    load_wikitext = None  # type: ignore


def create_model(vocab_size=32000, d_model=256, n_modes=256, n_atoms_max=1024, device="cpu"):
    return ToroidalFractalIntelligence(vocab_size, d_model, n_modes, n_atoms_max).to(device)


def train(model, dataset_name="wikitext", dataset_subset="wikitext-2-raw-v1",
          batch_size=32, max_steps=10000, learning_rate=3e-4,
          save_dir="./checkpoints", device="cpu"):
    if dataset_name != "wikitext":
        raise ValueError("The current reference trainer supports dataset='wikitext'. Use load_custom_text/load_from_file for other sources.")
    dataloader = load_wikitext(subset=dataset_subset, seq_len=128)
    dataloader.batch_size = batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    trainer = ToroidalTrainer(model, dataloader, optimizer, save_dir=save_dir)
    print("=" * 60)
    print("TOROIDAL FRACTAL INTELLIGENCE — TRAINING")
    print("=" * 60)
    print(f"Model config: d_model={model.encoder.d_model}, n_modes={model.state.n_modes}, n_atoms_max={model.encoder.n_atoms_max}")
    print(f"Dataset: {dataset_name}/{dataset_subset}")
    print(f"Batch size: {batch_size}")
    print(f"Max steps: {max_steps}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    result = trainer.train(max_steps=max_steps)
    print("=" * 60)
    print("TRAINING COMPLETE")
    print(f"Final loss: {result['final_loss']:.4f}")
    print(f"Best loss: {result['best_loss']:.4f}")
    print(f"Total steps: {result['total_steps']}")
    print(f"Total time: {result['total_time']:.1f}s")
    print("=" * 60)
    return result


def chat(model, tokenizer, prompt, max_length=200, temperature=1.0, top_k=50):
    model.eval()
    prompt_ids = tokenizer.encode(prompt, return_tensor=True)
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_length=max_length, temperature=temperature, top_k=top_k)
    return tokenizer.decode(generated)


def interactive_chat(model, tokenizer):
    print("TOROIDAL FRACTAL INTELLIGENCE — INTERACTIVE MODE")
    print("Type 'quit' to exit, 'status' for model info, 'save' to save")
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "status":
                s = model.get_shared_params_summary()
                print(f"Atoms={s['n_atoms']} Modes={s['n_modes']} d_model={s['d_model']} Coupling={s['coupling_scale']:.4f} Decay={s['energy_decay']:.4f} PhaseSync={s['phase_sync']:.4f}")
            elif user_input.lower() == "save":
                model.save("interactive_model.pt")
                print("Model saved to interactive_model.pt")
            elif user_input:
                print("\nThoughts:", chat(model, tokenizer, user_input))
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    parser = argparse.ArgumentParser(description="Toroidal Fractal Intelligence")
    parser.add_argument("--mode", choices=["train", "chat", "interactive"], default="train")
    parser.add_argument("--dataset", default="wikitext")
    parser.add_argument("--dataset-subset", default="wikitext-2-raw-v1")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-modes", type=int, default=256)
    parser.add_argument("--n-atoms-max", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model = create_model(args.vocab_size, args.d_model, args.n_modes, args.n_atoms_max, args.device)
    if args.checkpoint:
        model.load(args.checkpoint)
    tokenizer = ToroidalTokenizer("gpt2")
    if args.mode == "train":
        train(model, args.dataset, args.dataset_subset, args.batch_size, args.max_steps, args.learning_rate, args.save_dir, args.device)
    elif args.mode == "chat":
        print(f"\nResponse: {chat(model, tokenizer, args.prompt)}")
    else:
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
