import torch
import argparse
import yaml
from transformers import GPT2TokenizerFast
from train import UniformCTMC
from model import DiT, DiTConfig


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=False)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    torch.manual_seed(config["rng_seed"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    ctmc = UniformCTMC(config)

    model_config = DiTConfig.from_dict(config["model"])
    model = DiT(model_config) # this model outputs the log of the scores
    model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    with torch.no_grad():
        tokens = ctmc.forward_euler_sample(log_score_model=model, batch_size=2, seq_len=1024, num_steps=1000, device=device)
    texts = tokenizer.batch_decode(tokens)

    print(f"{texts=}")