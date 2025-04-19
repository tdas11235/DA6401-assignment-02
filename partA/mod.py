import wandb

api = wandb.Api()
ENTITY = "ch21b108-indian-institute-of-technology-madras"
PROJECT = "da6401-a2-test-2"
runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in runs:
    config = run.config
    if "activation_pair" in config and "activation_pair_str" not in config:
        pair = config["activation_pair"]
        if isinstance(pair, list) and len(pair) == 2:
            pair_str = f"{pair[0]}+{pair[1]}"
            print(f"Updating {run.name}: {pair_str}")
            run.config["activation_pair_str"] = pair_str
            run.update()  # Push changes to W&B
