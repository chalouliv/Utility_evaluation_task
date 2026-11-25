import random
from pathlib import Path
import numpy as np
import yaml
from src import load_real, model_selection, load_split_synthetic, evaluate_fidelity, evaluate_utility, plot_utility
 
DPATH_ROOT = Path(__file__).parent
FPATH_CONFIG = DPATH_ROOT / "config.yaml"
WORKFLOWS = [
    ("load_real", load_real),
    ("model_selection", model_selection),
    ("load_split_synthetic", load_split_synthetic),
    ("evaluate_fidelity", evaluate_fidelity),
    ("evaluate_utility", evaluate_utility),
    ("plot_utility", plot_utility),
]
 
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
 
def main() -> None:
    config = yaml.safe_load(FPATH_CONFIG.read_text())
    run_config = config["run"]
    dpath_output = DPATH_ROOT / Path(run_config["output_dir"]).expanduser()
    seed_everything(run_config["seed"])
    for name, workflow in WORKFLOWS:
        if not run_config[name]:
            continue
        workflow.run(config, dpath_output)
 
if __name__ == "__main__":
    main()