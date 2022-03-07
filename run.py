from hydra import compose, initialize

from models.SBASC.main import SBASC
from models.WBASC.main import WBASC

if __name__ == "__main__":
    with initialize(config_path="conf"):
        cfg = compose("config.yaml", overrides=['domain=restaurant5', 'model=WBASC'])

        # Run SBASC
        SBASC(cfg)(load=True)
        print(cfg)
