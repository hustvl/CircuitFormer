import pytorch_lightning as pl
from pytorch_lightning.strategies.ddp import DDPStrategy
import hydra

from model import MInterface
from data import DInterface
from utils import setup_config
from pathlib import Path

@hydra.main(config_path='config', config_name='config')
def main(cfg):
    callbacks = setup_config(cfg)
    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    data_module = DInterface(cfg.data)
    model = MInterface(cfg.model)
    trainer = pl.Trainer(**cfg.trainer,
                         **callbacks,
                         strategy=DDPStrategy(find_unused_parameters=True),)
    trainer.fit(model, data_module)


if __name__ == '__main__':
    main()
