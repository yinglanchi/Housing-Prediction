import hydra
import logging
import torch
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloader
from trainer import Trainer
from utils.global_utils import log_loss_summary, add_dict
from omegaconf import OmegaConf
from omegaconf.omegaconf import open_dict
import os
from os.path import join as pjoin

from utils.interrupt_handler import InterruptHandler


def process_config(cfg, save=True):
    root_dir = cfg["exp_dir"]
    os.makedirs(root_dir, exist_ok=True)
    
    with open_dict(cfg):
        cfg["device"] = f'cuda:{cfg["cuda_id"]}' if torch.cuda.is_available() else "cpu"
    
    if save:
        yaml_path = pjoin(root_dir, "config.yaml")
        print(f"Saving config to {yaml_path}")
        with open(yaml_path, 'w') as f:
            print(OmegaConf.to_yaml(cfg), file=f)
            
    return cfg


def log_tensorboard(writer, mode, loss_dict, cnt, epoch):
    for key, value in loss_dict.items():
        writer.add_scalar(mode + "/" + key, value / cnt, epoch)
    writer.flush()
    

@hydra.main(config_path="../configs", config_name="config.yaml")
def main(cfg):
    cfg = process_config(cfg)
    
    """ Logging """
    log_dir = cfg["exp_dir"]
    
    logger = logging.getLogger("TrainModel")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(f'{log_dir}/log.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    """ Tensorboard """
    writer = SummaryWriter(pjoin(log_dir, "tensorboard"))
    
    """ DataLoaders """
    train_loader = get_dataloader(cfg, "train")
    # val_loader = get_dataloader(cfg, "val")
    test_loader = get_dataloader(cfg, "test")
    
    """ Trainer """
    input_size = train_loader.dataset[0]["features"].shape[0]
    trainer = Trainer(input_size, cfg, logger)
    start_epoch = trainer.resume()
    
    """ Test """
    def test_all(dataloader, mode, epoch):
        test_loss = {}
        for data in dataloader:
            _, loss_dict = trainer.test(data)
            loss_dict["cnt"] = 1
            add_dict(test_loss, loss_dict)
            
        cnt = test_loss.pop("cnt")
        log_loss_summary(test_loss, cnt,
                         lambda x, y: logger.info(f'{mode} {x} is {y}'))
        log_tensorboard(writer, mode, test_loss, cnt, epoch)
    
    """ Train """
    with InterruptHandler() as h:
        for epoch in range(start_epoch, cfg["total_epoch"]):
            train_loss = {}
            for data in train_loader:
                loss_dict = trainer.update(data)
                loss_dict["cnt"] = 1
                add_dict(train_loss, loss_dict)

            cnt = train_loss.pop("cnt")
            log_loss_summary(train_loss, cnt,
                            lambda x, y: logger.info(f'Train {x} is {y}'))
            log_tensorboard(writer, "train", train_loss, cnt, epoch)
            
            trainer.step_epoch()
            if epoch % cfg["freq"]["epoch_test"] == 0:
                test_all(test_loader, "test", epoch)
            # if epoch % cfg["freq"]["epoch_val"] == 0:
            #     test_all(val_loader, "val", epoch)
                
            if h.interrupted:
                break
    
    trainer.save()
    

if __name__ == "__main__":
    main()
