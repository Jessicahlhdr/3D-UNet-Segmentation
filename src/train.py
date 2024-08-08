import timeit
from pathlib import Path
import csv

import torch

from configs import load_config
from datasets import get_dataloader
from models import get_model
from utils import get_loss
from utils.metrics import compute_dice
from utils.training import print_logs, train, validate
from utils.checkpoint import load_checkpoint, save_checkpoint

import json

def save_logs(logs, file_path):
    with open(file_path, 'a') as f:
        json.dump(logs, f)
        f.write('\n')  

def main():
    t0 = timeit.default_timer()

    config = load_config()
    print('config:')
    print(config)
    print('')

    out_dir = Path(config.OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(str(out_dir / 'config.yaml'), 'w') as f:
        f.write(str(config))


    train_dataloader = get_dataloader(config, is_train=True)
    val_dataloader = get_dataloader(config, is_train=False)

    model = get_model(config)

    """
    optimizer = torch.optim.SGD(model.parameters(),
                   lr=config.TRAIN.LR,
                   weight_decay=config.TRAIN.WEIGHT_DECAY,
                   momentum=config.TRAIN.OPTIMIZER_SGD_MOMENTUM,
                   nesterov=config.TRAIN.OPTIMIZER_SGD_NESTEROV)
    """

    optimizer = torch.optim.SGD(model.parameters(),
                   lr=config.TRAIN.LR,
                   weight_decay=config.TRAIN.WEIGHT_DECAY,
                   momentum=config.TRAIN.OPTIMIZER_SGD_MOMENTUM,
                   nesterov=config.TRAIN.OPTIMIZER_SGD_NESTEROV)

    p = config.TRAIN.LR_POLY_EXPONENT
    max_epochs = config.TRAIN.EPOCHS
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / max_epochs)**p)

    criterion = get_loss(config)

    logs_path = out_dir / "training_logs.json"

    start_epoch = 0
    best_score = -1.0
    best_val_logs = {}

    # resume from checkpoint if provided
    checkpoint_path = config.TRAIN.CHECKPOINT_PATH
    if checkpoint_path and checkpoint_path != 'none':
        model, optimizer, lr_scheduler, start_epoch, best_score, best_val_logs = load_checkpoint(
            checkpoint_path, model, optimizer, lr_scheduler, start_epoch, best_score, best_val_logs)
        print(f'resume training from {checkpoint_path}')

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        logs = {'epoch': epoch}

        lr = lr_scheduler.get_last_lr()[0]
        print(f"\nEpoch: {epoch} / {config.TRAIN.EPOCHS}, lr: {lr:.9f}")
        logs['lr'] = lr

        print('training...')
        train_logs = train(train_dataloader, model, criterion, optimizer, config)
        print_logs(train_logs)
        logs.update(train_logs)

        # val
        if (epoch + 1) % config.TRAIN.VAL_INTERVAL == 0:
            print('validating...')
            val_logs = validate(val_dataloader, model, criterion, config)
            print_logs(val_logs)
            logs.update(val_logs)

            score = val_logs[config.TRAIN.MAIN_VAL_METRIC]
            if score > best_score:
                # update best score and save model weight
                best_score = score
                best_val_logs = val_logs
                torch.save(model.state_dict(), str(out_dir / 'model_best.pth'))
            print_logs(best_val_logs, prefix='best val scores:')
        else:
            print(f"skip val since val interval is set to {config.TRAIN.VAL_INTERVAL}")

        save_logs(logs, logs_path)

        lr_scheduler.step()

        save_checkpoint(str(out_dir / 'checkpoint_latest.pth'), model, optimizer, lr_scheduler, epoch + 1, best_score,
                        best_val_logs)

    elapsed = timeit.default_timer() - t0
    print('time: {:.3f} min'.format(elapsed / 60.0))

if __name__ == "__main__":
    main()
