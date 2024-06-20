import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import copy
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time
from torch import nn
from models import Network, Resi, Resi2, ResiBig, Mobi, Mobi2
from torchsummary import summary
from torch import optim
from torch.optim import lr_scheduler
import wandb
from dataset import setup_datasets
from inference import evalontest
from time import perf_counter as ctime
import os
import logging
import threading
from logging import Filter
import json
import argparse


class ThreadLogFilter(logging.Filter):
    """
    This filter only show log entries for specified thread name
    """

    def __init__(self, thread_name, *args, **kwargs):
        logging.Filter.__init__(self, *args, **kwargs)
        self.thread_name = thread_name

    def filter(self, record):
        return record.threadName == self.thread_name



def single_epoch(epoch:int, num_epochs, early_stopping_epochs, model, optimizer, criterion, scheduler, dataloaders, datasets_sizes, losses_save_name, save_prefix, device, iswandb, metrics: dict, best_model_wts, epochs_no_improvement, isbroken, best_loss, perform_scheduler_step=False):
    stime = ctime()
    logging.info(f'Epoch {epoch}/{num_epochs - 1}' + "\t learning rate: " + str(scheduler.get_last_lr()[0]))
    logging.info('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'valid']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = outputs

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
        if perform_scheduler_step and phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / datasets_sizes[phase]
        logging.info(f'{phase} Loss: {epoch_loss:.4f}')
        if phase == 'train':
            with open(os.path.join("models", losses_save_name), "a") as f:
                f.write(str(epoch_loss) + ';')
            last_train_loss = epoch_loss
            if iswandb:
                wandb.log({"train_loss": epoch_loss})

        if phase == "valid":
            with open(os.path.join("models", losses_save_name), "a") as f:
                f.write(str(epoch_loss) + "\n")
            if iswandb:
                wandb.log({"valid_loss": epoch_loss})
            if epoch_loss < best_loss:
                logging.info("best valid loss!")
                best_loss = epoch_loss
                metrics['train_loss_at_min_valid'] = last_train_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join("models", save_prefix + "_best_weights.pt"))
                epochs_no_improvement = 0
            else:
                epochs_no_improvement += 1
        if epochs_no_improvement == early_stopping_epochs:
            isbroken = True
            break
    etime = ctime()
    logging.info(f"epoch time taken: {etime - stime}")
    logging.info("")
    return model, optimizer, scheduler, metrics, best_model_wts, epochs_no_improvement, isbroken, best_loss


def train_model(model, experiment_name, dataloaders, criterion, optimizer, scheduler, num_epochs=2000, iswandb=False, device=("cuda" if torch.cuda.is_available() else "cpu"), early_stopping_epochs=100, num_epochs_no_stepLR=1000):
    if not os.path.exists("models/"):
        os.mkdir("models/")
    save_prefix = experiment_name.split('.')[0]
    datasets_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'valid']}
    losses_save_name = save_prefix + "_losses.txt"
    metrics = dict()
    with open(os.path.join("models", losses_save_name), "w") as f:
        f.write("train_loss;valid_loss\n")
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    epochs_no_improvement = 0
    isbroken = False
    last_epoch = 0
    for epoch in range(num_epochs_no_stepLR):
        model, optimizer, scheduler, metrics, best_model_wts, epochs_no_improvement, isbroken, best_loss = single_epoch(
            epoch, num_epochs, early_stopping_epochs, model, optimizer, criterion, scheduler, dataloaders, datasets_sizes, losses_save_name, save_prefix, device, iswandb, metrics, best_model_wts, epochs_no_improvement, isbroken, best_loss,
            perform_scheduler_step=False
        )
        last_epoch = epoch
        if isbroken:  #early stopping
            break
    logging.info(f"EPOCH_STARTED_STEP_LR: {last_epoch + 1}")
    metrics['epoch_started_step_lr'] = last_epoch + 1
    isbroken = False
    for epoch in range(last_epoch + 1, last_epoch + 1 + num_epochs - num_epochs_no_stepLR):
        model, optimizer, scheduler, metrics, best_model_wts, epochs_no_improvement, isbroken, best_loss = single_epoch(
            epoch, num_epochs, early_stopping_epochs, model, optimizer, criterion, scheduler, dataloaders, datasets_sizes, losses_save_name, save_prefix, device, iswandb, metrics, best_model_wts, epochs_no_improvement, isbroken, best_loss,
            perform_scheduler_step=True
        )
        if isbroken:  # early stopping
            break
    time_elapsed = time.time() - since
    logging.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    logging.info(f'Best val Loss: {best_loss:4f}')
    metrics['valid_loss_min'] = best_loss
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, metrics


def run_train(valid_patient_id, experiment_id, cuda_id=7, iswandb=False, wandb_project_name_appendix="", wandb_group_name="leave-one-out-00", wandb_reinit=True, wandb_settings=wandb.Settings(start_method="thread"), set_valid_to_train=False):
    if not os.path.exists("./models/"):
        os.mkdir("models")
    # cuda_id = 7
    run_name = "Resi2_6points"
    architecture_name = "Resi2"
    load_encoder_default_weights = True
    load_path = None
    # experiment_id = 21
    experiment_id = str(experiment_id).rjust(3, "0")
    save_name = f"experiment{experiment_id}_{run_name}" + ".pt"
    NUM_EPOCHS = 2000
    NUM_EPOCHS_NO_STEP_LR = 1000
    LEARNING_RATE = 1e-4
    STEP_LR_SIZE = 100
    STEP_LR_GAMMA = 0.8
    CRITERION = nn.L1Loss()
    loss_name = "L1"
    best_model_update_in = "valid"
    optimizer_type = "AdamW"
    num_points = 6
    load_path_num_points = None
    early_stopping_epochs = 330
    replace_linear_head = False
    if valid_patient_id is not None:
        valid_patients = [valid_patient_id]
    else:
        valid_patients = []
    config={
            "set_valid_to_train": set_valid_to_train,
            "valid_patient_id": valid_patient_id,
            "learning_rate": LEARNING_RATE,
            "num_points": num_points,
            "architecture": architecture_name,
            "epochs": NUM_EPOCHS,
            "loss": loss_name,
            "step_lr_size": STEP_LR_SIZE,
            "step_lr_gamma": STEP_LR_GAMMA,
            "best_model_update_in": best_model_update_in,
            "optimizer_type": optimizer_type,
            "weights_load_path": load_path,
            "early_stopping_epochs": early_stopping_epochs,
            "gpu_id": cuda_id,
            "replace_linear_head": replace_linear_head,
            "num_epochs_no_stepLR": NUM_EPOCHS_NO_STEP_LR
    }
    if iswandb:
        wandb.init(
            group=wandb_group_name,
            settings=wandb_settings,
            reinit=wandb_reinit,
            # Set the project where this run will be logged
            project=f"tmj-pointmarker{wandb_project_name_appendix}", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"experiment{experiment_id}_{run_name}", 
            # Track hyperparameters and run metadata
            config=config
        )
    with open("models/" + save_name.split(".pt")[0] + "_config.json", "w") as file:
        json.dump(config, file, indent=4)
    train_ts, valid_ts = setup_datasets(num_points, valid_patients=valid_patients)
    if set_valid_to_train:
        valid_ts, _ = setup_datasets(num_points, valid_patients, transform_train=False)
    # Training DataLoader
    PIN_MEMORY = True
    NUM_WORKERS = 4
    BATCH_SIZE = 64
    train_dl = DataLoader(train_ts,
                        batch_size=BATCH_SIZE, 
                        shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

    # Validation DataLoader
    val_dl = DataLoader(valid_ts,
                        batch_size=BATCH_SIZE, 
                        shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    test_dl = DataLoader(train_ts, # TODO on test set 
                        batch_size=BATCH_SIZE, 
                        shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
    dataloaders = {'train': train_dl, 'valid': val_dl, "test": test_dl}
    device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
    if not replace_linear_head:
        params_model = {
            "num_points": num_points
        }
    else:
        params_model = {
            "num_points": load_path_num_points
        }
    params_model['pretrained_encoder'] = load_encoder_default_weights
    if architecture_name == "Resi2":
        pdector = Resi2(params_model)
    elif architecture_name == "CustomNetwork":
        params_model = {
            "num_points": num_points,
            "shape_in": (3,224,224), 
            "initial_filters": 8,    
            "num_fc1": 100,
            "dropout_rate": 0.1,
            "num_points": num_points}
        pdector = Network(params_model)
    elif architecture_name == "Mobi":
        pdector = Mobi(params_model)
    elif architecture_name == "Mobi2":
        pdector = Mobi2(params_model)
    else:
        raise NotImplementedError("Only Resi2, Mobi, Mobi2 and CustomNetwork are implemented")
    _devi = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    summary(pdector.to(_devi), (3, 224, 224))
    if load_path is not None:
        pdector.load_state_dict(torch.load(load_path, map_location=device))
    if replace_linear_head:
        if architecture_name == "Resi2":
            pdector.linear1 = nn.Linear(512, num_points * 2)
            pdector.params['num_points'] = num_points
        else:
            raise ValueError("Other architecture than Resi2 is not supported.")
    model = pdector.to(device)
    if iswandb:
        wandb.watch(model, log_freq=100)
    # optimizer selection
    if optimizer_type == "SGD":
        OPTIMIZER = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    elif optimizer_type == "Adam":
        OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    elif optimizer_type == "AdamW":
        OPTIMIZER = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    else:
        raise NotImplementedError("Only SGD, Adam and AdamW are implemented")
    my_lr_scheduler = lr_scheduler.StepLR(OPTIMIZER, step_size=STEP_LR_SIZE, gamma=STEP_LR_GAMMA)
    if iswandb:
        wandb.watch(model, log="all", log_freq=100)
    model, metrics = train_model(model, save_name, dataloaders, CRITERION, OPTIMIZER, my_lr_scheduler, iswandb=iswandb, num_epochs=NUM_EPOCHS, device=device, early_stopping_epochs=early_stopping_epochs, num_epochs_no_stepLR=NUM_EPOCHS_NO_STEP_LR)
    logging.info("MSE LOSS evalontest TRAIN:")
    evalontest(model, nn.MSELoss(), dataloaders["test"], device=device)
    logging.info("L1 LOSS evalontest TRAIN:")
    evalontest(model, nn.L1Loss(), dataloaders["test"], device=device)
    if iswandb:
        wandb.run.summary["valid_loss_min"] = metrics['valid_loss_min']
        wandb.run.summary['train_loss_at_min_valid'] = metrics['train_loss_at_min_valid']
        wandb.run.summary['epoch_started_step_lr'] = metrics['epoch_started_step_lr']
    with open("models/" + save_name.split(".pt")[0]+"_metrics_summary.json", "w") as file:
        json.dump(metrics, file, indent=4)


def start_thread_logging():
    """
    Add a log handler to separate file for current thread
    """
    thread_name = threading.Thread.getName(threading.current_thread())
    log_file = './logs/{}-val-patient.log'.format(thread_name)
    log_handler = logging.FileHandler(log_file)

    log_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)-15s"
        "| %(threadName)-11s"
        "| %(levelname)-5s"
        "| %(message)s")
    log_handler.setFormatter(formatter)

    log_filter = ThreadLogFilter(thread_name)
    log_handler.addFilter(log_filter)

    logger = logging.getLogger()
    logger.addHandler(log_handler)

    return log_handler


def stop_thread_logging(log_handler):
    # Remove thread log handler from root logger
    logging.getLogger().removeHandler(log_handler)
    # Close the thread log handler so that the lock on log file can be released
    log_handler.close()


def worker(patient_id, iswandb, cuda_id, set_valid_to_train):
    thread_log_handler = start_thread_logging()
    logging.info('THREAD START.')
    run_train(patient_id, patient_id, cuda_id=cuda_id, iswandb=iswandb, wandb_project_name_appendix="-final", wandb_group_name=None, wandb_reinit=None, set_valid_to_train=set_valid_to_train)
    logging.debug('THREAD ENDING.')
    stop_thread_logging(thread_log_handler)


def config_root_logger():
    if not os.path.exists("./logs/"):
        os.mkdir("./logs/")
    log_file = './logs/perThreadLogging.log'

    formatter = "%(asctime)-15s" \
                "| %(threadName)-11s" \
                "| %(levelname)-5s" \
                "| %(message)s"

    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'root_formatter': {
                'format': formatter
            }
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'root_formatter'
            },
            'log_file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'filename': log_file,
                'formatter': 'root_formatter',
            }
        },
        'loggers': {
            '': {
                'handlers': [
                    'console',
                    'log_file',
                ],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    })


def run_training(patient_id, iswandb, cuda_id, set_valid_to_train):
    config_root_logger()
    if not os.path.exists("logs/"):
        os.mkdir("logs/")
    logging.info("started")
    stime = ctime()
    threads = []
    for patient_id in range(1, 36):
        # start thread with
        t = threading.Thread(target=worker,
                                name='Thread-{}'.format(str(patient_id).zfill(3)),
                                kwargs={
                                "patient_id": patient_id,
                                "iswandb": iswandb,
                                "cuda_id": cuda_id,
                                "set_valid_to_train": set_valid_to_train
                            })
        t.start()
        threads.append(t)
        for x in threads:
            x.join()
        etime = ctime()
        logging.info("finished")
        logging.info(f"time taken: {etime - stime}")
        logging.info(f"in minutes: {(etime - stime)/60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains model given val patient id.')
    parser.add_argument("patient_id", type=int)
    parser.add_argument("cuda_id", type=int)
    parser.add_argument("--iswandb", type=bool, default=True)
    parser.add_argument("--set_valid_to_train", type=bool, default=True)
    args = parser.parse_args()
    patient_id = args.patient_id
    cuda_id = args.cuda_id
    iswandb = args.iswandb
    run_training(patient_id, iswandb, cuda_id, args.set_valid_to_train)
