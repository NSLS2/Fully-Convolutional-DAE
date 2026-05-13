import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import os
from utils import setup_nn, set_seed
from train_and_test import train_autoencoder, test_autoencoder
from torch.utils.data import DataLoader
from corr_dataset import CorrDataSet
import re

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--train", required=True, help="log file corresponding to training data"
    )
    arg_parser.add_argument(
        "--test", required=True, help="log file corresponding to testing data"
    )
    arg_parser.add_argument(
        "--validate", required=True, help="log file corresponding to validation data"
    )
    arg_parser.add_argument(
        "--output-dir",
        required=True,
        help="output directory where the training results should be saved",
    )
    arg_parser.add_argument(
        "--epochs", nargs="?", const=100, type=int, default=100, help="epochs"
    )
    arg_parser.add_argument(
        "--checkpoint_period",
        nargs="?",
        const=1,
        type=int,
        default=1,
        help="checkpoint period",
    )
    arg_parser.add_argument(
        "--lr", nargs="?", const=0.005, type=float, default=0.005, help="learning rate"
    )
    arg_parser.add_argument(
        "--batchsize", nargs="?", const=64, type=int, default=64, help="batchsize"
    )
    arg_parser.add_argument(
        "--seed", nargs="?", const=0, type=int, default=0, help="random seed"
    )
    arg_parser.add_argument(
        "--ksize", nargs="?", const=3, type=int, default=3, help="kernal size"
    )
    arg_parser.add_argument(
        "--loss",
        choices=["mse", "mae", "smoothl1", "elastic", "symmetry"],
        type=str,
        default="mse",
        help='Loss function to use: "mse", "mae", "smoothl1", "elastic", or "symmetry"',
    )
    arg_parser.add_argument(
        "--resume-epoch", type=int, default=0, help="Epoch number to resume from"
    )
    arg_parser.add_argument(
        "--savefile",
        type=str,
        default=None,
        help="Path to a saved model to resume training or run inference",
    )
    args = arg_parser.parse_args()
    return args


if __name__ == "__main__":
    metadata = {}

    args = get_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
        metadata[arg] = getattr(args, arg)

    ####################################### define parameters ##############################################################

    # indicate which data to use, based on the log files
    train_data_file = args.train
    validation_data_file = args.validate
    test_data_file = args.test

    #data_seed = get_seed_from_train_arg(train_data_file) 

    # parameters of the model training
    epochs = args.epochs  # epochs    =  100
    checkpoint_period = (
        args.checkpoint_period
    )  # checkpoint_period = 1 # steps (in epoch) for saving the intermediate results
    lr = args.lr  # lr = 0.005 # learning rate
    batchsize = args.batchsize  # batchsize = 64
    ksize = args.ksize  # ksize = 3 # kernel size
    seed = args.seed  # seed = 0 #random seed
    save_folder = args.output_dir  # 'output/'
    os.makedirs(save_folder, exist_ok=True)
    loss = args.loss
    start_epoch = args.resume_epoch  # Epoch number to resume from
    remaining_epochs = epochs - start_epoch
    indicator = (f"_lr_{lr}_batchsize_{batchsize}_cv_4_8_16_32_k_{ksize}_trainseed_{seed}_loss_{loss}")

    if args.savefile is not None:
        savefile = args.savefile
    elif args.resume_epoch > 0:
        savefile = os.path.join(
            args.output_dir, f"autoencoder2d_{args.resume_epoch}{indicator}"
        )
    else:
        savefile = None

    assert 0 <= start_epoch <= epochs, "resume-epoch must be between 0 and total epochs"

    if start_epoch > 0:
        print(
            f"Resuming training from epoch {start_epoch} to {start_epoch + remaining_epochs}"
        )

    os.makedirs(save_folder, exist_ok=True)
    with open(os.path.join(save_folder, "metadata.json"), "w") as fp:
        json.dump(metadata, fp)

    set_seed(seed)

    ######################################## set up the data handling ##################################################

    # load the dataset
    train_ds = CorrDataSet(train_data_file)
    test_ds = CorrDataSet(test_data_file)
    val_ds = CorrDataSet(validation_data_file)

    # initialize the DataLoaders
    train_dataloader = DataLoader(
        train_ds, batch_size=batchsize, shuffle=True, num_workers=1
    )
    validation_dataloader = DataLoader(
        val_ds, batch_size=batchsize, shuffle=False, num_workers=1
    )
    test_dataloader = DataLoader(
        test_ds, batch_size=batchsize, shuffle=False, num_workers=1
    )

    ######################################## initialize the model ######################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net, optimizer, scheduler, loss_function = setup_nn(
        ksize=ksize, lr=lr, savefile=savefile, device=device, loss_name=loss
    )

    ######################################## train the model ###########################################################

    # Load previous training history if resuming
    train_error_path = os.path.join(save_folder, f"train_error{indicator}")
    val_error_path = os.path.join(save_folder, f"validation_error{indicator}")

    if start_epoch > 0:
        if os.path.exists(train_error_path):
            J_prev = torch.load(train_error_path)
        else:
            print("Warning: Previous training loss file not found.")
            J_prev = []

        if os.path.exists(val_error_path):
            V_prev = torch.load(val_error_path)
        else:
            print("Warning: Previous validation loss file not found.")
            V_prev = []
    else:
        J_prev = []
        V_prev = []

    # Train the model
    J_new, V_new, times = train_autoencoder(
        train_dataloader=train_dataloader,
        validation_dataloader=validation_dataloader,
        net=net,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        epochs=remaining_epochs,
        checkpoint_period=checkpoint_period,
        device=device,
        indicator=indicator,
        batchsize=batchsize,
        save_folder=save_folder,
        start_epoch=start_epoch,
    )

    # Combine previous + new losses
    J = list(J_prev) + list(J_new)
    V = list(V_prev) + list(V_new)

    # Save combined timing info
    print(np.mean(times), np.std(times))
    with open(os.path.join(save_folder, "timing.json"), "w") as f:
        json.dump({"mean": float(np.mean(times)), "std": float(np.std(times))}, f)

    # Plot and save full training loss
    plt.figure()
    plt.semilogy(J)
    plt.xlabel("Epoch", labelpad=10)
    plt.ylabel("Training Loss", labelpad=10)
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"train_loss_curve{indicator}.png"))
    plt.close()

    # Plot and save full validation loss
    plt.figure()
    plt.semilogy(V)
    plt.xlabel("Epoch", labelpad=10)
    plt.ylabel("Validation Loss", labelpad=10)
    plt.title("Validation Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"val_loss_curve{indicator}.png"))
    plt.close()

    # Save full loss history and final test error
    torch.save(J, train_error_path)
    torch.save(V, val_error_path)

    test_error = test_autoencoder(
        test_dataloader, net, loss_function, device, savefile, indicator
    )
    torch.save(test_error, os.path.join(save_folder, f"test_error{indicator}"))

    print(f"Test error: {test_error:.6f}")
