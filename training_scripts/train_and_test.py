import numpy as np
import torch
import time
import os
from torch.nn import Upsample


def export_weights(model, weights_file):
    """
    Save only the model architecture parameters and state_dict to a file.
    """
    channel_list = [1]
    kernel_list = []
    stride_list = []

    for j in range(len(model.encoder.f_conv)):
        ch = model.encoder.f_conv[j].out_channels
        channel_list.append(ch)

        k = model.encoder.f_conv[j].kernel_size
        kernel_list.append(k)

        st = model.encoder.f_conv[j].stride
        stride_list.append(st)

    torch.save(
        {
            "model_init": [channel_list, kernel_list, stride_list],
            "model_state": model.state_dict(),
        },
        weights_file,
    )


def validate(validation_dataloader, net, loss_function, device):
    """
    Validation step that is done at every epoch.
    """
    val_losses = []

    for sample_batched in validation_dataloader:
        X_test = sample_batched["data"].float().to(device)
        Y_test = sample_batched["target"].float().to(device)

        resize = Upsample(size=(X_test.shape[-1], X_test.shape[-1]), mode="bilinear")
        Y_pred = net(X_test)
        Y_pred = resize(Y_pred)

        val_loss = loss_function(X_test, Y_pred, Y_test)
        val_losses.append(val_loss.detach().cpu().numpy())

        del X_test, Y_test, Y_pred, val_loss

    return np.mean(val_losses)


def train_autoencoder(
    train_dataloader,
    validation_dataloader,
    net,
    optimizer,
    scheduler,
    loss_function,
    epochs,
    checkpoint_period,
    device,
    indicator="",
    batchsize=2,
    save_folder="output/",
    start_epoch=0,
):
    train_loss_history = []  # mean training loss per epoch
    val_loss_history = []  # mean validation loss per epoch
    best_score = 1e9  # keep track of the best score to save
    times = []  # time in seconds it takes to train the model per epoch

    # Initialize paths to None so we can print them after training ends
    best_model_path = None
    best_weights_path = None

    for i in range(start_epoch, start_epoch + epochs):
        start = time.time()

        train_losses = []  # training loss per batch

        for sample_batched in train_dataloader:
            optimizer.zero_grad()

            X = sample_batched["data"].float().to(device)
            Y = sample_batched["target"].float().to(device)

            resize = Upsample(size=(X.shape[-1], X.shape[-1]), mode="bilinear")
            Y_hat = net(X)
            Y_hat = resize(Y_hat)

            train_loss = loss_function(X, Y_hat, Y)
            train_loss.backward()
            optimizer.step()

            train_losses.append(train_loss.detach().cpu().numpy())

            del X, Y, train_loss

        mean_train_loss = np.mean(train_losses)
        mean_val_loss = validate(validation_dataloader, net, loss_function, device)

        scheduler.step()
        train_loss_history.append(mean_train_loss)
        val_loss_history.append(mean_val_loss)

        # Save training history after each epoch to allow resumption
        torch.save(
            train_loss_history, os.path.join(save_folder, f"train_error{indicator}")
        )
        torch.save(
            val_loss_history, os.path.join(save_folder, f"validation_error{indicator}")
        )

        if mean_train_loss < 0.00002:
            print("Saving current autoencoder model to disk (early stopping)")
            torch.save(
                net,
                os.path.join(save_folder, f"autoencoder2d_stopped_{i + 1}{indicator}"),
            )
            break

        if mean_val_loss < best_score:
            best_score = mean_val_loss
            best_model_path = os.path.join(
                save_folder, f"autoencoder2d_best{indicator}"
            )
            best_weights_path = os.path.join(
                save_folder, f"autoencoder2d_best_weights{indicator}.pt"
            )

            torch.save(net, best_model_path)
            export_weights(net, best_weights_path)

        print(
            f"EPOCH = {i + 1}, TRAIN LOSS = {mean_train_loss:.6f}, VAL LOSS = {mean_val_loss:.6f}"
        )

        if (i + 1) % checkpoint_period == 0:
            print("Saving current autoencoder model to disk (checkpoint)")
            torch.save(
                net, os.path.join(save_folder, f"autoencoder2d_{i + 1}{indicator}")
            )

        end = time.time()
        times.append(end - start)
        print(f"Epoch time: {times[-1]:.2f} seconds")

    if best_model_path and best_weights_path:
        print(
            f"Saved best model to {best_model_path} and weights to {best_weights_path}"
        )

    return train_loss_history, val_loss_history, times


def test_autoencoder(
    test_dataloader, net, loss_function, device, savefile=None, indicator=""
):
    """
    Evaluate the model on the test dataset after training.
    """

    if savefile is not None:
        net = torch.load(savefile)
        net.eval()

    test_losses = []

    for sample_batched in test_dataloader:
        X_test = sample_batched["data"].float().to(device)
        Y_test = sample_batched["target"].float().to(device)

        resize = Upsample(size=(X_test.shape[-1], X_test.shape[-1]), mode="bilinear")
        Y_pred = net(X_test)
        Y_pred = resize(Y_pred)

        test_loss = loss_function(X_test, Y_pred, Y_test)
        test_losses.append(test_loss.detach().cpu().numpy())

        del X_test, Y_test, Y_pred, test_loss

    mean_test_loss = np.mean(test_losses)

    return mean_test_loss
