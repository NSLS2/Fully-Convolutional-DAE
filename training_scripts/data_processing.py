#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import json
import re
import numpy as np
import torch
from datetime import datetime
import argparse

def fill_diagonal(x):
    """Replace diagonal values with average of neighbors (excluding out-of-bounds)."""
    N = x.shape[0]
    for j in range(1, N-1):   # avoid j=0 and j=N-1
        x[j, j] = 0.5 * (x[j-1, j] + x[j+1, j])
    if N > 1:
        x[0, 0] = x[1, 1]
        x[-1, -1] = x[-2, -2]

def create_dataset(path_in, path_out, path_target, logs_dirname="logs", 
                   enable_bootstrap=False, bs_data_path=None):
    """
    Creates a dataset by processing raw two-time correlation function files.

    Iterates through raw .npy files from `path_in`, extracts and normalizes
    image patches, generates targets using `model`, and saves them to `path_out`
    and `path_target` respectively. A summary dictionary of generated samples
    and a metadata file are saved to `logs_dirname`.

    Args:
        path_in (str): Path to the input folder with raw .npy files.
        path_out (str): Path to the output folder for processed sample files.
        path_target (str): Path to the output folder for target files.
        logs_dirname (str, optional): Folder for saving dataset dictionary and metadata. Defaults to "logs".
        enable_bootstrap (bool, optional): If True, also process bootstrapped data. Defaults to False.
        bs_data_path (str, optional): Path to the folder containing bootstrapped data files. Required if enable_bootstrap is True.

    Returns:
        None: Saves data and metadata to disk.
    """

    search_pattern = os.path.join(path_in, '*.npy') # look for .npy files directly
    files = glob.glob(search_pattern)
    dataset = {}
    patch_size = 100
    model_file = 'model_for_creating_target/autoencoder2d_best_lr_0.0001_latent_space_8_batchsize_8_cv_10_10_k_1_seed_0_adding_inverted_data'
    model = torch.load(model_file, 'cpu')
    model.eval()

    for file in files:
        file_processing(file, path_out, path_target, dataset, patch_size, model, 
                        enable_bootstrap, bs_data_path)

    today = datetime.now()
    path_in_list = os.path.normpath(path_in).split(os.sep)
    # Ensure filename is robust even if path_in is at root level or similar
    if len(path_in_list) >= 2:
        filename_parts = [path_in_list[-2], path_in_list[-1]] # Changed to take last two parts
    elif len(path_in_list) == 1 and path_in_list[0]:
        filename_parts = [path_in_list[0]]
    else:
        filename_parts = ["unspecified_input"]
    
    filename = "_".join(filename_parts + [today.strftime("%d-%m-%Y_%H-%M")])


    torch.save(dataset, data_path := os.path.join(logs_dirname, f"{filename}.pt"))
    print(f"\nData path dictionary saved to: {data_path}")

    metadata = {}
    metadata['date'] = today.strftime("%d-%m-%Y:%H-%M")
    metadata['path_in'] = path_in
    metadata['path_out'] = path_out
    metadata['patch_size'] = patch_size
    metadata['model_file'] = model_file
    metadata['enable_bootstrap'] = enable_bootstrap
    if enable_bootstrap:
        metadata['bs_data_path'] = bs_data_path

    with open(metadata_path := os.path.join(logs_dirname, f'{filename}_metadata.json'), 'w') as fp:
        json.dump(metadata, fp)
    print(f"Meta data saved to: {metadata_path}\n")

def file_processing(file, path_out, path_target, dataset, patch_size, model, 
                    enable_bootstrap, bs_data_path):
    """
    Processes a single raw two-time correlation function .npy file.

    Extracts, normalizes, and saves smaller patches (samples) from the input file.
    It also generates and saves corresponding target values using the provided model.
    Information about each created sample and its target is added to the dataset dictionary.

    Args:
        file (str): The path to the raw .npy file to be processed.
        path_out (str): The output directory where processed sample files will be saved.
        path_target (str): The output directory where corresponding target files will be saved.
        dataset (dict): The dictionary to which sample and target file paths are added.
        patch_size (int): The size of the square patches to be extracted.
        model (torch.nn.Module): The PyTorch model used to generate target values.
        enable_bootstrap (bool): If True, also process bootstrapped data.
        bs_data_path (str): Path to the folder containing bootstrapped data files.
    
    Returns:
        None: Processes the file and updates the `dataset` dictionary in-place.
    """
    data_file_match = re.search(r"g2_(?P<roi>\d+)_(?P<uid>\w+)\.npy$", os.path.basename(file))

    if data_file_match is None:
        print(f"Not a data file: {file}")
        return

    uid = data_file_match.group("uid")
    roi = data_file_match.group("roi")
    print(f"Processing {os.path.basename(file)}")

    # Centralize data sources
    data_sources = {
        "original": file,
    }
    if enable_bootstrap and bs_data_path:
        # Define all potential bootstrapped data files based on the current original file's UID and ROI
        # Assuming bootstrapped files follow a naming convention like g2_ROI_UID_PERCENTAGE.npy
        bs_percentages = ["05", "10", "25", "50"] # Define your bootstrap percentages
        for bp in bs_percentages:
            bs_filename = f"g2_{roi}_{uid}_{bp}.npy"
            data_sources[f"bs_{bp}"] = os.path.join(bs_data_path, bs_filename)

    loaded_signals = {}
    found_bootstrapped_files = False # Flag to track if any bootstrapped files were loaded

    for source_type, path in data_sources.items():
        if os.path.exists(path):
            loaded_signals[source_type] = np.load(path)
            if source_type == "original":
                print(f"  Loaded original file: {os.path.basename(path)}")
            else: # This is a bootstrapped file
                print(f"  Loaded bootstrapped file: {os.path.basename(path)}")
                found_bootstrapped_files = True
        elif source_type == "original":
            print(f"Error: Original file not found at {path}. Skipping.")
            return # Skip if the original file is missing
        # The 'else' block for bootstrapped files will now be handled outside the loop
        # to provide a single message if none are found.

    # After attempting to load all data sources, check if bootstrapping was enabled
    # and if no bootstrapped files were found.
    if enable_bootstrap and not found_bootstrapped_files and len(data_sources) > 1:
        # len(data_sources) > 1 ensures we actually tried to look for bootstrapped files
        print(f"  No bootstrapped files found for {os.path.basename(file)} in {bs_data_path}")

    # Process each loaded signal
    # --- Step 1: process only original first ---
    original_signal = loaded_signals.get("original", None)
    if original_signal is None:
        print(f"Error: Original file not found. Skipping {file}")
        return

    n_frames = original_signal.shape[0]
    fill_diagonal(original_signal)
    first_diagonal = np.mean(np.diag(original_signal, 1))

    # Store patch index parameters for reuse
    patch_index_list = []

    if first_diagonal < 2:
        for inv, x in enumerate([original_signal, original_signal[::-1, ::-1]]):
            for d in [1,2,3,4,10,20,30]:
                if n_frames // d > patch_size:
                    for m in range(d):
                        y = x[m::d, m::d].copy()
                        fill_diagonal(y)
                        n_max = y.shape[0]
                        all_indexes = list(range(patch_size, n_max+1, 100))

                        if not all_indexes:
                            continue

                        #index_size = np.min((10, len(all_indexes)))
                        index_size = len(all_indexes)
                        if index_size == 0:
                            continue
                        #indexes = np.random.choice(all_indexes, size=index_size, replace=False)
                        indexes = sorted(all_indexes)

                        for j in indexes:
                            # record patch location
                            patch_index_list.append((inv, d, m, j))

                            # --- extract original patch ---
                            sample = y[j-patch_size:j, j-patch_size:j]
                            sample = sample.reshape(1, patch_size, patch_size)
                            sample_mean = np.mean(sample)
                            sample_sd = np.std(sample)

                            if sample_sd != 0:
                                sample = (sample - sample_mean) / sample_sd
                            else:
                                sample = (sample - sample_mean)

                            # unique name
                            sample_name = f"{uid}_{roi}{j}{d}{m}{patch_size}_{inv}"

                            # save sample
                            torch.save(sample, os.path.join(path_out, sample_name))

                            # --- compute target only for original ---
                            target = model(torch.from_numpy(sample).float().reshape(1,1,patch_size,patch_size)).cpu().detach().numpy().reshape(1,patch_size,patch_size)
                            torch.save(target, os.path.join(path_target, sample_name))

                            # add to dataset
                            dataset[sample_name] = {
                                'data': os.path.join(path_out, sample_name),
                                'target': os.path.join(path_target, sample_name)
                            }
    else:
        print(f'Bad scan ROI found for {os.path.basename(file)} (source: original)')
        return

    # --- Step 2: process bootstrapped signals using same patch indexes ---
    for source_type, signal in loaded_signals.items():
        if source_type == "original":
            continue  # already done

        fill_diagonal(signal)

        for (inv, d, m, j) in patch_index_list:
            x = signal[::-1, ::-1] if inv else signal
            y = x[m::d, m::d].copy()
            fill_diagonal(y)

            sample = y[j-patch_size:j, j-patch_size:j]
            sample = sample.reshape(1, patch_size, patch_size)

            sample_mean = np.mean(sample)
            sample_sd = np.std(sample)
            if sample_sd != 0:
                sample = (sample - sample_mean) / sample_sd
            else:
                sample = (sample - sample_mean)

            sample_name = f"{source_type}_{uid}_{roi}{j}{d}{m}{patch_size}_{inv}"
            torch.save(sample, os.path.join(path_out, sample_name))

            # --- reuse target from original ---
            target_name = f"{uid}_{roi}{j}{d}{m}{patch_size}_{inv}"
            dataset[sample_name] = {
                'data': os.path.join(path_out, sample_name),
                'target': dataset[target_name]['target']  # link to original target
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augmentation for both 'train' and 'test' samples.")
    parser.add_argument("--input",            required=True,       help="Base input folder (e.g., 'data/')")
    parser.add_argument("--output",           required=True,       help="Base output folder for processed samples (e.g., 'processed_data/')")
    parser.add_argument("--step",             required=True,       choices=['train', 'test'], help="Choose 'train' or 'test' for the dataset subset.")
    parser.add_argument("--enable_bootstrap", action='store_true', help="Enable processing of bootstrapped data.")
    parser.add_argument("--bs_data_path",     default=None, help="Path to the folder containing bootstrapped data files (e.g., '../../bs_data'). Required if --enable_bootstrap is used.")
    args = parser.parse_args()

    #np.random.seed(args.seed)
    #print(f"\nRandom seed set to: {args.seed}")
    
    if not os.path.isdir(args.input):
        parser.error(f"Input data path not found: {args.input}")
        
    input_path = os.path.join(args.input, args.step, 'raw_g2')
    if not os.path.isdir(input_path):
        parser.error(f"Input raw_g2 path not found: {input_path}")
        
    # Validate bs_data_path if bootstrapping is enabled
    if args.enable_bootstrap:
        if not args.bs_data_path:
            parser.error("--bs_data_path is required when --enable_bootstrap is set.")
        if not os.path.isdir(args.bs_data_path):
            parser.error(f"Bootstrapped data path not found: {args.bs_data_path}")
        
    output_path = os.path.join(args.output, args.step, 'small_frames')
    target_path = os.path.join(args.output, args.step, 'small_targets')
    log_path = os.path.join(args.output, 'dataset_info')

    # Create directories for outputs only
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(target_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    print(f"\nProcessing data from: {input_path}")
    print(f"Input image files will be saved to: {output_path}")
    print(f"Target image files will be saved to: {target_path}")
    if args.enable_bootstrap:
        print(f"Bootstrapped data will be sourced from: {args.bs_data_path}\n")
    else:
        print("Bootstrapped data augmentation is disabled.\n")

    create_dataset(input_path, output_path, target_path, log_path,
                   args.enable_bootstrap, args.bs_data_path)

    print("Dataset creation process completed.")
