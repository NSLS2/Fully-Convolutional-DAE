# Training the Fully Convolutional Model

## 1. Setup Environment
Create and activate the required Conda environment.
```bash
conda env create -f environment.yml
conda activate <env_name>
```

## 2. Data Preprocessing & Augmentation
Run the processing script to prepare your dataset for training and evaluation.

```bash
python data_processing.py --input <path_to_input> --output <path_to_processed_output>
```

### Instructions:
1.  **Process both splits:** Repeat this process for the training and test data located at:
    *   `data/train/raw_g2`
    *   `data/test/raw_g2`
2.  **Output:** The script will generate `train_*.pt` and `test_*.pt` files in the `<path_to_processed_output>/dataset_info/` directory.
3.  **Next Step:** Use these generated `.pt` file paths for the `--train` and `--test` arguments in the following step.


## 3. Training the Model
Execute the driver script with your data paths and configuration.
```bash
python driver.py \
  --train `<train_data>` \
  --test `<test_data>` \
  --validate `<val_data>` \
  --output-dir `<output_dir>` \
  --seed `<seed>`
```
