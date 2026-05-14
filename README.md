# Training the Fully Convolutional Model

## 1. Setup Environment
Create and activate the required Conda environment.
```bash
conda env create -f environment.yml
conda activate <env_name>
```

## 2. Data Augmentation
Run the processing script to prepare your dataset.
```bash
python data_processing.py --input `<path_to_input>` --output `<path_to_save_output>`
```

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
