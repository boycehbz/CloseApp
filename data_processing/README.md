# Data Processing

This folder contains the preprocessing pipeline used to convert wild image
sequences into the training format expected by CloseApp.

The main entry point is:

```bash
cd ./data_processing
python generate_wild_data.py
```

## Directory Layout

The preprocessing code expects this structure:

```text
data_processing/
  generate_wild_data.py
  data/
    images/
      <sequence_name>/
        000000.jpg
        000001.jpg
        ...
    uv/
      query_posemap_128_cano_smpl.npz
      ...
  smpl/
    smpl/
      SMPL_NEUTRAL.pkl
      SMPL_MALE.pkl
      SMPL_FEMALE.pkl
    J_regressor_halpe.npy
    ...
  pretrained/
    yolox_data/
    ...
```
Input and output can be modified by changing the code in `generate_wild_processing.py`:

```python
root = 'data/images'
out = 'output/preprocess_data'
```

## Download

Download the pretrained models from [Baidu Netdisk](https://pan.baidu.com/s/1VutLOlsraOQMaYijZuQ88A?pwd=dyxe).

Download the smpl models from [Baidu Netdisk](https://pan.baidu.com/s/1ICN1pMd05BYJdsjzvNcrWw?pwd=ehbk).

Download the data from [Baidu Netdisk](https://pan.baidu.com/s/1FVIgruvvvMxEMfgDHEahMg?pwd=92vg).

## How to use the output
The default output is:

data_processing/output/preprocess_data/<sequence_name>/train/

To train CloseApp, move the sequence folder to:`CloseApp/data/preprocess_data/<sequence_name>/train/`

Then run training with:
```python
cd ..
python train.py -s data/preprocess_data/<sequence_name> -m output/<sequence_name> --train_stage=1 --save_render --use_appearance --save_params
```