# Project

### Description

In this project, we try to reproduce the result in "An attempt at beating the 3D U-Net". And we make some new innovations on introducing pretrain methods and discussing ensemble methods.

### Installation

First, we download the KiTS19 dataset from the official repository.

```
git clone https://github.com/neheller/kits19
cd kits19
pip3 install -r requirements.txt
python3 -m starter_code.get_imaging
```

Rename the dataset folder

```bash
mv kits19 kits19_data
```

It is recommended to create a Conda environment in the following way:


```bash
conda env create -f environment.yml  
conda activate project_u_net
```

The strictly necessary Python dependencies are automatically installed after creating the Conda environment. Some important packages are

- **for image**
  - SimpleITK
- **for model and computation**
  - batchgenerators
  - pytorch


### How to use

First, resample the data and make cross-validation fold

```bash
python resample_kits19.py
python validation_fold.py
```

And then, modify the hyperparameter in the file `config/defaults.py` to set your model, learning rate, epochs and soon. Run the code

```bash
python train.py
```



### Framework

- Languages: Python
- Environment: Conda

### Inference

Download the best model form the link https://drive.google.com/drive/folders/1aXk7szjt5ZU44zF8Q-rnx-SygE2Q2YK4?usp=drive_link or use your trained best model. The best model of each fold should be placed in the folder `fold_i`.

And then run the script

```
python inference.py
```

to get the segmentation.

### Appendix

There is no use of auto-generation tools.
