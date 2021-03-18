# CSE-517-Project

### What is this?

This is the code repository for our (Andy Jones, Brandon Ko, Kyle Xiao) CSE 517 reproducibility project. This repo contains our code used in attempting to reproduce the model and findings of Sawhney et al.'s "Deep Attentive Learning for Stock Movement Prediction From Social Media Text and Company Correlations" (https://www.aclweb.org/anthology/2020.emnlp-main.676.pdf). Contains model implementation as well as scripts for training and evaluating models, and for pre-processing the StockNet dataset (https://github.com/yumoxu/stocknet-dataset) and WikiData relations data.

### Contents

'model_python' contains the .py code used to run scripts:

- 'layers.py' contains the implementation of the PyTorch modules used in the model
- 'model.py' contains our MAN-SF model implementation
- 'data_processing.py' contains functions and code for pre-processing data
- 'stockdataset.py' contains the implementation of the StockDataset class
- 'train.py' contains functions and code for training a model
- 'evaluate.py' contains functions and code for evaluating a model
- 'main.py' is the primary interface from which various scripts can be run
- 'requirements.txt' lists the dependencies need to run the scripts in this repo

'model_notebooks' contains various .ipynb notebooks, primarily used for scratch work and sandbox testing of the code present in 'model_python'

'wikidata_raw' contains a folder named 'wikidata_entries' with Wikidata entries for each stock entity along with a file named 'links.csv' mapping stock ticker symbols to their Wikidata URL (which contains their Wikidata entity number).

### Dependencies

Requires installation of the following:

- torch
- tensorflow
- tensorflow_hub
- matplotlib
- numpy
- pandas
- seaborn
- pandas_market_calendars
- tqdm



### Obtaining the data

The StockNet dataset is automatically downloaded when the preprocessing script is run. The WikiData relations data is available in this repo.

### Running the code

1. Navigate to the python directory
```
cd model_python
```

2. Install all dependencies
```
pip install -r requirements.txt
```

3. Download and preprocess the data
```
python main.py --preprocess_in ../wikidata_raw --preprocess_out <your processed data folder (i.e. ../tmp)>
```

4. Train and save an instance of MAN-SF
```
python main.py --train <your processed data folder (i.e. ../tmp)> --model <folder to save model in (i.e. ../tmp/model.pkl)>
```

5. Evaluate the saved MAN-SF model
```
python main.py --evaluate <your processed data folder (i.e. ../tmp)> --model <path to the saved model (i.e. ../tmp/model.pkl)>
```
