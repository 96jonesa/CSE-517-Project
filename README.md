# CSE-517-Project

### What is this?



### Contents

'model_python' contains the .py code used to run scripts:

- 'layers.py' contains the implementation of the PyTorch modules used in the model
- 'model.py' contains our MAN-SF model implementation
- 'data_processing.py' contains functions and code for pre-processing data
- 'stockdataset.py' contains the implementation of the StockDataset class
- 'train.py' contains functions and code for training a model

### Dependencies



### Running the code

1. Navigate to the python directory by running `cd model_python`

2. Install all dependencies by running `pip install -r requirements.txt`

3. Download and preprocess the data by running `python main.py --preprocess_in ../wikidata_raw --preprocess_out <your processed data folder>`

4. Train and save an instance of MAN-SF by running `python main.py --train <your processed data folder> --model <folder to save model in >`

5. Evaluate the saved MAN-SF model by running `python main.py --train <your processed data folder> --model <path to the saved model>`

### TODO

- Dependencies (Done) (added a requirements.txt in model_python)
- Data download instruction (Done) (StockNet is downloaded through CLI, Wikidata is included in our repo)
- Preprocessing code + command (Done)
- Training code + command (Need to finalize training method)
- Evaluation code + command
- Pretrained model (if applicable)
- Table of results
