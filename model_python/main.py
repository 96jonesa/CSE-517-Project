import os
import pickle
import argparse
from data_processing import prep_dataset, build_second_order_wikidata_graphs
from stockdataset import StockDataset

STOCKNET_REPO_NAME = 'stocknet-dataset-master'
PROCESSED_STOCKNET_DATA_FOLDER = 'processed_stocknet_data'

"""
Valid executions:
main.py --preprocess_in     <filepath>  --preprocess_out    <filepath>
main.py --train             <filepath>  --model             <filepath>
main.py --evaluate          <filepath>  --model             <filepath>

--preprocess_in should contain a filepath to a directory with the following structure:

root/
- links.csv
- wikidata_entries/
-- individual .txt files of wikidata entries for each stock
"""
def main():
    parser = argparse.ArgumentParser(description='Preprocess data, train or evaluate the MAN-SF model. If executed \
        with --train or --evaluate, --model is required. If executed with --preprocess_in, \
        --preprocess_out is required.')

    # Preprocessing input / Training / Evaluation arguments
    group1 = parser.add_mutually_exclusive_group(required=True)
    group1.add_argument('--preprocess_in', '-pi', metavar='FILEPATH', type=str,
                        help='Preprocesses the data. FILEPATH is filepath to load raw training data')
    group1.add_argument('--train', '-t', metavar='FILEPATH', type=str,
                        help='Trains a MAN-SF model. FILEPATH is filepath for training data')
    group1.add_argument('--evaluate', '-e', metavar='FILEPATH', type=str,
                        help='Evaluates a trained MAN-SF model. FILEPATH is filepath for evaluation data')

    # Preprocessing output / Model
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--preprocess_out', '-po',metavar='FILEPATH', type=str,
                    help='FILEPATH is filepath for exporting processed data as well as downloading the StockNet data')
    group2.add_argument('--model', '-m', metavar='FILEPATH', type=str,
                        help='FILEPATH is filepath to export model (for training) or filepath to load model (for evaluation)')
    
    args = parser.parse_args()

    # Validate arguments
    if args.preprocess_in:
        if not args.preprocess_out:
            parser.error('--preprocess_in requires an additional argument --preprocess_out')
    if args.train or args.evaluate:
        if not args.model:
            parser.error('--train or --evaluation requires an additional argument --model')

    if args.preprocess_in:
        preprocess(args.preprocess_in, args.preprocess_out)
    elif args.train:
        train(args.model, args.train)
    else:
        evaluate(args.model, args.evaluate)

def preprocess(in_filepath, out_filepath):
    # StockNet data setup
    # Download data (if not already available)
    master_zip_filepath = os.path.join(out_filepath, "master.zip")
    stocknet_dataset_filepath = os.path.join(out_filepath, STOCKNET_REPO_NAME)
    data_output_filepath = os.path.join(out_filepath, PROCESSED_STOCKNET_DATA_FOLDER)
    if not os.path.exists(master_zip_filepath):
        os.system(f'wget https://github.com/yumoxu/stocknet-dataset/archive/master.zip\
                    -P {out_filepath}')
    if not os.path.isdir(stocknet_dataset_filepath):
        os.system(f'unzip {master_zip_filepath} -d {out_filepath}')
    if not os.path.isdir(data_output_filepath):
        os.mkdir(data_output_filepath)

    # Train / Val / Test split
    train_start_date = '2014-01-01'
    train_end_date = '2015-07-31'
    val_start_date = '2015-08-01'
    val_end_date = '2015-09-30'
    test_start_date = '2015-10-01'
    test_end_date = '2016-01-01'

    train = prep_dataset(stocknet_dataset_filepath, train_start_date, train_end_date)
    val = prep_dataset(stocknet_dataset_filepath, val_start_date, val_end_date)
    test = prep_dataset(stocknet_dataset_filepath, test_start_date, test_end_date)

    def save_object(obj, filename):
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    # Output StockNet to file
    save_object(train, os.path.join(data_output_filepath, 'train.pkl'))
    save_object(val, os.path.join(data_output_filepath, 'val.pkl'))
    save_object(test, os.path.join(data_output_filepath, 'test.pkl'))

    # Wikidata graph setup
    wikidata_graph = build_second_order_wikidata_graphs(in_filepath)

    # Output graph to file
    with open(os.path.join(out_filepath, 'wikidata_adjacency_list_first_and_second_order.txt'), 'w') as file:
        for key in wikidata_graph.keys():
            file.write(key + ':' + str(wikidata_graph[key]) + '\n')

def train(model_filepath, data_filepath):
    print(f'Running training on {data_filepath}, saving model to {model_filepath}')
    data_output_filepath = os.path.join(data_filepath, PROCESSED_STOCKNET_DATA_FOLDER)

    # Import data from .pkl files
    with open(os.path.join(data_output_filepath, 'train.pkl'), 'rb') as obj:
        train = pickle.load(obj)
        train_company_to_price_df, train_company_to_tweets, train_date_universe, train_n_days, train_n_stocks, train_max_tweets = train

    with open(os.path.join(data_output_filepath, 'val.pkl'), 'rb') as obj:
        val = pickle.load(obj)
        val_company_to_price_df, val_company_to_tweets, val_date_universe, val_n_days, val_n_stocks, val_max_tweets = val

    # Create StockDataset instances
    train_dataset = StockDataset(train_company_to_price_df, train_company_to_tweets, train_date_universe, train_n_days, train_n_stocks, train_max_tweets)
    val_dataset = StockDataset(val_company_to_price_df, val_company_to_tweets, val_date_universe, val_n_days, val_n_stocks, val_max_tweets)

    # TODO: need to create training method
    # man_sf_model = train_model(train_dataset, val_dataset)

def evaluate(model_filepath, data_filepath):
    print(f'Running evaluation on {data_filepath} with model {model_filepath}')
    data_output_filepath = os.path.join(data_filepath, PROCESSED_STOCKNET_DATA_FOLDER)

    with open(os.path.join(data_output_filepath, 'test.pkl'), 'rb') as obj:
        test = pickle.load(obj)
        test_company_to_price_df, test_company_to_tweets, test_date_universe, test_n_days, test_n_stocks, test_max_tweets = test
    test_dataset = StockDataset(test_company_to_price_df, test_company_to_tweets, test_date_universe, test_n_days, test_n_stocks, test_max_tweets)

    with open(model_filepath, 'rb') as model:
        man_sf_model = pickle.load(model)

    # TODO: need to create eval method
    # evaluate(model, test_dataset)

if __name__ == '__main__':
    main()