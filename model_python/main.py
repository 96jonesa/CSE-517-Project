import argparse

"""
Valid executions:
main.py --preprocess_in     <filepath>  --preprocess_out    <filepath>
main.py --train             <filepath>  --model             <filepath>
main.py --evaluate          <filepath>  --model             <filepath>

--preprocess_in should contain a filepath to a directory with the following structure:

root/
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
                    help='FILEPATH is filepath for exporting processed data')
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
    print(f'unimplemented preprocess stub with input filepath {in_filepath} and output filepath {out_filepath}')

def train(model_filepath, data_filepath):
    print(f'unimplemented training stub with filepath {data_filepath} and model path {model_filepath}')

def evaluate(model_filepath, data_filepath):
    print(f'unimplemented evaluation stub with filepath {data_filepath} and model path {model_filepath}')

if __name__ == '__main__':
    main()