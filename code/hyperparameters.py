"""
@author: Anton Steenvoorden
"""
import argparse


class HyperParameters:
    parser = argparse.ArgumentParser()

    # train
    ## files
    parser.add_argument('--train_path', default='../data/movielens/data/ml1m_train_10.csv', help="train data location")
    parser.add_argument('--valid_path', default='../data/movielens/data/ml1m_valid_10.csv', help="valid data location")
    parser.add_argument('--test_path', default='../data/movielens/data/ml1m_test_10.csv', help="test data location")
    parser.add_argument('--properties_path', default='../data/movielens/data/ml1m_properties.csv', help="attributes per item location")

    # Misc
    parser.add_argument("--evaluate", type=str, default=False, help="provide timestamp of model to evaluate")
    parser.add_argument("--continue_from", type=str, default=False, help="provide timestamp of model to evaluate")
    parser.add_argument('--log_dir', default="logs", help="logging directory")
    parser.add_argument('--metrics_log_dir', default="logs_metrics", help="metrics logging directory")
    parser.add_argument('--tensorboard_log_dir', default="tensorboard_logs", help="tensorboard log directory")
    parser.add_argument('--model_dir', default="trained_models", help="directory to store trained models")
    parser.add_argument('--model_type', default="ADSR", help="model to use from {BSR, BSR_USER, ANAM, ANAM_USER, "
                                                             "MTASR, MTASR_USER, ADSR, ADSR_USER}")
    parser.add_argument('--device', default=None, help="select device type")

    # Model
    parser.add_argument('--num_gru_units', default=128, type=int,
                        help="GRU hidden dimension")
    parser.add_argument('--num_item_units', default=128, type=int,
                        help="hidden dimension of item embedding")
    parser.add_argument('--num_user_units', default=128, type=int,
                        help="hidden dimension of user embedding")
    parser.add_argument('--num_context_units', default=128, type=int,
                        help="hidden dimension of context embedding")

    parser.add_argument('--recommendation_len', default=10, type=int,
                        help="length of recommendation list")
    parser.add_argument('--max_len', default=9, type=int,
                        help="length of sequence")

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--num_epochs', default=60, type=int)
    parser.add_argument('--lr', default=0.001, type=float, help="learning rate. Default is Adam's default")
    parser.add_argument('--dropout_rate', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)

    parser.add_argument('--lambda_score', default=1, type=float)
    parser.add_argument('--lambda_multitask_loss', default=0.9, type=float)
    parser.add_argument('--lambda_diversity_loss', default=0.9, type=float)

    parser.add_argument('--random_seed', default=42, type=int, help="random seed")
    parser.add_argument('--notes', default=None, help="Set this argument to add some notes")
