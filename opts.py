import argparse


def parse_opts_offline(args=None):
    # Offline means not real time 
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cmd',
        default='train',
        type=str,
        help='command: train, test or transfer')
    parser.add_argument(
        '--training_path',
        type=str,
        help='path of the training/retraining data')
    parser.add_argument(
        '--validation_path',
        type=str,
        help='path of the validation data')
    parser.add_argument(
        '--testing_path',
        type=str,
        help='path of the validation data')
    parser.add_argument(
        '--oldmodel_path',
        type=str,
        help='path of the old/testing model, used for inference and transfer learning, model loading')
    parser.add_argument(
        '--oldmodel_weight_path',
        type=str,
        help='path of the old/testing model weights, used for inference and transfer learning, model loading')
    parser.add_argument(
        '--model_path',
        type=str,
        help='path of the training/retraining result model, used as the result of training and transfer learning')
    parser.add_argument(
        '--weight_path',
        type=str,
        help='path of the training/retraining result model weights, used as the result of training and transfer learning')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='batch size')
    parser.add_argument(
        '--timesteps',
        type=int,
        help='timesteps, observation')
    parser.add_argument(
        '--predict_steps',
        type=int,
        help='prediction steps, prediction')
    parser.add_argument(
        '--data_dim',
        type=int,
        help='dimension of the data')
    parser.add_argument(
        '--epochs',
        type=int,
        help='epochs')
    parser.add_argument(
        '--hidden_size',
        type=int,
        help='hidden size')
    parser.add_argument(
        '--train_num',
        type=int,
        help='number of training samples')
    parser.add_argument(
        '--valid_num',
        type=int,
        help='number of validation samples')
    parser.add_argument(
        '--test_num',
        type=int,
        help='number of testing samples')
    parser.add_argument(
        '--testing_res',
        type=str,
        help='path of testing result')
    parser.add_argument(
        '--testing_res_CDF',
        type=str,
        help='path of testing CDF result')

    args = parser.parse_args(args)
    return args

