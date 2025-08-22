from configs import parse_args
from data import load_datasets
from model import create_model
from train import train

if __name__ == "__main__":
    args = parse_args()
    print(args)

    train_data, test_data, train_indices, test_indices = load_datasets(
        args.dataset_name,
        args.dataset_path,
        args.split_train_name,
        args.split_test_name,
        args.target_field,
    )

    model, optimizer, loss_fn = create_model(args)
    train(args, model, optimizer, loss_fn, train_data, test_data)
