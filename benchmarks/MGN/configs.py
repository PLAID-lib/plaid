import json
import argparse

def parse_args():
    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")

    # Add all the existing arguments
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--split_train_name", type=str, default="train")
    parser.add_argument("--split_test_name", type=str, default="test")
    parser.add_argument("--bandwidth", type=float)
    
    parser.add_argument("--activation", type=str, choices=["relu", "elu", "leaky"], default="relu")
    parser.add_argument("--aggregation", type=str, default="sum")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=1111)
    parser.add_argument("--input_dim_nodes", type=int, default=2)
    parser.add_argument("--input_dim_edges", type=int, default=1)
    parser.add_argument("--output_dim", type=int, default=2)
    parser.add_argument("--processor_size", type=int, default=10)
    parser.add_argument("--num_layers_node_processor", type=int, default=2)
    parser.add_argument("--num_layers_edge_processor", type=int, default=2)
    parser.add_argument("--hidden_dim_node_encoder", type=int, default=32)
    parser.add_argument("--num_layers_node_encoder", type=int, default=2)
    parser.add_argument("--hidden_dim_edge_encoder", type=int, default=32)
    parser.add_argument("--num_layers_edge_encoder", type=int, default=2)
    parser.add_argument("--hidden_dim_node_decoder", type=int, default=32)
    parser.add_argument("--num_layers_node_decoder", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    
    parser.add_argument("--error_n_levels", type=int, default=40)
    parser.add_argument("--error_k_hop_levels", type=int, default=3)
    parser.add_argument("--error_min_points", type=int, default=5)
    parser.add_argument("--error_threshold", type=float, default=0.08)
    parser.add_argument("--error_check_interval", type=int, default=10)
    parser.add_argument("--method_error", type=str, default="relative_error")

    parser.add_argument('--mode', type=str, default="classic", choices=["classic"])
    parser.add_argument("--target_field", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)

    # Parse and return the arguments
    args = parser.parse_args()

    # Load configurations from JSON file if specified
    if args.config:
        with open(args.config, 'r') as f:
            configs = json.load(f)[0]

        # Override defaults with JSON configurations
        for arg, value_list in configs.items():
            if hasattr(args, arg):
                setattr(args, arg, value_list[0])

    return args
