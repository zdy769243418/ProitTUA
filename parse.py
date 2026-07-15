import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")
    parser.add_argument('--dim', type=int, default=32, help='Dim of latent vectors.')
    parser.add_argument('--layers', nargs='?', default='[32,16]', help="Dim of mlp layers.")
    parser.add_argument('--num_neg', type=int, default=4, help='Number of negative items.')
    parser.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ML-1M', help='Choose a dataset.')
    parser.add_argument('--device', nargs='?', default='cpu' if cuda.is_available() else 'cpu',
                        help='Which device to run the model.')

    parser.add_argument('--target_item_ids', type=int, nargs='+', default=[3225, 3536, 3435, 3533],
                        help='Sampled target item ids used by the attack.')
    parser.add_argument('--target_item_category', type=int, default=4,
                        help='Category id shared by the target items; used to select target users.')
    parser.add_argument('--target_users_file', nargs='?', default='target_user.json',
                        help='JSON file containing target users by category.')
    parser.add_argument('--item_categories_file', nargs='?', default='Data/ML-1M/mapped_item_categories.txt',
                        help='Attack-domain item-to-category mapping file.')
    parser.add_argument('--category_mapping_file', nargs='?', default='Data/ML-Au/category_mapping.txt',
                        help='Category-name-to-id mapping file shared by attack and auxiliary domains.')
    parser.add_argument('--result_dir', nargs='?', default='Result', help='Output directory for training logs.')

    parser.add_argument('--auxiliary_train_data_path', nargs='?', default='Data/ML-Au/train.dat',
                        help='Auxiliary-domain train data path.')
    parser.add_argument('--auxiliary_test_data_path', nargs='?', default='Data/ML-Au/test.dat',
                        help='Auxiliary-domain test data path.')
    parser.add_argument('--auxiliary_item_categories_file', nargs='?', default='Data/ML-Au/mapped_item_categories.txt',
                        help='Auxiliary-domain item-to-category mapping file.')
    parser.add_argument('--num_categories', type=int, default=20,
                        help='Number of item categories shared by both domains.')
    parser.add_argument('--auxiliary_negative_samples', type=int, default=4,
                        help='Number of negative samples for auxiliary recommendation training.')
    parser.add_argument('--auxiliary_stage1_epochs', type=int, default=2,
                        help='Local auxiliary training epochs per benign warm-up round.')
    parser.add_argument('--auxiliary_stage2_match_epochs', type=int, default=20,
                        help='Fine-tuning epochs for the auxiliary category matching network.')
    parser.add_argument('--auxiliary_stage2_classification_epochs', type=int, default=10,
                        help='Fine-tuning epochs for attack-domain category prediction.')
    parser.add_argument('--auxiliary_stage2_sample_items', type=int, default=32,
                        help='Auxiliary items sampled per attack-domain item during Stage 2 classification.')
    parser.add_argument('--mined_popular_items', type=int, default=4,
                        help='Number of mined popular items kept for each predicted category.')
    parser.add_argument('--auxiliary_verbose', action='store_true',
                        help='Print auxiliary-domain mining diagnostics.')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--std', type=float, default=0.01, help='Embedding initialization std.')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    parser.add_argument('--launch', type=int, default=8, help='The epoch of attack launch.')

    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--clients_limit', type=float, default=0.005, help='Limit of proportion of malicious clients.')
    parser.add_argument('--attack_popular_factor', type=float, default=2.1,
                        help='Initial weight applied to the mined popular-item embedding in attack gradients.')
    parser.add_argument('--attack_grad_scale', type=float, default=6.0,
                        help='Initial scale factor applied to attack gradients.')
    parser.add_argument('--attack_decay_ratio', type=float, default=0.4,
                        help='Final ratio of attack strength after cosine decay.')
    parser.add_argument('--top_k_rec', type=int, default=10, help='length of recommendation list.')

    args = parser.parse_args()
    return args


args = parse_args()
