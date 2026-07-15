import argparse
import torch.cuda as cuda


def parse_args():
    parser = argparse.ArgumentParser(description="Run Recommender Model.")

    #-------Model-------#
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--dim', type=int, default=32, help='Dim of latent vectors.')
    model_group.add_argument('--layers', nargs='?', default='[32,16]', help='Dim of mlp layers.')
    model_group.add_argument('--num_neg', type=int, default=4, help='Number of negative items.')
    model_group.add_argument('--device', nargs='?', default='cpu' if cuda.is_available() else 'cpu',
                             help='Which device to run the model.')

    #-------Data-------#
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--path', nargs='?', default='Data/', help='Input data path.')
    data_group.add_argument('--dataset', nargs='?', default='ML-1M', help='Choose a dataset.')
    data_group.add_argument('--item_categories_file', nargs='?', default='Data/ML-1M/mapped_item_categories.txt',
                            help='Attack-domain item-to-category mapping file.')
    data_group.add_argument('--category_mapping_file', nargs='?', default='Data/ML-Au/category_mapping.txt',
                            help='Category-name-to-id mapping file shared by attack and auxiliary domains.')
    data_group.add_argument('--result_dir', nargs='?', default='Result', help='Output directory for training logs.')

    #-------Target Items and Users-------#
    target_group = parser.add_argument_group('Target Items and Users')
    target_group.add_argument('--target_item_ids', type=int, nargs='+', default=[3225, 3536, 3435, 3533],
                              help='Sampled target item ids used by the attack.')
    target_group.add_argument('--target_item_category', type=int, default=4,
                              help='Category id shared by the target items; used to select target users.')
    target_group.add_argument('--target_users_file', nargs='?', default='target_user.json',
                              help='JSON file containing target users by category.')

    #-------Auxiliary-Domain Training-------#
    auxiliary_group = parser.add_argument_group('Auxiliary-Domain Training')
    auxiliary_group.add_argument('--auxiliary_train_data_path', nargs='?', default='Data/ML-Au/train.dat',
                                 help='Auxiliary-domain train data path.')
    auxiliary_group.add_argument('--auxiliary_test_data_path', nargs='?', default='Data/ML-Au/test.dat',
                                 help='Auxiliary-domain test data path.')
    auxiliary_group.add_argument('--auxiliary_item_categories_file', nargs='?',
                                 default='Data/ML-Au/mapped_item_categories.txt',
                                 help='Auxiliary-domain item-to-category mapping file.')
    auxiliary_group.add_argument('--num_categories', type=int, default=20,
                                 help='Number of item categories shared by both domains.')
    auxiliary_group.add_argument('--auxiliary_negative_samples', type=int, default=4,
                                 help='Number of negative samples for auxiliary recommendation training.')
    auxiliary_group.add_argument('--auxiliary_stage1_epochs', type=int, default=2,
                                 help='Local auxiliary training epochs per benign warm-up round.')
    auxiliary_group.add_argument('--auxiliary_stage2_match_epochs', type=int, default=20,
                                 help='Fine-tuning epochs for the auxiliary category matching network.')
    auxiliary_group.add_argument('--auxiliary_stage2_classification_epochs', type=int, default=10,
                                 help='Fine-tuning epochs for attack-domain category prediction.')
    auxiliary_group.add_argument('--auxiliary_stage2_sample_items', type=int, default=32,
                                 help='Auxiliary items sampled per attack-domain item during Stage 2 classification.')
    auxiliary_group.add_argument('--mined_popular_items', type=int, default=4,
                                 help='Number of mined popular items kept for each predicted category.')
    auxiliary_group.add_argument('--auxiliary_verbose', action='store_true',
                                 help='Print auxiliary-domain mining diagnostics.')

    #-------Federated Training-------#
    training_group = parser.add_argument_group('Federated Training')
    training_group.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    training_group.add_argument('--std', type=float, default=0.01, help='Embedding initialization std.')
    training_group.add_argument('--epochs', type=int, default=20, help='Number of epochs.')
    training_group.add_argument('--launch', type=int, default=8, help='The epoch of attack launch.')
    training_group.add_argument('--batch_size', type=int, default=256, help='Batch size.')

    #-------Attack-------#
    attack_group = parser.add_argument_group('Attack')
    attack_group.add_argument('--clients_limit', type=float, default=0.005,
                              help='Limit of proportion of malicious clients.')
    attack_group.add_argument('--attack_popular_factor', type=float, default=1.0,
                              help='Initial weight applied to the mined popular-item embedding in attack gradients.')
    attack_group.add_argument('--attack_grad_scale', type=float, default=22.0,
                              help='Initial scale factor applied to attack gradients.')
    attack_group.add_argument('--attack_decay_ratio', type=float, default=0.7,
                              help='Final ratio of attack_grad_scale after cosine decay.')

    #-------Defense-------#
    defense_group = parser.add_argument_group('Defense')
    defense_group.add_argument('--defense_strategy', default='NoDefense', choices=['NoDefense', 'GradientDynamics'],
                               help='Defense strategy used during aggregation.')
    defense_group.add_argument('--defense_suspicious_items', type=int, default=4,
                               help='Top-tau suspicious items selected by gradient dynamics analysis.')
    defense_group.add_argument('--defense_client_ratio', type=float, default=0.05,
                               help='Top-rho client ratio used to trace dominant clients for each suspicious item.')

    #-------Evaluation-------#
    evaluation_group = parser.add_argument_group('Evaluation')
    evaluation_group.add_argument('--top_k_rec', type=int, default=10, help='Length of recommendation list.')

    args = parser.parse_args()
    return args


args = parse_args()
