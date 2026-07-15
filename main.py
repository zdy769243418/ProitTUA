import torch
import random
import numpy as np
from pathlib import Path
from time import time

from parse import args
from data import load_dataset, load_item_categories, load_target_users
from client import FedRecClient
from server import FedRecServer
from attack import malicious_client_by_random
from auxiliary_miner import AuxiliaryPopularItemMiner


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    target_items = list(args.target_item_ids)
    item_categories_attack = load_item_categories(args.item_categories_file, args.category_mapping_file)

    client_ids, m_item, all_train_ind, all_test_ind = load_dataset(args.path + args.dataset)

    clients = []

    target_users = load_target_users(args.target_users_file)
    if args.target_item_category not in target_users:
        raise ValueError(f'target category {args.target_item_category} not found in {args.target_users_file}')

    for client_id, train_ind, test_ind in zip(client_ids, all_train_ind, all_test_ind):
        clients.append(
            FedRecClient(client_id, train_ind, test_ind, target_items, m_item, args.dim,
                         target_users, args.target_item_category).to(args.device))

    server = FedRecServer(m_item, args.dim, eval(args.layers), target_items, len(clients),
                          args.target_users_file, item_categories_attack).to(args.device)
    auxiliary_miner = AuxiliaryPopularItemMiner(target_items, m_item, args.dim, item_categories_attack)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_suffix = 'Attack' if args.defense_strategy == 'NoDefense' else 'Attack_Defense'
    result = result_dir / f'{args.dataset}_{result_suffix}.txt'
    items_emb_start_attack = []

    with open(result, 'w') as f:
        for epoch in range(1, args.launch + 1):
            t1 = time()
            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []

            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                items_emb, loss = server.train_(clients, batch_clients_idx, epoch, items_emb_start_attack)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            server.apply_defense_after_epoch(epoch, len(clients))
            items_emb_start_attack.append(items_emb)
            auxiliary_miner.train_stage1(epoch, items_emb)

            t2 = time()

            test_result_hr, test_result_ndcg, test_result_mrr, AP, TCR, AER, ACR, rank1, rank2 = server.eval_(clients,
                                                                                                              epoch)

            training_info = ("Iteration %d, loss = %.5f [%.1fs], (%.7f) on test-hr, (%.7f) on test-ndcg, (%.7f) on "
                             "test-mrr, (%.7f) AP, (%.7f) TCR, (%.7f) AER, (%.7f) ACR, (%.7f) target_user_rank, "
                             "(%.7f) non_target_user_rank. [%.1fs]") % (
                                epoch, total_loss, t2 - t1,
                                test_result_hr, test_result_ndcg, test_result_mrr,
                                AP, TCR, AER, ACR, rank1, rank2,
                                time() - t2)

            print(training_info)
            f.write(training_info + '\n')

        popular_items_by_category = auxiliary_miner.mine_popular_items(
            server.items_emb.weight.clone().detach(), items_emb_start_attack)
        malicious_clients_limit = max(int(len(clients) * args.clients_limit), 1)
        clients.extend(malicious_client_by_random(
            malicious_clients_limit, m_item, target_items, popular_items_by_category, item_categories_attack))

        for epoch in range(args.launch + 1, args.epochs + 1):
            t1 = time()

            rand_clients = np.arange(len(clients))
            np.random.shuffle(rand_clients)

            total_loss = []

            for i in range(0, len(rand_clients), args.batch_size):
                batch_clients_idx = rand_clients[i: i + args.batch_size]
                items_emb, loss = server.train_(clients, batch_clients_idx, epoch, items_emb_start_attack)
                total_loss.extend(loss)
            total_loss = np.mean(total_loss).item()

            server.apply_defense_after_epoch(epoch, len(clients))
            t2 = time()

            test_result_hr, test_result_ndcg, test_result_mrr, AP, TCR, AER, ACR, rank1, rank2 = server.eval_(clients,
                                                                                                              epoch)

            training_info = ("Iteration %d, loss = %.5f [%.1fs], (%.7f) on test-hr, (%.7f) on test-ndcg, (%.7f) on "
                             "test-mrr, (%.7f) AP, (%.7f) TCR, (%.7f) AER, (%.7f) ACR, (%.7f) target_user_rank, "
                             "(%.7f) non_target_user_rank. [%.1fs]") % (
                                epoch, total_loss, t2 - t1,
                                test_result_hr, test_result_ndcg, test_result_mrr,
                                AP, TCR, AER, ACR, rank1, rank2,
                                time() - t2)

            print(training_info)
            f.write(training_info + '\n')


if __name__ == "__main__":
    setup_seed(20220110)
    main()
