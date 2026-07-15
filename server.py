import torch
import torch.nn as nn

from parse import args
from data import load_target_users


class FedRecServer(nn.Module):
    def __init__(self, m_item, dim, layers, target_items, num_clients, target_users_file,
                 item_categories_attack):
        super().__init__()
        self.m_item = m_item
        self.dim = dim
        self.layers = layers
        self.target_items = target_items
        self.num_clients = num_clients
        self.target_users = load_target_users(target_users_file)
        self.item_categories_attack = item_categories_attack

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, std=args.std)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([nn.Linear(layers_dim[i - 1], layers_dim[i])
                                            for i in range(1, len(layers_dim))])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

        self.suspicious_clients = set()
        self.item_suspicion_scores = torch.zeros(m_item, device=args.device)
        self.previous_item_grad_norms = None
        self.current_defense_epoch = None
        self.current_item_grad_sum = None
        self.current_client_item_norms = None

    def defense_enabled(self):
        return args.defense_strategy == 'GradientDynamics'

    def reset_defense_epoch(self, epoch):
        self.current_defense_epoch = epoch
        self.current_item_grad_sum = torch.zeros_like(self.items_emb.weight)
        self.current_client_item_norms = {item: [] for item in range(self.m_item)}

    def record_client_gradients(self, client_idx, items, items_emb_grad):
        if not self.defense_enabled():
            return

        item_ids = items.detach().cpu().tolist()
        grad = items_emb_grad.detach()
        self.current_item_grad_sum[items] += grad
        grad_norms = torch.norm(grad, dim=1).detach().cpu().tolist()
        for item_id, grad_norm in zip(item_ids, grad_norms):
            self.current_client_item_norms[item_id].append((client_idx, grad_norm))

    def apply_defense_after_epoch(self, epoch, num_clients):
        if not self.defense_enabled() or self.current_item_grad_sum is None:
            return

        item_grad_norms = torch.norm(self.current_item_grad_sum, dim=1)
        if self.previous_item_grad_norms is None:
            self.previous_item_grad_norms = item_grad_norms.detach()
            return

        if epoch <= args.launch:
            self.previous_item_grad_norms = item_grad_norms.detach()
            return

        fluctuation = torch.square(item_grad_norms - self.previous_item_grad_norms)
        self.item_suspicion_scores = torch.maximum(self.item_suspicion_scores, fluctuation.detach())
        self.previous_item_grad_norms = item_grad_norms.detach()

        suspicious_item_count = min(args.defense_suspicious_items, self.m_item)
        suspicious_items = torch.topk(self.item_suspicion_scores, suspicious_item_count).indices.cpu().tolist()
        dominant_client_count = max(int(num_clients * args.defense_client_ratio), 1)

        dominant_client_sets = []
        for item in suspicious_items:
            client_norms = self.current_client_item_norms.get(item, [])
            if not client_norms:
                dominant_client_sets.append(set())
                continue
            client_norms.sort(key=lambda x: x[1], reverse=True)
            dominant_client_sets.append({idx for idx, _ in client_norms[:dominant_client_count]})

        if dominant_client_sets:
            self.suspicious_clients = set.intersection(*dominant_client_sets)
        else:
            self.suspicious_clients = set()

    def train_(self, clients, batch_clients_idx, epoch, items_emb_start_attack):
        items_emb = self.items_emb.weight
        linear_layers = [[layer.weight, layer.bias] for layer in self.linear_layers]
        batch_loss = []

        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]
        batch_items_emb_grad = torch.zeros_like(items_emb)

        if self.defense_enabled() and self.current_defense_epoch != epoch:
            self.reset_defense_epoch(epoch)

        for idx in batch_clients_idx:
            if self.defense_enabled() and idx in self.suspicious_clients:
                continue

            client = clients[idx]
            items, items_emb_grad, linear_layers_grad, loss = client.train_(
                items_emb, linear_layers, epoch, items_emb_start_attack)

            self.record_client_gradients(idx, items, items_emb_grad)

            if loss is not None:
                batch_loss.append(loss)
            batch_items_emb_grad[items] += items_emb_grad

            for i in range(len(linear_layers)):
                batch_linear_layers_grad[i][0] += linear_layers_grad[i][0]
                batch_linear_layers_grad[i][1] += linear_layers_grad[i][1]

        with torch.no_grad():
            self.items_emb.weight.data.add_(batch_items_emb_grad, alpha=-args.lr)
            for i in range(len(linear_layers)):
                self.linear_layers[i].weight.data.add_(batch_linear_layers_grad[i][0], alpha=-args.lr)
                self.linear_layers[i].bias.data.add_(batch_linear_layers_grad[i][1], alpha=-args.lr)

        return self.items_emb.weight.clone().detach(), batch_loss

    def eval_(self, clients, epoch):

        items_emb = self.items_emb.weight
        linear_layers = [(layer.weight, layer.bias) for layer in self.linear_layers]
        test_cnt, test_results_hr, test_results_ndcg, test_results_mrr = 0, 0., 0., 0.

        with torch.no_grad():

            # 用于存储每个 target_item 出现在哪些用户的推荐列表中
            target_item_users = {target_item: set() for target_item in self.target_items}

            avg_rank_target_users = []
            avg_rank_non_target_users = []

            for client in clients:
                test_result, target_in_rec, avg_rank = client.eval_(items_emb, linear_layers, epoch)
                if avg_rank is not None:
                    if avg_rank[0] == 1:
                        avg_rank_target_users.append(avg_rank[1])
                    else:
                        avg_rank_non_target_users.append(avg_rank[1])

                if test_result is not None:
                    test_cnt += 1
                    test_results_hr += test_result[0]
                    test_results_ndcg += test_result[1]
                    test_results_mrr += test_result[2]

                    # 提取用户 ID
                if target_in_rec is not None:
                    user_id = target_in_rec[0]
                    # 提取推荐的目标物品 ID
                    recommended_target_items = target_in_rec[1:]
                    for target_item in recommended_target_items:
                        if target_item in self.target_items:
                            target_item_users[target_item].add(user_id)

            AP = 0.0
            TCR = 0.0
            AER = 0.0
            ACR = 0.0

            for item in self.target_items:
                category = self.item_categories_attack[item]
                hit = self.target_users[category].intersection(target_item_users[item])

                if len(target_item_users[item]) == 0:
                    AP += 0.0

                    TCR += 0.0

                    ACR += 0.0

                else:
                    AP += len(hit) / (len(target_item_users[item]))

                    TCR += len(hit) / (len(self.target_users[category]))

                    ACR += len(target_item_users[item]) / self.num_clients

                if len(self.target_users[category]) == self.num_clients:
                    AER += 0.0
                else:
                    AER += (len(target_item_users[item]) - len(hit)) / (
                            self.num_clients - len(self.target_users[category]))

            return test_results_hr / test_cnt, test_results_ndcg / test_cnt, test_results_mrr / test_cnt, AP / len(
                self.target_items), TCR / len(self.target_items), AER / len(self.target_items), ACR / len(
                self.target_items), sum(avg_rank_target_users) / len(avg_rank_target_users), sum(
                avg_rank_non_target_users) / len(avg_rank_non_target_users)
