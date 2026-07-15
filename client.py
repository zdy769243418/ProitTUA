import torch
import torch.nn as nn
import numpy as np
from parse import args
from evaluate import evaluate_ndcg, evaluate_hr, evaluate_target_attack, evaluate_mrr


class FedRecClient(nn.Module):
    def __init__(self, client_id, train_ind, test_ind, target_ind, m_item, dim,
                 target_users, target_item_category):
        super().__init__()
        self.client_id = client_id
        self._train_ = train_ind
        self._test_ = test_ind
        self.target_items = list(target_ind)
        self._target_ = []
        self.m_item = m_item
        self.dim = dim
        self.target_users = target_users
        self.target_item_category = target_item_category

        for item in self.target_items:
            if item not in train_ind and item not in test_ind:
                self._target_.append(item)

        items, labels = [], []
        for pos_item in train_ind:
            items.append(pos_item)
            labels.append(1.)

            for _ in range(args.num_neg):
                neg_item = np.random.randint(m_item)
                while neg_item in train_ind:
                    neg_item = np.random.randint(m_item)
                items.append(neg_item)
                labels.append(0.)

        self._train_items = torch.Tensor(items).long()
        self._train_labels = torch.Tensor(labels).to(args.device)
        self._user_emb = nn.Embedding(1, dim)
        nn.init.normal_(self._user_emb.weight, std=args.std)

    def forward(self, items_emb, linear_layers, for_train=False):
        if for_train:
            items_emb = items_emb[self._train_items]
        user_emb = self._user_emb.weight.repeat(len(items_emb), 1)
        value = torch.cat((user_emb, items_emb), dim=-1)

        for i, (weight, bias) in enumerate(linear_layers):
            value = value @ weight.t() + bias
            if i < len(linear_layers) - 1:
                value = value.relu()
            else:
                value = value.sigmoid()
        return value.view(-1)

    def train_(self, items_emb, linear_layers, epoch, items_emb_start_attack):
        items_emb = items_emb.clone().detach().requires_grad_(True)
        linear_layers = [(weight.clone().detach().requires_grad_(True),
                          bias.clone().detach().requires_grad_(True))
                         for (weight, bias) in linear_layers]
        self._user_emb.zero_grad()

        predictions = self.forward(items_emb, linear_layers, for_train=True)
        loss = nn.BCELoss()(predictions, self._train_labels)
        loss.backward()

        user_emb_grad = self._user_emb.weight.grad
        self._user_emb.weight.data.add_(user_emb_grad, alpha=-args.lr)
        items_emb_grad = items_emb.grad[self._train_items]
        linear_layers_grad = [[weight.grad, bias.grad] for (weight, bias) in linear_layers]

        return self._train_items, items_emb_grad, linear_layers_grad, loss.cpu().item()

    def eval_(self, items_emb, linear_layers, epoch):
        rating = self.forward(items_emb, linear_layers)
        rating[self._train_] = - (1 << 10)

        if self._test_:
            r_hr = evaluate_hr(rating, self._test_, args.top_k_rec)
            r_ndcg = evaluate_ndcg(rating, self._test_, args.top_k_rec)
            r_mrr = evaluate_mrr(rating, self._test_, args.top_k_rec)
            test_result = np.array([r_hr, r_ndcg, r_mrr])
            rating[self._test_] = - (1 << 10)
        else:
            test_result = None

        target_in_rec = evaluate_target_attack(rating, self._target_, args.top_k_rec, self.client_id)
        rating_np = rating.cpu().numpy() if torch.is_tensor(rating) else rating
        sorted_indices = np.argsort(rating_np)[::-1]

        total_rank = 0
        for item_id in self.target_items:
            if item_id < len(sorted_indices):
                rank = np.where(sorted_indices == item_id)[0]
                if len(rank) > 0:
                    total_rank += rank[0] + 1
                else:
                    print(f"物品ID {item_id} 不在推荐列表中")
            else:
                print(f"物品ID {item_id} 超出索引范围")

        user_label = 1 if self.client_id in self.target_users[self.target_item_category] else 0
        return test_result, target_in_rec, (user_label, total_rank / len(self.target_items))
