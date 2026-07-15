import math

import torch
import torch.nn as nn

from parse import args


def get_attack_strength(epoch):
    decay_steps = max(args.epochs - args.launch - 1, 1)
    progress = (epoch - args.launch - 1) / decay_steps
    progress = min(max(progress, 0.0), 1.0)
    ratio = args.attack_decay_ratio + (1 - args.attack_decay_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return args.attack_popular_factor * ratio, args.attack_grad_scale * ratio


class OurAttackClient(nn.Module):
    def __init__(self, target_items, m_item, popular_items_by_category, item_categories_attack):
        super().__init__()
        self._target_ = list(target_items)
        self.m_item = m_item
        self.popular_items_by_category = popular_items_by_category
        self.item_categories_attack = item_categories_attack

    def train_(self, items_emb, linear_layers, epoch, items_emb_start_attack):
        all_items_emb = items_emb.clone().detach().requires_grad_(False)
        batch_items_emb_grad = torch.zeros_like(items_emb)
        batch_linear_layers_grad = [[torch.zeros_like(w), torch.zeros_like(b)] for (w, b) in linear_layers]
        popular_factor, grad_scale = get_attack_strength(epoch)

        for target_item in self._target_:
            target_category = self.item_categories_attack[target_item]
            popular_items = [
                item for item in self.popular_items_by_category.get(target_category, [])
                if item not in self._target_
            ]
            if not popular_items:
                raise ValueError(f'No mined popular items for target category {target_category}.')

            target_emb = all_items_emb[target_item].clone().detach().requires_grad_(True)
            popular_emb = torch.mean(all_items_emb[popular_items], dim=0)
            batch_items_emb_grad[target_item] = (
                target_emb - popular_emb * popular_factor
            ) * grad_scale

        return torch.Tensor(self._target_).long(), batch_items_emb_grad[self._target_], batch_linear_layers_grad, None

    def eval_(self, _items_emb, _linear_layers, epoch):
        return None, None, None


def malicious_client_by_random(num_clients, m_item, target_items, popular_items_by_category, item_categories_attack):
    return [
        OurAttackClient(
            target_items, m_item, popular_items_by_category, item_categories_attack
        ).to(args.device)
        for _ in range(num_clients)
    ]
