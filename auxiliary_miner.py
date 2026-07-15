import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

from data import load_item_categories
from parse import args


class AuxiliaryRecommendationModel(nn.Module):
    def __init__(self, m_user, m_item, dim, layers, std):
        super().__init__()
        self.user_emb = nn.Embedding(m_user, dim)
        nn.init.normal_(self.user_emb.weight, std=std)

        self.items_emb = nn.Embedding(m_item, dim)
        nn.init.normal_(self.items_emb.weight, mean=0.5, std=std)

        layers_dim = [2 * dim] + layers + [1]
        self.linear_layers = nn.ModuleList([
            nn.Linear(layers_dim[i - 1], layers_dim[i])
            for i in range(1, len(layers_dim))
        ])
        for layer in self.linear_layers:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_emb(user_ids)
        item_emb = self.items_emb(item_ids)
        value = torch.cat((user_emb, item_emb), dim=-1)

        for i, layer in enumerate(self.linear_layers):
            value = value @ layer.weight.t() + layer.bias
            if i < len(self.linear_layers) - 1:
                value = value.relu()
            else:
                value = value.sigmoid()
        return value.view(-1)


class SimilarityNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size * 2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 1)
        self._initialize_weights()

    def forward(self, x1, x2):
        x1 = normalize_features(x1)
        x2 = normalize_features(x2)
        value = torch.cat((x1, x2), dim=1)
        value = F.relu(self.bn1(self.fc1(value)))
        value = self.dropout1(value)
        value = F.relu(self.bn2(self.fc2(value)))
        value = self.dropout2(value)
        return self.fc3(value)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class ClassifierNetwork(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, num_classes)
        self._initialize_weights()

    def forward(self, value):
        value = normalize_features(value)
        value = F.relu(self.bn1(self.fc1(value)))
        value = self.dropout1(value)
        value = F.relu(self.bn2(self.fc2(value)))
        value = self.dropout2(value)
        return self.fc3(value)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        if self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss


class ItemPairDataset(Dataset):
    def __init__(self, item_categories):
        self.pairs = []
        self.labels = []
        category_items = {}
        for item_id, category in item_categories.items():
            category_items.setdefault(category, []).append(item_id)

        for item_id, category in item_categories.items():
            same_category_items = category_items[category]
            different_category_items = []
            for other_category, items in category_items.items():
                if other_category != category:
                    different_category_items.extend(items)

            for _ in range(2):
                if len(same_category_items) > 1:
                    other_item_id = random.choice([i for i in same_category_items if i != item_id])
                    self.pairs.append((item_id, other_item_id))
                    self.labels.append(1.0)

            for _ in range(2):
                if different_category_items:
                    other_item_id = random.choice(different_category_items)
                    self.pairs.append((item_id, other_item_id))
                    self.labels.append(0.0)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        item1_id, item2_id = self.pairs[idx]
        return item1_id, item2_id, torch.tensor(self.labels[idx], dtype=torch.float32)


class ItemPairDatasetForMatch(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item1_id, item2_id, label = self.data[idx]
        return item1_id, item2_id, label


class ItemPairDatasetForClassification(Dataset):
    def __init__(self, batch_data, attack_item_emb, auxiliary_item_emb):
        self.batch_data = batch_data
        self.attack_item_emb = attack_item_emb
        self.auxiliary_item_emb = auxiliary_item_emb

    def __len__(self):
        return len(self.batch_data)

    def __getitem__(self, idx):
        attack_item_id, auxiliary_item_id = self.batch_data[idx]
        return self.attack_item_emb[attack_item_id], self.auxiliary_item_emb[auxiliary_item_id]


def normalize_features(features):
    mean = features.mean(dim=1, keepdim=True)
    std = features.std(dim=1, keepdim=True)
    return (features - mean) / (std + 1e-8)


def read_raw_item_categories(file_path):
    item_categories = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item_id, category = line.strip().split()
            item_categories[int(item_id)] = category
    return item_categories


def get_same_and_different_categories(auxiliary_categories_file, attack_categories_file, target_items):
    auxiliary_categories = read_raw_item_categories(auxiliary_categories_file)
    attack_categories = read_raw_item_categories(attack_categories_file)
    result = {}
    for item_id, category in auxiliary_categories.items():
        same_type = [target_id for target_id in target_items if attack_categories[target_id] == category]
        different_type = [other_id for other_id, other_category in auxiliary_categories.items()
                          if other_category != category][:16]
        result[item_id] = {
            'positive': same_type,
            'negative': different_type,
            'label': 0 if same_type else 1,
        }
    return result


def custom_loss(kl_div_sum_batch, similarity_prob_batch, max_threshold=0.8, min_threshold=0.2):
    predicted_prob_batch = torch.sigmoid(-kl_div_sum_batch / 5)
    predicted_prob_batch = torch.clamp(predicted_prob_batch, 1e-8, 1 - 1e-8)

    loss = torch.zeros_like(similarity_prob_batch)
    mask_max = similarity_prob_batch > max_threshold
    mask_min = similarity_prob_batch < min_threshold
    loss[mask_max] = -torch.log(predicted_prob_batch[mask_max])
    loss[mask_min] = -torch.log(1 - predicted_prob_batch[mask_min])

    valid_mask = mask_max | mask_min
    valid_loss = loss[valid_mask]
    if valid_loss.numel() == 0:
        return kl_div_sum_batch.sum() * 0.0
    return valid_loss.mean()


def chamfer_distance(v_x, v_y):
    distances_x_to_y = torch.sqrt(((v_x.unsqueeze(1) - v_y) ** 2).sum(dim=2))
    min_distances_x_to_y = distances_x_to_y.max(dim=1)[0]

    distances_y_to_x = torch.sqrt(((v_y.unsqueeze(1) - v_x) ** 2).sum(dim=2))
    min_distances_y_to_x = distances_y_to_x.max(dim=1)[0]

    return 0.5 * (min_distances_x_to_y.sum() / v_x.shape[0]
                  + min_distances_y_to_x.sum() / v_y.shape[0])


def linear_interpolation_augmentation(positive_embeddings, num_new_embeddings,
                                      alpha_min=0.1, alpha_max=0.9, mean=0, std=0.01):
    if not positive_embeddings:
        return []

    new_embeddings = []
    if len(positive_embeddings) < 2:
        for _ in range(num_new_embeddings):
            noise = positive_embeddings[0] + torch.randn_like(positive_embeddings[0]) * std + mean
            new_embeddings.append(noise)
        return new_embeddings

    for _ in range(num_new_embeddings):
        i, j = torch.randperm(len(positive_embeddings))[:2]
        alpha = (alpha_max - alpha_min) * torch.rand(1).item() + alpha_min
        new_embeddings.append((1 - alpha) * positive_embeddings[i] + alpha * positive_embeddings[j])
    return new_embeddings


def load_auxiliary_train_data(file_path, num_neg_samples, batch_size):
    all_data = []
    max_user_id = -1
    max_item_id = -1
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            all_data.append((user_id, item_ids))
            max_user_id = max(max_user_id, user_id)
            if item_ids:
                max_item_id = max(max_item_id, max(item_ids))

    preprocessed_data = []
    for user_id, positive_items in all_data:
        items = []
        labels = []
        positive_set = set(positive_items)
        for pos_item in positive_items:
            items.append(pos_item)
            labels.append(1.0)
            for _ in range(num_neg_samples):
                neg_item = np.random.randint(max_item_id + 1)
                while neg_item in positive_set:
                    neg_item = np.random.randint(max_item_id + 1)
                items.append(neg_item)
                labels.append(0.0)
        preprocessed_data.append((user_id, items, labels))

    data = []
    num_batches = len(preprocessed_data) // batch_size
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch_data = preprocessed_data[start_idx:end_idx]
        batch_users = []
        batch_items = []
        batch_labels = []
        for user_id, items, labels in batch_data:
            batch_users.extend([user_id] * len(items))
            batch_items.extend(items)
            batch_labels.extend(labels)
        data.append((batch_users, batch_items, batch_labels))

    return data, max_user_id + 1, max_item_id + 1


def load_auxiliary_test_data(file_path):
    data = []
    max_user_id = -1
    max_item_id = -1
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split()
            user_id = int(parts[0])
            item_ids = [int(x) for x in parts[1:]]
            data.append((user_id, item_ids))
            max_user_id = max(max_user_id, user_id)
            if item_ids:
                max_item_id = max(max_item_id, max(item_ids))
    return data, max_user_id + 1, max_item_id + 1


class AuxiliaryPopularItemMiner:
    def __init__(self, target_items, attack_m_item, dim, attack_item_categories):
        self.target_items = list(target_items)
        self.attack_m_item = attack_m_item
        self.dim = dim
        self.attack_item_categories = attack_item_categories
        self.device = torch.device(args.device)

        self.aux_train_data, train_user_count, train_item_count = load_auxiliary_train_data(
            args.auxiliary_train_data_path, args.auxiliary_negative_samples, args.batch_size)
        _, test_user_count, test_item_count = load_auxiliary_test_data(
            args.auxiliary_test_data_path)

        mapped_item_count = max(
            int(line.split()[0]) for line in open(args.auxiliary_item_categories_file, encoding='utf-8')) + 1
        self.aux_user_count = max(train_user_count, test_user_count)
        self.aux_item_count = max(train_item_count, test_item_count, mapped_item_count)

        self.aux_item_categories = load_item_categories(
            args.auxiliary_item_categories_file, args.category_mapping_file)
        self.aux_raw_categories = read_raw_item_categories(args.auxiliary_item_categories_file)
        self.attack_raw_categories = read_raw_item_categories(args.item_categories_file)
        self.same_category_result = get_same_and_different_categories(
            args.auxiliary_item_categories_file, args.item_categories_file, self.target_items)

        self.aux_model = AuxiliaryRecommendationModel(
            self.aux_user_count, self.aux_item_count, dim, eval(args.layers), args.std).to(self.device)
        self.aux_criterion = nn.BCELoss().to(self.device)
        self.aux_optimizer = optim.Adam(self.aux_model.parameters(), lr=args.lr)

        self.classification_network = ClassifierNetwork(dim, args.num_categories).to(self.device)
        self.classification_optimizer = optim.Adam(
            self.classification_network.parameters(), lr=args.lr * 0.01, weight_decay=0.001)
        self.classification_criterion = FocalLoss()

        self.category_match_network = SimilarityNetwork(dim).to(self.device)
        self.category_match_optimizer = optim.Adam(
            self.category_match_network.parameters(), lr=args.lr, weight_decay=0.001)
        self.category_match_criterion = nn.BCEWithLogitsLoss()

        self.new_classification_network = ClassifierNetwork(dim, args.num_categories).to(self.device)
        self.new_classification_optimizer = optim.Adam(
            self.new_classification_network.parameters(), lr=args.lr * 0.01)
        self.new_category_match_network = SimilarityNetwork(dim).to(self.device)
        self.new_category_match_optimizer = optim.Adam(
            self.new_category_match_network.parameters(), lr=args.lr)

    def train_stage1(self, epoch, attack_domain_item_embeddings):
        if epoch > args.launch:
            return

        attack_domain_item_embeddings = attack_domain_item_embeddings.detach().to(self.device)
        self.aux_model.train()
        self.classification_network.train()
        self.category_match_network.train()
        random.shuffle(self.aux_train_data)

        item_id_batches_align = self._build_alignment_batches()
        item_id_batches_classification = self._build_classification_batches()
        match_loader = self._build_match_loader()

        for local_epoch in range(args.auxiliary_stage1_epochs):
            match_iter = iter(match_loader)
            total_rec_loss = 0.0
            total_align_loss = 0.0
            total_classification_loss = 0.0
            total_match_loss = 0.0
            trained_batches = 0

            for batch_idx, batch in enumerate(self.aux_train_data):
                users = torch.tensor(batch[0], dtype=torch.long, device=self.device)
                items = torch.tensor(batch[1], dtype=torch.long, device=self.device)
                labels = torch.tensor(batch[2], dtype=torch.float32, device=self.device)

                self.aux_optimizer.zero_grad()
                self.classification_optimizer.zero_grad()

                output = self.aux_model(users, items).float()
                rec_loss = self.aux_criterion(output, labels)
                align_loss = self._alignment_loss(
                    item_id_batches_align[batch_idx], attack_domain_item_embeddings)
                classification_loss = self._classification_loss(item_id_batches_classification[batch_idx])

                total_loss = rec_loss + align_loss * 5 + classification_loss * 0.1
                total_loss.backward()
                self.aux_optimizer.step()
                self.classification_optimizer.step()

                total_rec_loss += rec_loss.item()
                total_align_loss += align_loss.item()
                total_classification_loss += classification_loss.item()

                try:
                    item1_ids, item2_ids, match_labels = next(match_iter)
                except StopIteration:
                    match_iter = iter(match_loader)
                    item1_ids, item2_ids, match_labels = next(match_iter)
                match_loss = self._train_match_batch(item1_ids, item2_ids, match_labels)
                total_match_loss += match_loss
                trained_batches += 1

            if args.auxiliary_verbose and trained_batches:
                print(
                    f'Auxiliary Stage 1 epoch {epoch}.{local_epoch + 1}: '
                    f'rec={total_rec_loss / trained_batches:.6f}, '
                    f'align={total_align_loss / trained_batches:.6f}, '
                    f'cls={total_classification_loss / trained_batches:.6f}, '
                    f'match={total_match_loss / trained_batches:.6f}'
                )

    def mine_popular_items(self, attack_domain_item_embeddings, items_emb_start_attack):
        if not items_emb_start_attack:
            raise ValueError('items_emb_start_attack is empty; run benign warm-up epochs before mining.')

        attack_domain_item_embeddings = attack_domain_item_embeddings.detach().to(self.device)
        self._fine_tune_stage2(attack_domain_item_embeddings)
        predicted_categories = self._predict_attack_categories(attack_domain_item_embeddings)
        popular_items = self._select_popular_items_by_embedding_shift(
            predicted_categories, items_emb_start_attack)
        self._ensure_target_category_items(popular_items, items_emb_start_attack)
        if args.auxiliary_verbose:
            print('mined popular items:', popular_items)
        return popular_items

    def _build_alignment_batches(self):
        batch_count = len(self.aux_train_data)
        batch_size = self.aux_item_count // batch_count
        batches = []
        for batch_idx in range(batch_count):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size if batch_idx < batch_count - 1 else self.aux_item_count
            batches.append(torch.arange(start_idx, end_idx, dtype=torch.long, device=self.device))
        return batches

    def _build_classification_batches(self):
        item_ids = list(self.aux_item_categories.keys())
        random.shuffle(item_ids)
        batch_count = len(self.aux_train_data)
        batch_size = len(item_ids) // batch_count
        remainder = len(item_ids) % batch_count
        batches = []
        start_idx = 0
        for batch_idx in range(batch_count):
            end_idx = start_idx + batch_size + (1 if batch_idx < remainder else 0)
            batches.append(torch.tensor(item_ids[start_idx:end_idx], dtype=torch.long, device=self.device))
            start_idx = end_idx
        return batches

    def _build_match_loader(self):
        dataset = ItemPairDataset(self.aux_raw_categories)
        batch_size = max(len(dataset) // max(len(self.aux_train_data), 1), 1)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    def _alignment_loss(self, current_item_ids, attack_domain_item_embeddings):
        item_embeddings_align = self.aux_model.items_emb(current_item_ids)
        coarse_loss = chamfer_distance(item_embeddings_align, attack_domain_item_embeddings)
        fine_loss = self._contrastive_loss(current_item_ids, attack_domain_item_embeddings)
        return coarse_loss + fine_loss

    def _contrastive_loss(self, current_item_ids, attack_domain_item_embeddings):
        total_loss = torch.tensor(0.0, device=self.device)
        count = 0
        for item_id in current_item_ids:
            item_key = item_id.item()
            if self.same_category_result[item_key]['label'] != 0:
                continue
            positive_embeddings = [
                attack_domain_item_embeddings[pos_id].unsqueeze(0)
                for pos_id in self.same_category_result[item_key]['positive']
            ]
            positive_embeddings.extend(linear_interpolation_augmentation(positive_embeddings, 8))
            if not positive_embeddings:
                continue

            item_emb = self.aux_model.items_emb(item_id).unsqueeze(0)
            pos_distances = [F.pairwise_distance(item_emb, pos_emb) for pos_emb in positive_embeddings]
            total_loss += torch.exp(torch.mean(torch.stack(pos_distances)) * 2)
            count += 1

        if count == 0:
            return total_loss
        return total_loss / count

    def _classification_loss(self, current_item_ids):
        labels = torch.tensor(
            [self.aux_item_categories[item_id.item()] for item_id in current_item_ids],
            dtype=torch.long,
            device=self.device,
        )
        item_embeddings = self.aux_model.items_emb(current_item_ids)
        classification_output = self.classification_network(item_embeddings)
        return self.classification_criterion(classification_output, labels)

    def _train_match_batch(self, item1_ids, item2_ids, labels):
        self.category_match_optimizer.zero_grad()
        for param in self.aux_model.items_emb.parameters():
            param.requires_grad = False

        item1_ids = item1_ids.to(self.device)
        item2_ids = item2_ids.to(self.device)
        labels = labels.float().to(self.device)
        item1_features = self.aux_model.items_emb(item1_ids)
        item2_features = self.aux_model.items_emb(item2_ids)
        outputs = self.category_match_network(item1_features, item2_features)
        match_loss = self.category_match_criterion(outputs.squeeze(), labels)
        match_loss.backward()
        self.category_match_optimizer.step()

        for param in self.aux_model.items_emb.parameters():
            param.requires_grad = True
        return match_loss.item()

    def _fine_tune_stage2(self, attack_domain_item_embeddings):
        self.new_classification_network.load_state_dict(self.classification_network.state_dict())
        self.new_category_match_network.load_state_dict(self.category_match_network.state_dict())
        self._fine_tune_match_network(attack_domain_item_embeddings)
        self._fine_tune_classification_network(attack_domain_item_embeddings)

    def _fine_tune_match_network(self, attack_domain_item_embeddings):
        batch_data = []
        for target_id in self.target_items:
            target_category = self.attack_raw_categories[target_id]
            positive_samples = [item_id for item_id, category in self.aux_raw_categories.items()
                                if category == target_category]
            negative_samples = [item_id for item_id, category in self.aux_raw_categories.items()
                                if category != target_category]
            for positive_item_id in positive_samples:
                batch_data.append((target_id, positive_item_id, 1.0))
            sample_count = min(len(positive_samples), len(negative_samples))
            for negative_item_id in random.sample(negative_samples, sample_count):
                batch_data.append((target_id, negative_item_id, 0.0))

        loader = DataLoader(
            ItemPairDatasetForMatch(batch_data), batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.new_category_match_network.train()
        auxiliary_item_embeddings = self.aux_model.items_emb.weight.detach()

        for _ in range(args.auxiliary_stage2_match_epochs):
            for item1_ids, item2_ids, labels in loader:
                item1_ids = item1_ids.to(self.device)
                item2_ids = item2_ids.to(self.device)
                labels = labels.float().to(self.device)
                item1_features = attack_domain_item_embeddings[item1_ids]
                item2_features = auxiliary_item_embeddings[item2_ids]
                outputs = self.new_category_match_network(item1_features, item2_features)
                loss = self.category_match_criterion(outputs.squeeze(), labels)
                self.new_category_match_optimizer.zero_grad()
                loss.backward()
                self.new_category_match_optimizer.step()

    def _fine_tune_classification_network(self, attack_domain_item_embeddings):
        auxiliary_item_ids = list(self.aux_raw_categories.keys())
        pair_data = []
        sample_size = min(args.auxiliary_stage2_sample_items, len(auxiliary_item_ids))
        for attack_item_id in range(self.attack_m_item):
            for auxiliary_item_id in random.sample(auxiliary_item_ids, sample_size):
                pair_data.append((attack_item_id, auxiliary_item_id))

        loader = DataLoader(
            ItemPairDatasetForClassification(
                pair_data, attack_domain_item_embeddings, self.aux_model.items_emb.weight.detach()),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.new_category_match_network.eval()
        self.new_classification_network.train()

        for _ in range(args.auxiliary_stage2_classification_epochs):
            for attack_item_emb, auxiliary_item_emb in loader:
                attack_item_emb = attack_item_emb.to(self.device)
                auxiliary_item_emb = auxiliary_item_emb.to(self.device)
                with torch.no_grad():
                    similarity_prob = torch.sigmoid(
                        self.new_category_match_network(attack_item_emb, auxiliary_item_emb).squeeze())

                attack_category_probs = F.softmax(self.new_classification_network(attack_item_emb), dim=1)
                auxiliary_category_probs = F.softmax(self.new_classification_network(auxiliary_item_emb), dim=1)
                kl_div_12 = F.kl_div(torch.log(attack_category_probs + 1e-8),
                                     auxiliary_category_probs, reduction='none')
                kl_div_21 = F.kl_div(torch.log(auxiliary_category_probs + 1e-8),
                                     attack_category_probs, reduction='none')
                kl_div_sum = (kl_div_12 + kl_div_21) / 2
                loss = custom_loss(kl_div_sum.sum(dim=1), similarity_prob)

                self.new_classification_optimizer.zero_grad()
                loss.backward()
                self.new_classification_optimizer.step()

    def _predict_attack_categories(self, attack_domain_item_embeddings):
        self.new_classification_network.eval()
        with torch.no_grad():
            logits = self.new_classification_network(attack_domain_item_embeddings)
            _, predictions = torch.max(F.softmax(logits, dim=1), dim=1)

        categories = {category: [] for category in range(args.num_categories)}
        for item_index, category in enumerate(predictions.tolist()):
            categories[category].append(item_index)
        return categories

    def _select_popular_items_by_embedding_shift(self, predicted_categories, items_emb_start_attack):
        start_embeddings = items_emb_start_attack[0].detach().to(self.device)
        end_embeddings = items_emb_start_attack[-1].detach().to(self.device)
        popular_items = {category: [] for category in range(args.num_categories)}

        for category, item_indices in predicted_categories.items():
            if not item_indices:
                continue
            item_indices_tensor = torch.tensor(item_indices, dtype=torch.long, device=self.device)
            accumulated_gradient = end_embeddings[item_indices_tensor] - start_embeddings[item_indices_tensor]
            distances = torch.norm(accumulated_gradient, dim=1)
            top_k = min(args.mined_popular_items, len(item_indices))
            top_indices = torch.argsort(distances, descending=True)[:top_k].tolist()
            popular_items[category] = [item_indices[index] for index in top_indices]
        return popular_items

    def _ensure_target_category_items(self, popular_items, items_emb_start_attack):
        start_embeddings = items_emb_start_attack[0].detach().to(self.device)
        end_embeddings = items_emb_start_attack[-1].detach().to(self.device)
        for target_item in self.target_items:
            category = self.attack_item_categories[target_item]
            if popular_items.get(category):
                continue
            candidates = [item_id for item_id, item_category in self.attack_item_categories.items()
                          if item_category == category and item_id not in self.target_items]
            if not candidates:
                continue
            candidate_tensor = torch.tensor(candidates, dtype=torch.long, device=self.device)
            distances = torch.norm(end_embeddings[candidate_tensor] - start_embeddings[candidate_tensor], dim=1)
            top_k = min(args.mined_popular_items, len(candidates))
            top_indices = torch.argsort(distances, descending=True)[:top_k].tolist()
            popular_items[category] = [candidates[index] for index in top_indices]
