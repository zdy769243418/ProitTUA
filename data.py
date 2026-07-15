import os
import json


def load_item_categories(file_path, mapping_file_path):
    item_category_map = {}
    item_ids = []
    categories = set()

    if os.path.exists(mapping_file_path):
        with open(mapping_file_path, 'r', encoding='utf-8') as map_file:
            category_mapping = {}
            for line in map_file:
                category, index = line.strip().split(':')
                category_mapping[category] = int(index)
    else:
        category_mapping = {}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            item_id, category = line.strip().split(' ')
            item_id = int(item_id)
            item_category_map[item_id] = category
            item_ids.append(item_id)
            categories.add(category)

    item_ids.sort()

    if not os.path.exists(mapping_file_path):
        for index, category in enumerate(sorted(categories)):
            category_mapping[category] = index
        with open(mapping_file_path, 'w', encoding='utf-8') as map_file:
            for category, index in category_mapping.items():
                map_file.write(f"{category}:{index}\n")

    item_categories = {}
    for item_id in item_ids:
        category = item_category_map[item_id]
        item_categories[item_id] = category_mapping[category]

    return item_categories


def load_target_users(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        target_users = json.load(file)

    return {int(category): set(users) for category, users in target_users.items()}


def load_file(file_path):
    m_item, all_pos = 0, []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            pos = list(map(int, line.rstrip().split(' ')))[1:]
            if pos:
                m_item = max(m_item, max(pos) + 1)
            all_pos.append(pos)

    return m_item, all_pos


def load_file_train(file_path):
    m_item, all_pos, client_ids = 0, [], []

    with open(file_path, 'r') as file:
        for line in file.readlines():
            values = list(map(int, line.rstrip().split(' ')))
            client_ids.append(values[0])
            pos = values[1:]
            if pos:
                m_item = max(m_item, max(pos) + 1)
            all_pos.append(pos)

    return client_ids, m_item, all_pos


def load_dataset(path):
    client_ids, train_m_item, all_train_ind = load_file_train(path + '/train.dat')
    test_m_item, all_test_ind = load_file(path + '/test.dat')
    m_item = max(train_m_item, test_m_item)

    return client_ids, m_item, all_train_ind, all_test_ind
