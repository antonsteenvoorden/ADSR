import csv
import torch
import random
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CustomDataset(Dataset):
    item_x = None
    item_y = None

    users = None
    length = None

    full_sequences = None
    num_item = None

    def __init__(self, item_x, item_y, users, full_sequences=None, num_samples=100, num_item=0, sampling=False):
        self.item_x = item_x
        self.item_y = item_y

        self.users = users
        self.length = len(item_y)

        self.full_sequences = full_sequences  # sequences sorted by user
        self.num_samples = num_samples
        self.num_item = num_item
        self.sampling = sampling

    def sample(self, sequence):
        item = random.randrange(1, self.num_item)
        while item in sequence:
            item = random.randrange(1, self.num_item)
        return item

    def __getitem__(self, index, sampling=False):
        user_id = self.users[index]

        item_x = self.item_x[index]
        item_y = self.item_y[index]

        if self.sampling:
            all_samples = [item_y]
            for _ in range(self.num_samples):  # sample negative
                item = self.sample(self.full_sequences[user_id.item()])
                all_samples.append(item)
        else:
            all_samples = [0]

        x = (item_x, torch.zeros(1), user_id, torch.tensor(all_samples))
        y = (item_y, torch.zeros(1))
        return x, y

    def __len__(self):
        return self.length


def read(path, properties_path=None):
    clean = lambda l: [int(x) for x in l.strip('[]').split(',')]
    with open(path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='|')
        header = next(csv_reader, None)  # skip the header

        item_x, item_y, users = [], [], []
        # Note the pops for the single element lists
        for row in csv_reader:
            (seq, target, user) = row[:3]
            seq = clean(seq)
            target = clean(target).pop()
            user = clean(user).pop()
            item_x.append(seq)
            item_y.append(target)
            users.append(user)

    # as everything is wrapped in lists we need to call max multiple times to extract the actual number
    # we also add 1 to account for the 0 indexing
    item_num = max(max(max(item_x)), (max(item_y))) + 1
    user_num = max(users) + 1

    # Convert to tensors (needed for the dataset to get items correctly)
    item_x = torch.tensor(item_x, dtype=torch.long)
    item_y = torch.tensor(item_y, dtype=torch.long)
    users = torch.tensor(users, dtype=torch.long)
    return item_x, item_y, users, item_num, user_num


def read_full_sequences(path):
    clean = lambda l: [int(x) for x in l.strip("\n").strip('"').strip('[]').split(',')]

    with open(path, 'r') as file:
        full_sequences = [[]]
        for line in file.readlines():
            sequence = clean(line)
            full_sequences.append(sequence)
    return full_sequences


def prepare_dataset(path, properties_path=None, full_sequences=None, sampling=False):
    # context_x and context_y, context_num are tuples, as we might have multiple contexts
    item_x, item_y, users, i_num, u_num = read(path, properties_path=properties_path)
    dataset = CustomDataset(item_x, item_y, users, full_sequences=full_sequences, num_item=i_num, sampling=sampling)
    return dataset, i_num, u_num


def get_data_loaders(train_path, val_path, test_path, batch_size, num_workers=4,
                     shuffle=True, properties_path=None, sampling=False):
    if "ijcai15" in train_path and "buy_2" in train_path:
        print("Loading full sequences 2")
        full_sequences = read_full_sequences("../data/ijcai15/data/sequences_full_2.csv")
    elif "ijcai15" in train_path:
        print("Loading full sequences 1")
        full_sequences = read_full_sequences("../data/ijcai15/data/sequences_full.csv")
    elif "tmall" in train_path:
        print("Loading full sequences TMall")
        # full_sequences = read_full_sequences("../data/tmall/data/sequences_full.csv")
        full_sequences = []
    else:
        print("Loading full sequences ML1M")
        full_sequences = read_full_sequences("../data/movielens/data/ml1m_sequences_full.csv")

    train_dataset, i_num, u_num = prepare_dataset(train_path,
                                                  properties_path=properties_path,
                                                  full_sequences=full_sequences,
                                                  sampling=sampling)
    val_dataset, val_i_num, val_u_num = prepare_dataset(val_path,
                                                        properties_path=properties_path,
                                                        full_sequences=full_sequences,
                                                        sampling=sampling)
    test_dataset, test_i_num, test_u_num = prepare_dataset(test_path,
                                                           properties_path=properties_path,
                                                           full_sequences=full_sequences,
                                                           sampling=sampling)

    if (val_i_num > i_num) or (test_i_num > i_num):
        i_num = max([i_num, val_i_num, test_i_num])
        u_num = max([u_num, val_u_num, test_u_num])
        print("Overriding item_num, context_num, user_num")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader, i_num, u_num
