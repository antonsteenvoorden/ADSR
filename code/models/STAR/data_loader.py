import csv
import torch
from torch.utils.data import Dataset, DataLoader

device = 'cpu'


# all target attributes, when calculating metrics one is selected
class CustomDataset(Dataset):
    item_x = None
    item_y = None
    context_x = None

    users = None
    length = None

    def __init__(self, item_x, item_y, users, context_x):
        self.item_x = item_x
        self.item_y = item_y

        self.context_x = context_x

        self.users = users
        self.length = len(item_y)

    def __getitem__(self, index):
        user_id = self.users[index]

        item_x = self.item_x[index]
        item_y = self.item_y[index]

        # As we possibly have multiple contexts we wrap it in a tuple
        context_x = self.context_x[index]

        x = (item_x, context_x, user_id)
        y = (item_y, user_id)
        return x, y

    def __len__(self):
        return self.length


def reorder_zeroes(row, context_row):
    n = torch.nonzero(row).view(-1)
    p = len(n)
    new = torch.zeros_like(row)
    new[-p:] = row[n]
    new_context = torch.zeros_like(row)
    new_context[-p:] = context_row[n]
    return new, new_context


def read(path, properties_path=None):
    clean = lambda l: [int(x) for x in l.strip('[]').split(',')]

    item_num = 0

    with open(path, 'r') as file:
        csv_reader = csv.reader(file, delimiter='|')
        header = next(csv_reader, None)  # skip the header

        context_x = []
        item_x, item_y, users = [], [], []

        # Note the pops for the single element lists
        for row in csv_reader:
            (seq, target, user) = row[:3]
            ctxt_x = row[-1]
            seq = clean(seq)
            target = clean(target).pop()
            user = clean(user).pop()
            ctxt_x = clean(ctxt_x)

            if max(seq) > item_num:
                item_num = max(seq)

            ctxt_x = torch.tensor(ctxt_x, device=device)
            seq = torch.tensor(seq, device=device)
            #REPLACE THE ZEROES AT THE END TO THE BEGINNING
            seq, ctxt_x = reorder_zeroes(seq, ctxt_x)

            item_x.append(seq)
            item_y.append(target)
            users.append(user)

            context_x.append(ctxt_x)

    # as everything is wrapped in lists we need to call max multiple times to extract the actual number
    # we also add 1 to account for the 0 indexing
    item_num = max(item_num, (max(item_y))) + 1
    context_num = 31
    user_num = max(users) + 1

    # Convert to tensors (needed for the dataset to get items correctly)
    item_x = torch.stack(item_x, dim=0)
    context_x = torch.stack(context_x, dim=0)
    item_y = torch.tensor(item_y, dtype=torch.long)
    users = torch.tensor(users, dtype=torch.long)
    return item_x, item_y, users, context_x, item_num, context_num, user_num


def prepare_dataset(path, properties_path=None):
    # context_x and context_y, context_num are tuples, as we might have multiple contexts
    item_x, item_y, users, context_x, i_num, c_num, u_num = read(path, properties_path=properties_path)
    dataset = CustomDataset(item_x, item_y, users, context_x)
    return dataset, i_num, c_num, u_num


def get_data_loaders(train_path, val_path, test_path, batch_size, num_workers=4, shuffle=True, properties_path=None):
    train_dataset, i_num, c_num, u_num = prepare_dataset(train_path, properties_path=properties_path)
    val_dataset, val_i_num, val_c_num, val_u_num = prepare_dataset(val_path, properties_path=properties_path)
    test_dataset, test_i_num, test_c_num, test_u_num = prepare_dataset(test_path, properties_path=properties_path)

    if (val_i_num > i_num) or (val_c_num > c_num) or (val_u_num) > (u_num):
        raise Exception(f"Validation set has a higher number of unique input ids {val_i_num} > {i_num}, "
                        f"{val_c_num} > {c_num}, {val_u_num} > {u_num}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_dataloader, eval_dataloader, test_dataloader, i_num, c_num, u_num
