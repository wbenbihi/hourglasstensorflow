import random
from typing import Set
from typing import List
from typing import Tuple


def split_train_test(items: Set[str], ratio: float = 0.8) -> Tuple[Set[str], Set[str]]:

    length = len(items)
    items_to_select = int(length * ratio)
    train_samples = set(random.sample(items, items_to_select))
    test_samples = set([item for item in items if item not in train_samples])
    return train_samples, test_samples
