import random
from typing import Set
from typing import List
from typing import Tuple


def split_train_test(items: Set[str], ratio: float = 0.8) -> Tuple[Set[str], Set[str]]:
    """Helper function to split a set of item according to a ratio

    Args:
        items (Set[str]): set of item to split
        ratio (float, optional): ratio to apply for splitting. Defaults to 0.8.

    Returns:
        Tuple[Set[str], Set[str]]: splitted sets
    """
    length = len(items)
    items_to_select = int(length * ratio)
    train_samples = set(random.sample(items, items_to_select))
    test_samples = set([item for item in items if item not in train_samples])
    return train_samples, test_samples
