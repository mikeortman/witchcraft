from typing import List, Tuple


def skipgramify(list: List[any], window_size: int) -> List[Tuple[any, any, int]]:
    if window_size < 0:
        window_size = 0

    skipgrams = []
    list_size: int = len(list)
    for i in range(list_size):
        current_item = list[i]

        if current_item is None:
            continue

        skipgrams += [(current_item, current_item, 0)]

        for y in range(1, window_size + 1):
            if i - y >= 0 and list[i - y] is not None:
                skipgrams += [(current_item, list[i - y], -y)]

        for y in range(1, window_size + 1):
            if i + y < list_size and list[i + y] is not None:
                skipgrams += [(current_item, list[i + y], y)]

    return skipgrams