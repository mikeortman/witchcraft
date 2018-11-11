from typing import Optional

def hash_fnv1a(input_str: str, bucket_size: Optional[int] = None) -> int:
    # https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
    output_hash: int = 2166136261
    for c in input_str:
        output_hash = (output_hash ^ ord(c)) * 16777619

    if bucket_size is not None:
        output_hash = output_hash % bucket_size

    return output_hash