from typing import Callable, List
import tensorflow as tf


class WitchcraftDataset:
    def __init__(self, tf_dataset_base) -> None:
        self._current_dataset = tf_dataset_base

    def flat_map(self, mapper: Callable[[any], List[any]]) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.flat_map(mapper)
        return self

    def map(self, mapper: Callable[[any], any], num_parallel_calls: int = 1) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.map(mapper, num_parallel_calls=num_parallel_calls)
        return self

    def batch(self, batch_size: int) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.batch(batch_size)
        return self

    def shuffle(self, shuffle_buffer: int) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.shuffle(shuffle_buffer)
        return self

    def repeat(self) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.repeat()
        return self

    def prefetch(self, buffer_size: int) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.prefetch(buffer_size)
        return self

    def cache(self) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.cache()
        return self

    def to_tf_iterator(self) -> tf.data.Iterator:
        return self._current_dataset.make_one_shot_iterator()

    @classmethod
    def from_files(cls, file_pattern: str) -> 'WitchcraftDataset':
        return WitchcraftDatasetFiles(file_pattern)
