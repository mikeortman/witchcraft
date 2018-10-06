from typing import Callable, List
import tensorflow as tf


class WitchcraftDataset:
    def __init__(self, tf_dataset_base: tf.data.Dataset) -> None:
        self._current_dataset: tf.data.Dataset = tf_dataset_base

    def flat_map(self, mapper: Callable[[any], List[any]]) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.flat_map(mapper)
        return self

    def map(self, mapper: Callable[[any], any]) -> 'WitchcraftDataset':
        self._current_dataset = self._current_dataset.map(mapper)
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

    def to_tf_iterator(self) -> tf.data.Iterator:
        return self._current_dataset.make_one_shot_iterator()

    @classmethod
    def from_files(cls, file_pattern: str) -> 'WitchcraftDataset':
        return WitchcraftDatasetFiles(file_pattern)


class WitchcraftDatasetFiles(WitchcraftDataset):
    def __init__(self, file_pattern: str, shuffle: bool = False) -> None:
        super(WitchcraftDatasetFiles, self).__init__(tf.data.Dataset.list_files(file_pattern, shuffle=shuffle))
