
import pickle
import codecs
import torch
from torch.utils.data import Dataset, IterableDataset

class MSMarcoDataset(Dataset):
    def __init__(self, cached_features_file, subset):
        self._subset = subset
        self._is_training = (subset == 'train')
        self._features = torch.load(cached_features_file)
        self._size = len(self._features)

    def get_features(self):
        return self._features

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        feature = self._features[index]

        index = torch.tensor(index, dtype=torch.long)
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        outputs = (input_ids, input_mask, segment_ids, index)

        if self._is_training:
            is_selected = torch.tensor(feature.is_selected, dtype=torch.float)
            start_position = torch.tensor(feature.start_position if feature.start_position else 0, dtype=torch.long)
            end_posittion = torch.tensor(feature.end_position if feature.end_position else 0, dtype=torch.long)
            is_impossible = torch.tensor(feature.is_impossible, dtype=torch.long)
            outputs = (input_ids, input_mask, segment_ids, index, is_selected, start_position, end_posittion, is_impossible)

        return outputs


class MSMarcoIterDataset(IterableDataset):
    def __init__(self, cached_features_file, subset):
        self._subset = subset
        self._file = cached_features_file
        self._is_training = (subset == 'train')

    def __iter__(self):
        file_itr = open(self._file)
        mapped_itr = map(self.line_mapper, file_itr)

        return mapped_itr

    def line_mapper(self, line):
        s = line.strip()
        feature = pickle.loads(codecs.decode(s.encode(), "base64"))

        index = torch.tensor(index, dtype=torch.long)
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        input_mask = torch.tensor(feature.input_mask, dtype=torch.long)
        segment_ids = torch.tensor(feature.segment_ids, dtype=torch.long)
        outputs = (input_ids, input_mask, segment_ids, index)

        if self._is_training:
            is_selected = torch.tensor(feature.is_selected, dtype=torch.float)
            start_position = torch.tensor(feature.start_position, dtype=torch.long)
            end_posittion = torch.tensor(feature.end_position, dtype=torch.long)
            is_impossible = torch.tensor(feature.is_impossible, dtype=torch.long)
            outputs = (input_ids, input_mask, segment_ids, index, is_selected, start_position, end_posittion, is_impossible)

        return outputs

