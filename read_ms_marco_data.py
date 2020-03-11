
import logging
import json
import numpy as np
import torch
import os

from tqdm import tqdm, trange

from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class MSMARCOExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qid,
                 query,
                 passages,
                 answers,
                 is_selected):
        self.qid = qid
        self.query = query
        self.passages = passages
        self.answers = answers
        self.is_selected = is_selected


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 qid,
                 tokens,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_selected=None,
                 answers=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qid = qid
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.answers = answers
        self.is_selected = is_selected
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


def read_ms_marco_examples(input_file, subset):
    data = json.load(open(input_file, 'r'))
    num_questions = len(data['query'])

    examples = []
    for i in range(num_questions):
        qid = str(i)
        query = data['query'][qid]
        passages = []
        answers = data['answers'][qid]
        is_selected = []
        for passage in data['passages'][qid]:
            passages.append(passage['passage_text'])
            if subset == 'train':
                is_selected.append(passage['is_selected'])
            else:
                is_selected.append(0)
        qid = subset + '_' + qid
        example = MSMARCOExample(qid, query, passages, answers, is_selected)
        examples.append(example)

    return examples


def fmeasure(precision, recall):
  """Computes f-measure given precision and recall values."""

  if precision + recall > 0:
    return 2 * precision * recall / (precision + recall)
  else:
    return 0.0


def _lcs_table(ref, can):
    """Create 2-d LCS score table."""
    rows = len(ref)
    cols = len(can)
    lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == can[j - 1]:
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1
            else:
                lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
    return lcs_table


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 max_query_length, subset, cached_features_file,
                                 num_passage_per_query=10,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 sequence_a_is_doc=False):
    if subset == 'train':
        is_training = True
    elif subset == 'dev':
        is_training = False
    elif subset == 'eval':
        is_training = False
    else:
        raise(ValueError("Unrecognized Subset: %s" % subset))

    features = []
    # for (example_index, example) in enumerate(tqdm(examples)):
    for (example_index, example) in enumerate(examples):
        if example_index % 100 == 0:
            print("%d / %d Finished" % (example_index, len(examples)))
        query_tokens = tokenizer.tokenize(example.query)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        answers = example.answers
        if is_training:
            if answers[0] == 'No Answer Present.':
                is_impossible = True
                answer_tokens = []
            else:
                answer_tokens = tokenizer.tokenize(answers[0])
                if len(answer_tokens) == 0:
                    is_impossible = True
                else:
                    is_impossible = False
        else:
            is_impossible = None

        tokens_list = []
        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        is_selected_list = []
        max_score = 0.0
        start_position = None
        end_position = None
        for pid in range(num_passage_per_query):
            passage = example.passages[pid] if pid < len(example.passages) else ''
            is_selected = example.is_selected[pid] if pid < len(example.passages) else 0
            is_selected_list.append(is_selected)

            tokens = []
            segment_ids = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

            # XLNet: P SEP Q SEP CLS
            # Others: CLS Q SEP P SEP
            if not sequence_a_is_doc:
                # Query
                tokens += query_tokens
                segment_ids += [sequence_a_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

            passage_tokens = tokenizer.tokenize(passage)
            passage_tokens = passage_tokens[:max_tokens_for_doc]

            # Paragraph
            for i in range(len(passage_tokens)):
                tokens.append(passage_tokens[i])
                if not sequence_a_is_doc:
                    segment_ids.append(sequence_b_segment_id)
                else:
                    segment_ids.append(sequence_a_segment_id)
                p_mask.append(0)

            if sequence_a_is_doc:
                # SEP token
                tokens.append(sep_token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                tokens += query_tokens
                segment_ids += [sequence_b_segment_id] * len(query_tokens)
                p_mask += [1] * len(query_tokens)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            tokens_list.append(tokens)
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)

            if is_training:
                if is_impossible:
                    start_position = 0
                    end_position = 0
                else:
                    # get golden_span
                    for i in range(len(passage_tokens)):
                        lcs_table = _lcs_table(answer_tokens, passage_tokens[i:])
                        for j in range(i, len(passage_tokens)):
                            l = j - i + 1
                            lcs_length = lcs_table[-1][l]
                            precision = lcs_length / l
                            recall = lcs_length / len(answer_tokens)
                            rouge_l = fmeasure(precision, recall)
                            if rouge_l > max_score:
                                max_score = rouge_l
                                start_position = pid * max_seq_length + len(query_tokens) + 2 + i
                                end_position = pid * max_seq_length + len(query_tokens) + 2 + j

        features.append(
            InputFeatures(
                qid=example.qid,
                tokens=tokens_list,
                input_ids=input_ids_list,
                input_mask=input_mask_list,
                segment_ids=segment_ids_list,
                answers=answers,
                is_selected=is_selected_list,
                start_position=start_position,
                end_position=end_position,
                is_impossible=is_impossible))
    
    torch.save(features, cached_features_file)


if __name__ == "__main__":
    input_file = '/data3/private/liujiahua/dataset/msmarco/dev_v2.1.json'
    subset = 'dev'
    cached_features_file = 'dev.dat'

    model_name_or_path = 'bert-base-uncased'
    do_lower_case = True
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path,
                                                do_lower_case=do_lower_case,
                                                cache_dir=None)
    max_seq_length = 384
    max_query_length = 64


    logger.info("Loading features from cached file %s", cached_features_file)
    examples = read_ms_marco_examples(input_file=input_file, subset=subset)
    logger.info("Creating features from dataset file at %s", input_file)
    """
    features = convert_examples_to_features(examples=examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=max_seq_length,
                                            max_query_length=max_query_length,
                                            subset=subset,
                                            cached_features_file=cached_features_file)
    """
    num_shard = 5 
    import multiprocessing as mp
    pool = mp.Pool(num_shard)

    num_per_shard = (len(examples) - 1) // num_shard + 1
    for i in range(num_shard):
        start = i * num_per_shard
        end = (i + 1) * num_per_shard
        pool.apply_async(convert_examples_to_features, (examples[start:end], tokenizer, max_seq_length, max_query_length, subset, cached_features_file + '.' + str(i)))

    pool.close()
    pool.join()

    """
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_ms_marco_examples(input_file=input_file, subset=subset)
        import time
        dt = time.time()
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=max_seq_length,
                                                max_query_length=max_query_length,
                                                subset=subset)
        print(time.time() - dt)
        torch.save(features, cached_features_file)
    """

