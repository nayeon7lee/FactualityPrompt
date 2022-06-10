import argparse
import json
from tqdm import tqdm
from common.dataset.reader import JSONLineReader
from common.util.log_helper import LogHelper


def _sent_to_str(sent):
    return sent[-2] + "$$$" + str(sent[-1])


def _replace_sent_with_str(sent, string):
    segments = string.split(r"$$$")
    if len(segments) != 2:
        raise Exception("Illegal string: " + string)
    sent[-2] = segments[0]
    sent[-1] = int(segments[1])
    return sent


def _build_new_sent_with_str(string, num_of_segments=2):
    if num_of_segments == 2:
        sent = ["", -1]
    elif num_of_segments == 4:
        sent = [-1, -1, "", -1]
    else:
        raise Exception("Illegal num_of_segments: " + str(num_of_segments))
    return _replace_sent_with_str(sent, string)


def _sents_from_evidences(evidences):
    sents = set()
    for evidence in evidences:
        for s in evidence:
            sent = _sent_to_str(s)
            sents.add(sent)
    return sents


def _fill_pred_sents_with_gold(pred_sents, gold_sents, max_sent):
    selected_sents = pred_sents[:max_sent]
    neg_indices = []
    for i, selected in enumerate(selected_sents):
        key_selected = _sent_to_str(selected)
        if key_selected in gold_sents:
            gold_sents.remove(key_selected)
        else:
            neg_indices.append(i)
    if len(gold_sents) == 0:
        return selected_sents
    if len(selected_sents) <= max_sent:
        for _ in range(max_sent - len(selected_sents)):
            selected_sents.append(_build_new_sent_with_str(gold_sents.pop()))
            if len(gold_sents) == 0:
                return selected_sents
    if len(neg_indices) > 0:
        neg_indices = reversed(neg_indices)
        for i in neg_indices:
            sent = selected_sents[i]
            selected_sents[i] = _replace_sent_with_str(sent, gold_sents.pop())
            if len(gold_sents) == 0:
                return selected_sents
    if len(gold_sents) > 0:
        logger.warn(str(len(gold_sents)) +
                    " gold sentences cannot be filled into prediction")
    return selected_sents


if __name__ == '__main__':
    LogHelper.setup()
    logger = LogHelper.get_logger('fill_gold_sentences')
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input', help='/path/to/input/file', required=True)
    parser.add_argument(
        '--output', help='/path/to/output/file', required=True)
    parser.add_argument(
        '--max-sent', type=int, help='Maximal number of sentences per claim', default=10)
    args = parser.parse_args()
    jlr = JSONLineReader()
    data = jlr.read(args.input)
    with open(args.output, "w+") as output_file:
        for data in tqdm(data):
            if data['verifiable'] != 'NOT VERIFIABLE':
                pred_sents = data['predicted_sentences']
                gold_evidences = data['evidence']
                gold_sents = _sents_from_evidences(gold_evidences)
                filled_pred_sents = _fill_pred_sents_with_gold(
                    pred_sents, gold_sents, args.max_sent)
                data['predicted_sentences'] = filled_pred_sents
            output_file.write(json.dumps(data) + "\n")
