import argparse
import json
import random

from tqdm import tqdm

from common.dataset.reader import JSONLineReader
from common.util.log_helper import LogHelper
from drqa.retriever.utils import normalize


def predicted_evidence_to_list(pred_evidences):
    evidences = []
    for e in pred_evidences:
        evidences.append(normalize(str(e[-2])) + '§§§' + normalize(str(e[-1])))
    return evidences


def gold_evidence_to_list(gold_evidences):
    evidences = []
    for e_set in gold_evidences:
        evidence_set = []
        for e in e_set:
            evidence_set.append(normalize(str(e[-2])) + '§§§' + normalize(str(e[-1])))
        evidences.append(evidence_set)
    return evidences


def is_gold_evidence_predicted(_line):
    _all_predicted_evidences = predicted_evidence_to_list(_line['predicted_evidence'])
    _all_gold_evidences = gold_evidence_to_list(_line['evidence'])
    return any(all(e in _all_predicted_evidences for e in e_set) for e_set in _all_gold_evidences)


def random_fill_gold_evidence(_line):
    _all_gold_evidences = gold_evidence_to_list(_line['evidence'])
    _all_predicted_evidences = predicted_evidence_to_list(_line['predicted_evidence'])
    e_set = random.sample(_all_gold_evidences, 1)[0]
    logger.debug("fill with evidence set: " + str(e_set))
    for e in e_set:
        e_segments = e.split('§§§')
        if e not in _all_predicted_evidences:
            _line['predicted_evidence'] = [[e_segments[0], int(e_segments[1])]] + _line['predicted_evidence']
    _line['predicted_evidence'] = _line['predicted_evidence'][:args.max_evidence]
    return _line


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='/path/to/input/file')
    parser.add_argument('output', help='/path/to/output/file')
    parser.add_argument('--max_evidence', help='max num of evidences', type=int, default=5)
    args = parser.parse_args()
    LogHelper.setup()
    logger = LogHelper.get_logger("replace_noise_dataset")
    random.seed(55)
    jlr = JSONLineReader()
    lines = jlr.read(args.input)
    counter = 0
    with open(args.output, 'w') as f:
        for i, line in tqdm(enumerate(lines)):
            if not line['label'] == 'NOT ENOUGH INFO' and not is_gold_evidence_predicted(line):
                counter += 1
                logger.info("line " + str(i + 1) + " should be filled")
                line = random_fill_gold_evidence(line)
            f.write(json.dumps(line) + '\n')
    logger.info(str(counter) + " samples filled with gold evidence")
