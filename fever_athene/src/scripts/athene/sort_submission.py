import argparse
import json
import os

from common.dataset.reader import JSONLineReader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submission', help='/path/to/submission/file', required=True)
    parser.add_argument('--data', help='/path/to/data/file', required=True)
    parser.add_argument('--output', help='/path/to/output/file', required=True)
    args = parser.parse_args()
    jlr = JSONLineReader()
    submission_lines = jlr.read(args.submission)
    data_lines = jlr.read(args.data)
    assert len(submission_lines) == len(data_lines), "lengths of submission and data set are different!"
    submission_dict = {}
    for line in submission_lines:
        submission_dict[line['id']] = line
    assert len(submission_dict) == len(submission_lines), "lines in submission are not unique!"
    sorted_lines = []
    for d in data_lines:
        sorted_lines.append(submission_dict[d['id']])
    assert len(sorted_lines) == len(data_lines), "some claims from data set are missing in submission!"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        for l in sorted_lines:
            f.write(json.dumps(l) + '\n')
