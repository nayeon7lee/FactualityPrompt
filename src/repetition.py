'''
    This code is adapted from https://github.com/ari-holtzman/degen/blob/master/metrics/repetition.py by Ari Holtzman.
'''
import argparse
import json
import os

from transformers import GPT2Tokenizer

from src.const import DATA_DIR, HOME_DIR, GEN_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    parser.add_argument("--numbers-only", action="store_true")
    parser.add_argument("--output", action="store_true")
    parser.add_argument("--final", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large", do_lower_case=True)
    SEP = tokenizer.encode(tokenizer.bos_token)[0]

    objs = []
    max_n = 90

    args.file = "{}/{}".format(GEN_DIR, args.file)
    with open(args.file, 'r') as fin:
        for l in fin:
            objs.append(json.loads(l.strip()))

    n_repeated_examples = 0
    repeated_times_sum = 0

    nn = len(objs)
    for idx, obj in enumerate(objs):
        gen = obj['text']

        if "WikiNamePrefix" in args.file:
            wikiPrefix = obj['prompt'].split(". ")[-1].strip()
            gen = gen.replace(wikiPrefix, " ")

        if gen[-1] == SEP:
            gen.pop(-1)
        rev_gen = list(reversed(gen))
        last_n_repeats = [0] * max_n

        for n in range(1, max_n + 1):
            n_repeat = 1
            while len(rev_gen[n*n_repeat:n*(n_repeat+1)]) == n and \
                    rev_gen[n*n_repeat:n*(n_repeat+1)] == rev_gen[:n]:
                n_repeat += 1
            last_n_repeats[n - 1] = n_repeat
        max_repeated_n = max(range(max_n), key=lambda x: last_n_repeats[x])

        if last_n_repeats[max_repeated_n] > 1 and (max_repeated_n+1 >= 3 or last_n_repeats[max_repeated_n] > 50):
            obj['repetition'] = {
                'repeated_phrase': list(reversed(rev_gen[:max_repeated_n + 1])),
                'repeated_times': last_n_repeats[max_repeated_n],
                'repeated_phrase_length': max_repeated_n + 1,
            }
            n_repeated_examples += 1

            repeated_times_sum += last_n_repeats[max_repeated_n]

        else:
            obj['repetition'] = None

    # if not args.numbers_only:
    #     print("filename\tnumber of repeating examples")
    # print(f"{os.path.basename(args.file)},{n_repeated_examples},{repeated_times_sum/nn}")
    print(f"{n_repeated_examples},{repeated_times_sum/nn}")


    if args.output:
        output_filename = os.path.join(os.path.dirname(args.file), "repetition_" + os.path.basename(args.file))
        with open(output_filename, 'w+') as fout:
            for obj in objs:
                print(json.dumps(obj), file=fout)

    if args.final:
        gen_path = args.file
        res_path = gen_path.replace(".jsonl", "_results.jsonl")
        with open(res_path, 'a') as outfile:
            res_obj = {
                "repetition": n_repeated_examples,
            }
            json.dump(res_obj, outfile)
            outfile.write("\n")


if __name__ == '__main__':
    main()