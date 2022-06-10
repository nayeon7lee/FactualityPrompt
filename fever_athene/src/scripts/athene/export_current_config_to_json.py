import argparse

from athene.utils.config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='/path/to/file/to/save/config')
    args = parser.parse_args()
    Config.save_config(args.output)
