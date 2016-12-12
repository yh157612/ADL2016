import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path')
    parser.add_argument('en_path')
    parser.add_argument('es_path')
    args = parser.parse_args()

    with open(args.json_path) as f:
        data_obj = json.load(f)

    os.makedirs(os.path.dirname(os.path.abspath(args.en_path)),
                mode=0o755, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.es_path)),
                mode=0o755, exist_ok=True)

    with open(args.en_path, 'w') as en, open(args.es_path, 'w') as es:
        for data in data_obj:
            en.write('{0}\n{0}\n'.format(data[0]))
            es.write('{}\n{}\n'.format(data[1], data[2]))


if __name__ == '__main__':
    main()
