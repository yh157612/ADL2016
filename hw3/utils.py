import os
import random
from collections import Counter


def prepare_train_data(file_path, max_vocab_size):
    all_sentence = []
    all_tags = []
    all_intent = []
    with open(file_path) as f:
        for line in f:
            sentence, tags = line.split('\t')
            sentence = [('<num>' if word.isdigit() else word) for word in sentence.split()[1:-1]]
            tags = tags.split()
            intent = tags[-1]
            tags = tags[1:-1]
            all_sentence.append(sentence)
            all_tags.append(tags)
            all_intent.append(intent)

    counter = Counter()
    vocab = ['<unk>']
    for sentence in all_sentence:
        counter.update(sentence)
    vocab.extend([word for word, _ in counter.most_common(max_vocab_size - 1)])
    dictionary = {word: index for index, word in enumerate(vocab)}
    all_sentence = [[(dictionary[word] if word in dictionary else 0) for word in sentence]
                    for sentence in all_sentence]

    tags_list = []
    tags_dict = {}
    for tags in all_tags:
        for tag in tags:
            if tag not in tags_dict:
                tags_list.append(tag)
                tags_dict[tag] = len(tags_dict)
    all_tags = [[tags_dict[tag] for tag in tags] for tags in all_tags]
    intent_list = []
    intent_dict = {}
    for intent in all_intent:
        if intent not in intent_dict:
            intent_list.append(intent)
            intent_dict[intent] = len(intent_dict)
    all_intent = [intent_dict[intent] for intent in all_intent]

    return all_sentence, all_tags, all_intent, vocab, dictionary, tags_list, tags_dict, intent_list, intent_dict


def split_data(all_sentence, all_tags, all_intent, dev_ratio=0.1):
    data = list(zip(all_sentence, all_tags, all_intent))
    random.shuffle(data)
    dev_data = data[:int(len(data) * dev_ratio)]
    train_data = data[int(len(data) * dev_ratio):]
    return tuple(zip(*train_data)), tuple(zip(*dev_data))


def prepare_test_data(file_path, dictionary):
    all_sentence = []
    with open(file_path) as f:
        for line in f:
            sentence = [('<num>' if word.isdigit() else word) for word in line.split()[1:-1]]
            all_sentence.append(sentence)
    all_sentence = [[(dictionary[word] if word in dictionary else 0) for word in sentence]
                    for sentence in all_sentence]
    return all_sentence


def save_vocabulary(file_path, vocab):
    file_path = os.path.abspath(file_path)
    os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)
    with open(file_path, mode='w') as f:
        f.write('\n'.join(vocab))


def load_vocabulary(file_path):
    with open(file_path) as f:
        vocab = f.read().split()
    dictionary = {word: index for index, word in enumerate(vocab)}
    return vocab, dictionary


def main():
    prepare_train_data('data/atis.train.w-intent.iob', max_vocab_size=10000)


if __name__ == '__main__':
    main()
