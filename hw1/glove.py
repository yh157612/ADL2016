import argparse
import tf_glove

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('corpus_path')
arg_parser.add_argument('output_path')
args = arg_parser.parse_args()

corpus = open(args.corpus_path).read().split()

model = tf_glove.GloVeModel(embedding_size=100,
                            context_size=10,
                            min_occurrences=5)
model.fit_to_corpus(corpus)
# model.train(50, log_dir=args.output_path, save_embed_interval=5)
model.train_concurrent(100)
model.save_embeddings(args.output_path)
