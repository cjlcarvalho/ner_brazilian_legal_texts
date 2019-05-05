import os
from typing import Union

from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

from utils import adapt_lener_to_flair, LENER_DATASET_DIR

# Increase data percentage to train with a bigger amount of files
adapt_lener_to_flair(percentage=0.02)


class EmbeddingModel:
    def __init__(self, working_directory: Union[bytes, str], use_gpu=False):

        self._working_directory = working_directory
        self._columns = {0: "w", 1: "chunk"}
        self._flair_data_folder = os.path.join(LENER_DATASET_DIR, "flair")

        self._model = None
        self._corpus = None
        self._build_model(use_gpu)

    def train(self):

        trainer = ModelTrainer(self._model, self._corpus)

        trainer.train(
            self._working_directory,
            learning_rate=0.1,
            mini_batch_size=32,
            anneal_factor=0.5,
            patience=5,
            max_epochs=150,
        )

    def _build_model(self, use_gpu=False):

        self._corpus = NLPTaskDataFetcher.load_column_corpus(
            self._flair_data_folder,
            self._columns,
            train_file="train.txt",
            test_file="test.txt",
            dev_file="dev.txt",
        )

        tag_dict = self._corpus.make_tag_dictionary("ner")

        embedding_types = [
            WordEmbeddings("glove"),
            FlairEmbeddings("portuguese-forward"),
            FlairEmbeddings("portuguese-backward"),
        ]

        embeddings = StackedEmbeddings(embeddings=embedding_types)

        self._model = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dict,
            tag_type="ner",
            # use_crf=True,
            use_rnn=True,
        )

        if use_gpu:
            self._model.cuda(0)


if __name__ == "__main__":

    model = EmbeddingModel(
        os.path.join(os.path.dirname(__file__), "output"), use_gpu=True
    )
    model.train()
