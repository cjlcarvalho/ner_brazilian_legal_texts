from typing import Type

from kashgari.tasks.seq_labeling import BLSTMCRFModel, SequenceLabelingModel

from utils import adapt_lener_to_kashgari, LeNerCorpus


# Increase data percentage to train with a bigger amount of files
adapt_lener_to_kashgari()


class EmbeddingModel:
    def __init__(self, model_type: Type[SequenceLabelingModel]):

        self._model = model_type()

    def train(self, **kwargs):

        epochs = kwargs.get("epochs")

        x_train, y_train = LeNerCorpus.get_sequence_tagging_data()

        self._model.fit(x_train, y_train, epochs=epochs)

    def evaluate(self, data_type: str = "test"):

        if data_type not in ("test", "dev"):
            raise Exception("Wrong data type for evaluation")

        x, y = LeNerCorpus.get_sequence_tagging_data(data_type=data_type)

        self._model.evaluate(x, y, debug_info=True)
