import os
import random

from seqeval.metrics import accuracy_score, classification_report

from typing import Optional, Type, Union

from pyseqlab.attributes_extraction import GenericAttributeExtractor
from pyseqlab.features_extraction import FeatureExtractor, FOFeatureExtractor
from pyseqlab.linear_chain_crf import LCRF, LCRFModelRepresentation
from pyseqlab.utilities import TemplateGenerator
from pyseqlab.workflow import GenericTrainingWorkflow

from utils import LENER_DATASET_DIR, PySeqLabSequenceBuilder, adapt_lener_to_pyseqlab


adapt_lener_to_pyseqlab()


class CRFModel:
    def __init__(
        self,
        model_type: Type[LCRF],
        model_representation_type: Type[LCRFModelRepresentation],
        feature_extraction_type: [Type[FeatureExtractor], Type[FOFeatureExtractor]],
        working_directory: Union[bytes, str],
    ):

        self._pyseqlab_dataset_dir = os.path.join(LENER_DATASET_DIR, "pyseqlab")
        self._model_type = model_type
        self._model_representation_type = model_representation_type
        self._feature_extraction_type = feature_extraction_type
        self._working_directory = working_directory

        self._data_parser_options = dict(
            header="main", y_ref=True, column_sep=" ", seg_other_symbol="O"
        )

        self._attribute_description = dict(
            w=dict(description="word observation track", encoding="categorical")
        )

        self._generic_attribute_extractor = GenericAttributeExtractor(
            self._attribute_description
        )

        self._data_split_option = dict(method="cross_validation", k_fold=5)

        self._training_workflow: Optional[GenericTrainingWorkflow] = None
        self._model_object: Optional[LCRF] = None
        self._trained_model_dir = None
        self._training_data_split = None

    def train(self, **kwargs):

        optimization_method = kwargs.get("optimization_method")
        if optimization_method not in ("SGA-ADADELTA", "SGA", "SVRG"):
            raise Exception("Unknown optimization method.")

        regularization_type = kwargs.get("regularization_type")
        regularization_value = kwargs.get("regularization_value")
        epochs = kwargs.get("epochs")

        optimization_option = dict(
            method=optimization_method,
            regularization_type=regularization_type,
            regularization_value=regularization_value,
            num_epochs=epochs,
        )

        self._build_model()

        # TODO: train using data splits
        if self._training_data_split is not None:
            for fold in self._training_data_split:
                train_sequences_id = self._training_data_split[fold]["train"]
                self._trained_model_dir = self._training_workflow.train_model(
                    train_sequences_id, self._model_object, optimization_option
                )

            print("-" * 100)
            print("Model trained successfully.")

    def predict(self, sequences, output_file):

        decoding_method = "viterbi"

        return self._model_object.decode_seqs(
            decoding_method=decoding_method,
            out_dir=self._trained_model_dir,
            seqs=sequences,
            file_name=output_file,
            sep="\t",
        )

    def evaluate(self, sequence_file):

        sequences = PySeqLabSequenceBuilder(
            self._pyseqlab_dataset_dir
        ).build_sequences_from_file(sequence_file)

        y_true = [sequence.flat_y for sequence in sequences]

        prediction = self.predict(
            sequences,
            os.path.join(os.path.dirname(sequence_file), "output_evaluation.txt"),
        )

        y_pred = [value["Y_pred"] for key, value in prediction.items()]

        print(y_true)
        print(y_pred)
        print(accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))

    def _build_model(self):

        assert self._model_object is None

        if self._training_workflow is None:
            self._build_training_workflow()

        training_sequence = PySeqLabSequenceBuilder(
            self._pyseqlab_dataset_dir
        ).generate_sequences("train")

        self._training_data_split = self._training_workflow.seq_parsing_workflow(
            self._data_split_option,
            seqs=random.sample(training_sequence, 1000),
            full_parsing=True,
        )

        self._model_object = self._training_workflow.build_crf_model(
            self._training_data_split[0]["train"], "f_0"
        )
        self._model_object.weights.fill(0)

    def get_model_features(self):

        return dict(
            number=len(self._model_object.model.modelfeatures_codebook),
            features=self._model_object.model.modelfeatures,
        )

    def _build_training_workflow(self):

        assert self._training_workflow is None

        template_xy = {}

        template_gen = TemplateGenerator()
        template_gen.generate_template_XY(
            "w", ("1-gram:2-grams", range(-1, 2)), "1-state", template_xy
        )

        template_y = template_gen.generate_template_Y("1-state:2-states")

        if self._feature_extraction_type == FOFeatureExtractor:
            feature_extractor = self._feature_extraction_type(
                template_xy, template_y, self._attribute_description, start_state=False
            )
        else:
            feature_extractor = self._feature_extraction_type(
                template_xy, template_y, self._attribute_description
            )

        self._training_workflow = GenericTrainingWorkflow(
            self._generic_attribute_extractor,
            feature_extractor,
            None,
            self._model_representation_type,
            self._model_type,
            self._working_directory,
        )
