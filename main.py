import argparse
import os

from pyseqlab.features_extraction import FOFeatureExtractor, HOFeatureExtractor
from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.ho_crf import HOCRFAD, HOCRFADModelRepresentation
from pyseqlab.hosemi_crf_ad import HOSemiCRFAD, HOSemiCRFADModelRepresentation

from lcrf import LCRFModel
from utils import LENER_DATASET_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=("FirstOrderCRF", "HOCRFAD", "HOSemiCRFAD"),
    )
    parser.add_argument("--method", type=str, required=True, choices=("CRF",))

    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), "output")

    if args.method == "CRF":
        if args.model == "FirstOrderCRF":

            model = LCRFModel(
                FirstOrderCRF,
                FirstOrderCRFModelRepresentation,
                FOFeatureExtractor,
                output_path,
            )

        elif args.model == "HOCRFAD":

            model = LCRFModel(
                HOCRFAD, HOCRFADModelRepresentation, HOFeatureExtractor, output_path
            )

        elif args.model == "HOSemiCRFAD":

            model = LCRFModel(
                HOSemiCRFAD,
                HOSemiCRFADModelRepresentation,
                HOFeatureExtractor,
                output_path,
            )

        else:

            raise Exception("Model unknown")

        model.train(
            optimization_method="SGA-ADADELTA",
            regularization_type="l2",
            regularization_value=0,
            epochs=10,
        )
        pyseqlab_dataset_dir = os.path.join(LENER_DATASET_DIR, "pyseqlab")
        model.evaluate(
            os.path.join(pyseqlab_dataset_dir, "train", "ACORDAOTCU25052016.conll")
        )
