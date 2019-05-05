import argparse
import os

from pyseqlab.features_extraction import FOFeatureExtractor, HOFeatureExtractor
from pyseqlab.fo_crf import FirstOrderCRF, FirstOrderCRFModelRepresentation
from pyseqlab.ho_crf import HOCRFAD, HOCRFADModelRepresentation
from pyseqlab.hosemi_crf_ad import HOSemiCRFAD, HOSemiCRFADModelRepresentation

from crf import CRFModel
from utils import LENER_DATASET_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    output_path = os.path.join(os.path.dirname(__file__), "output")

    if args.model == "FirstOrderCRF":

        model = CRFModel(
            FirstOrderCRF,
            FirstOrderCRFModelRepresentation,
            FOFeatureExtractor,
            output_path,
        )

    elif args.model == "HOCRFAD":

        model = CRFModel(
            HOCRFAD, HOCRFADModelRepresentation, HOFeatureExtractor, output_path
        )

    elif args.model == "HOSemiCRFAD":

        model = CRFModel(
            HOSemiCRFAD, HOSemiCRFADModelRepresentation, HOFeatureExtractor, output_path
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
