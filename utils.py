import os
import shutil

from typing import Union

from pyseqlab.utilities import DataFileParser


LENER_DATASET_DIR: Union[bytes, str] = os.path.join(os.path.dirname(__file__), "lener")


def join_path(*args):

    return os.path.join(*args)


def adapt_lener_to_flair(percentage: float = 1):

    assert 0 <= percentage <= 1

    flair_data_dir = join_path(LENER_DATASET_DIR, "flair")
    if os.path.isdir(flair_data_dir):
        shutil.rmtree(flair_data_dir)
    os.mkdir(flair_data_dir)

    def adapt_files(dir_name):
        with open(join_path(flair_data_dir, f"{dir_name}.txt"), "w") as new_file:
            original_dir = join_path(LENER_DATASET_DIR, dir_name)
            original_files = os.listdir(original_dir)
            percentage_limit = max(int(len(original_files) * percentage), 1)
            for i in range(0, percentage_limit):
                original_file = original_files[i]
                with open(
                    join_path(original_dir, original_file)
                ) as original_file_content:
                    content = "".join(original_file_content.readlines())
                    new_file.write(content)

    adapt_files("train")
    adapt_files("test")
    adapt_files("dev")


def adapt_lener_to_pyseqlab():

    pyseqlab_data_dir = join_path(LENER_DATASET_DIR, "pyseqlab")
    if not os.path.isdir(pyseqlab_data_dir):
        os.mkdir(pyseqlab_data_dir)
    else:
        return

    def adapt_files(dir_name):
        new_dir = join_path(pyseqlab_data_dir, dir_name)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        original_dir_path = join_path(LENER_DATASET_DIR, dir_name)
        for dir_file in os.listdir(original_dir_path):
            with open(join_path(original_dir_path, dir_file)) as _f:
                lines = _f.readlines()
                if lines[0] != "w chunk\n":
                    with open(
                        join_path(original_dir_path, "temp_" + dir_file), "w"
                    ) as output:
                        output.write("w chunk\n")
                        output.write("".join(lines))
                    shutil.move(
                        join_path(original_dir_path, "temp_" + dir_file),
                        join_path(new_dir, dir_file),
                    )

    adapt_files("train")
    adapt_files("test")
    adapt_files("dev")


class PySeqLabSequenceBuilder:
    def __init__(self, root_path):

        self._dev_directory = join_path(root_path, "dev")
        self._train_directory = join_path(root_path, "train")
        self._test_directory = join_path(root_path, "test")

    def generate_sequences(self, sequence_type):

        if sequence_type == "dev":
            sequence_path = self._dev_directory
        elif sequence_type == "train":
            sequence_path = self._train_directory
        elif sequence_type == "test":
            sequence_path = self._test_directory
        else:
            raise Exception("unknown sequence type")

        sequences = []

        # TODO: reduce memory consumption
        for file in os.listdir(sequence_path):
            sequences.extend(
                self.build_sequences_from_file(join_path(sequence_path, file))
            )

        return sequences

    @staticmethod
    def build_sequences_from_file(file_path):

        return [
            seq
            for seq in DataFileParser().read_file(
                file_path, header="main", y_ref=True, column_sep=" "
            )
        ]
