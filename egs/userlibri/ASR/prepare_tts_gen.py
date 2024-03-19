import logging
import os
import re
import shutil
import tarfile
import zipfile
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import sys

from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import AlignmentItem, SupervisionSegment, SupervisionSet
from lhotse.utils import (
    Pathlike,
    is_module_available,
    safe_extract,
)

def prepare_userlibri(
    corpus_dir: str,
    dataset_parts: str = "auto",
    output_dir: str = None,
    gen_dir: str = None,
    num_jobs: int = 1,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.
    When all the manifests are available in the ``output_dir``, it will simply read and return them.
    :param corpus_dir: Pathlike, the path of the data dir.
    :param dataset_parts: string or sequence of strings representing dataset part names, e.g. 'train-clean-100', 'train-clean-5', 'dev-clean'.
        By default we will infer which parts are available in ``corpus_dir``.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict whose key is the dataset part, and the value is Dicts with the keys 'audio' and 'supervisions'.
    """
    
    corpus_dir_str = corpus_dir
    gen_dir_str = gen_dir

    corpus_dir = Path(corpus_dir)
    gen_dir = Path(gen_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    spkwise_parent = corpus_dir / gen_dir
    spks = os.listdir(spkwise_parent)
    spks_ = []
    for spk in spks:
        if "txt" not in spk:
            spks_.append(spk)
    spks = spks_

    spks_parts = (
        set(spks)
    )

    manifests = {}

    for s_or_b, dataset_parts in zip([gen_dir], [spks_parts]):
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Maybe the manifests already exist: we can read them and save a bit of preparation time.
            manifests = read_manifests_if_cached(
                dataset_parts=dataset_parts, output_dir=output_dir
            )

        with ThreadPoolExecutor(num_jobs) as ex:
            for part in tqdm(dataset_parts, desc="Dataset parts"):
                logging.info(f"Processing UserLibri subset: {part}")
                if manifests_exist(part=part, output_dir=output_dir):
                    logging.info(f"UserLibri subset: {part} already prepared - skipping.")
                    continue
                recordings = []
                supervisions = []
                part_path = corpus_dir / s_or_b / part
                
                text_file = part + "_lm_train.txt"
                trans_path = corpus_dir_str + "/" + gen_dir_str + "/" + text_file
                trans_parent = corpus_dir_str + "/" + gen_dir_str
                futures = []

                alignments = {}
                with open(trans_path) as f:
                    for line in f:
                        futures.append(
                            ex.submit(parse_utterance, part_path, line, alignments)
                        )

                for future in tqdm(futures, desc="Processing", leave=False):
                    result = future.result()
                    if result is None:
                        continue
                    recording, segment = result
                    recordings.append(recording)
                    supervisions.append(segment)

                recording_set = RecordingSet.from_recordings(recordings)
                supervision_set = SupervisionSet.from_segments(supervisions)

                validate_recordings_and_supervisions(recording_set, supervision_set)

                if output_dir is not None:
                    supervision_set.to_file(
                        output_dir / f"userlibri_supervisions_{part}.jsonl.gz"
                    )
                    recording_set.to_file(
                        output_dir / f"userlibri_recordings_{part}.jsonl.gz"
                    )

                manifests[part] = {
                    "recordings": recording_set,
                    "supervisions": supervision_set,
                }

    return manifests


def parse_utterance(
    dataset_split_path: Path,
    line: str,
    alignments: Dict[str, List[AlignmentItem]],
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    recording_id, text = line.strip().split(maxsplit=1)
    # Create the Recording first
    audio_path = (
        dataset_split_path
        / f"{recording_id}.wav"
    )
    if not audio_path.is_file():
        logging.warning(f"No such file: {audio_path}")
        return None
    recording = Recording.from_file(audio_path, recording_id=recording_id)
    # Then, create the corresponding supervisions
    segment = SupervisionSegment(
        id=recording_id,
        recording_id=recording_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language="English",
        speaker=re.sub(r"-.*", r"", recording.id),
        text=text.strip(),
        alignment={"word": alignments[recording_id]}
        if recording_id in alignments
        else None,
    )
    return recording, segment


def parse_alignments(ali_path: Pathlike) -> Dict[str, List[AlignmentItem]]:
    alignments = {}
    for line in Path(ali_path).read_text().splitlines():
        utt_id, words, timestamps = line.split()
        words = words.replace('"', "").split(",")
        timestamps = [0.0] + list(map(float, timestamps.replace('"', "").split(",")))
        alignments[utt_id] = [
            AlignmentItem(
                symbol=word, start=start, duration=round(end - start, ndigits=8)
            )
            for word, start, end in zip(words, timestamps, timestamps[1:])
        ]
    return alignments

def main():
    nj = 15
    output_dir = "data/manifests"
    corpus_dir = sys.argv[1]
    gen_dir = sys.argv[2]

    prepare_userlibri(corpus_dir, "auto", output_dir, gen_dir, nj)

main()
