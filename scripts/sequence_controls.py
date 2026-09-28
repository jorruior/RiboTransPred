"""Reproducible sequence controls in spliced-transcript orientation."""
import hashlib
import random
import numpy as np


def shuffle_five_prime_utr(sequence, annotation, seed=4, key="", window=4500):
    """Shuffle A/C/G/T before the first annotated CDS base; keep Ns fixed.

    Annotation uses '2' for CDS. Missing/unmarked CDS returns the input unchanged.
    Only the visible sequence window is perturbed, preserving its composition.
    """
    if not isinstance(annotation, str) or not annotation:
        return sequence
    start = annotation.find("2")
    if start < 1:
        return sequence
    stop = min(start, len(sequence), window)
    indices = [i for i in range(stop) if sequence[i].upper() in "ACGT"]
    if len(indices) < 2:
        return sequence
    digest = hashlib.blake2b(f"{int(seed)}:{key}".encode(), digest_size=16).digest()
    bases = [sequence[i] for i in indices]
    random.Random(int.from_bytes(digest, "little")).shuffle(bases)
    result = list(sequence)
    for i, base in zip(indices, bases): result[i] = base
    return "".join(result)


def training_sequences(frame, args):
    sequences = frame["sequence"].values
    if not getattr(args, "shuflledutr", False):
        return sequences
    if args.nosequence:
        raise ValueError("--nosequence and --shuflledutr cannot be combined")
    if "region" not in frame or (frame["region"].isna() | frame["region"].eq("")).any():
        raise ValueError("--shuflledutr requires coordinate files with the ninth-column CDS annotation")
    seed = args.seed if getattr(args, "shuffle_seed", None) is None else args.shuffle_seed
    return np.array([shuffle_five_prime_utr(row.sequence, row.region, seed,
                     f"{row.species}/{row.id}", args.region_len)
                     for row in frame.itertuples(index=False)], dtype=object)
