import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_tissues as training


class HomologyChecks(unittest.TestCase):
    def args(self, *flags):
        with patch.object(sys, "argv", ["train", *flags]):
            args = training.parse_args()
        args.region_len, args.nBins = 12, 4
        return args

    def test_mapping_validation_and_seed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "mapping.tsv"
            path.write_text("cluster_id\ttranscript_id\nc1\tt1\nc1\tt1\nc2\tt2\n")
            self.assertEqual(training.load_homology(path), {"t1": "c1", "t2": "c2"})
            path.write_text("cluster_id\ttranscript_id\nc1\tt1\nc2\tt1\n")
            with self.assertRaisesRegex(ValueError, "multiple"):
                training.load_homology(path)
            path.write_text("gene_id\ttranscript_id\ng1\tt1\n")
            with self.assertRaisesRegex(ValueError, "columns"):
                training.load_homology(path)
        clusters = [f"c{i}" for i in range(100)]
        tr, vl = training.split_homology_clusters(clusters, 4)
        self.assertEqual((len(tr), len(vl)), (70, 30))
        self.assertFalse(tr & vl)
        self.assertEqual((tr, vl), training.split_homology_clusters(reversed(clusters), 4))
        self.assertNotEqual((tr, vl), training.split_homology_clusters(clusters, 5))
        with self.assertRaisesRegex(ValueError, "two eligible"):
            training.split_homology_clusters(["one"], 4)

    def test_incompatible_modes(self):
        for flags in [("--test",), ("--trial", "1,2,3")]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                self.args("--homology", "mapping.tsv", *flags)

    def fixture(self, root):
        frames, mapping, tracks = {}, {}, []
        for sp in ("human", "mouse"):
            ids = [f"{sp}_{i}_{iso}" for i in range(10) for iso in range(2)]
            mapping.update({tx: f"c{i // 2}" for i, tx in enumerate(ids)})
            ids += [f"{sp}_unmapped", f"{sp}_ineligible", f"{sp}_noncoding", f"{sp}_empty"]
            mapping.update({tx: "excluded_cluster" for tx in ids[-3:]})
            frame = pd.DataFrame(dict(id=ids, sequence=["ATGACCTGATGC"] * 23 + ["N" * 12],
                                      region=["1" * 3 + "2" * 9] * 24,
                                      biotype=["protein_coding"] * 24,
                                      chr=(["1", "2", "3", "4", "5", "6", "7", "8", "16", "X"] * 3)[:24],
                                      species=sp))
            frame.loc[22, "biotype"] = "lncRNA"
            frames[sp] = frame
            for tissue in ("heart", "liver"):
                tc = tissue + "_normal"
                tracks.append(dict(species=sp, tissue=tissue, condition="normal", tissue_condition=tc))
                inp, out = training._npy_paths(str(root), sp, tc, 12, 4, False)
                Path(inp).parent.mkdir(parents=True, exist_ok=True)
                np.save(inp, np.ones((24, 12), dtype=np.float32))
                np.save(out, np.ones((24, 4), dtype=np.float32))
                eligibility = np.ones(24, dtype=bool); eligibility[21] = False
                np.save(inp.replace("_log_rnaseq_final_v2.npy", "_eligible_v2.npy"), eligibility)
        path = root / "homology.tsv"
        path.write_text("cluster_id\ttranscript_id\n" + "".join(f"{c}\t{t}\n" for t, c in mapping.items()))
        return frames, tracks, path, mapping

    def test_data_module_global_split_filters_and_no_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames, tracks, path, mapping = self.fixture(root)
            args = self.args("--homology", str(path))
            args.tracks_dir = str(root); args.save_dir = str(root / "results")
            def make_dm(items):
                return training.TissueRiboDataModule(args, items, [], {"heart": 0, "liver": 1}, {"normal": 0}, test_tracks=[{"species": "must_not_load"}])
            with patch.object(training, "load_coords", side_effect=lambda sp: frames[sp]), patch.object(training, "split_chromosomes", side_effect=AssertionError("Chromosome splitting used")):
                dm = make_dm(tracks); dm.setup("fit")
                for array in dm._mmap_refs: array._mmap.close()
                # Simulate another rank seeing tracks in a different order.
                with patch.object(training, "_RANK", 1):
                    other = make_dm(list(reversed(tracks))); other.setup("fit")
                    for array in other._mmap_refs: array._mmap.close()
            self.assertEqual(dm.train_clusters, other.train_clusters)
            self.assertEqual((len(dm.train_clusters), len(dm.val_clusters)), (7, 3))
            self.assertEqual((len(dm._train_ds), len(dm._val_ds)), (56, 24))
            self.assertIsNone(dm._test_ds); self.assertIsNone(dm._test_new_ds)
            self.assertEqual(dm.test_tracks, [])
            seen = {}
            for phase, dataset in [("train", dm._train_ds), ("val", dm._val_ds)]:
                for part in dataset.datasets:
                    sp = dm.track_names[part.track_id][0]
                    for idx in part.index_array:
                        self.assertLess(idx, 20)
                        cluster = mapping[frames[sp].iloc[idx]["id"]]
                        self.assertEqual(seen.setdefault(cluster, phase), phase)
            record = json.loads((root / "results/homology_split.json").read_text())
            self.assertEqual(set(record["train_clusters"]), dm.train_clusters)
            self.assertEqual(len(seen), 10)
            self.assertNotIn("excluded_cluster", seen)

    def test_chromosome_mode_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames, tracks, _, _ = self.fixture(root)
            args = self.args(); args.tracks_dir = str(root)
            dm = training.TissueRiboDataModule(args, tracks[:1], ["2", "3", "4"], {"heart": 0}, {"normal": 0})
            with patch.object(training, "load_coords", side_effect=lambda sp: frames[sp]):
                dm.setup("fit")
                for array in dm._mmap_refs: array._mmap.close()
            for dataset, expected in [(dm._train_ds, {"5", "6", "7", "8"}), (dm._val_ds, {"2", "3", "4"}), (dm._test_ds, {"1", "16", "X"})]:
                indices = dataset.datasets[0].index_array
                self.assertTrue(set(frames["human"].iloc[indices]["chr"]) <= expected)
                self.assertGreater(len(indices), 0)


if __name__ == "__main__":
    unittest.main()
