import sys
import tempfile
import unittest
from pathlib import Path
import csv
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from training_curves import record_epoch, save_training_curves

class Checks(unittest.TestCase):
    def test_epoch_alignment_and_vector_pdfs(self):
        history = []
        record_epoch(history, 0, "train", {"loss": 3, "pcc": .1})
        record_epoch(history, 1, "val", {"loss": 2.5, "pcc": .3})
        record_epoch(history, 1, "train", {"loss": 2, "pcc": .4})
        self.assertEqual(len(history), 2)
        with tempfile.TemporaryDirectory() as tmp:
            save_training_curves(history, tmp)
            with open(Path(tmp)/"training_history.csv", newline="") as f:
                rows=list(csv.DictReader(f))
            self.assertEqual(rows[0]["val_loss"], "")
            self.assertEqual(rows[1]["train_pcc"], "0.4")
            self.assertEqual(rows[1]["val_pcc"], "0.3")
            for metric in ("loss", "pcc"):
                data=(Path(tmp)/f"training_validation_{metric}.pdf").read_bytes()
                self.assertTrue(data.startswith(b"%PDF"))
                self.assertNotIn(b"/Subtype /Image", data)
                self.assertGreater(len(data), 1000)

    def test_no_validation_or_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_training_curves([], tmp)
            self.assertFalse(list(Path(tmp).iterdir()))
            history=[]
            record_epoch(history, 0, "train", {"loss": 1, "pcc": .2})
            save_training_curves(history, tmp)
            self.assertEqual(len(list(Path(tmp).glob("*.pdf"))), 2)

if __name__ == "__main__": unittest.main()
