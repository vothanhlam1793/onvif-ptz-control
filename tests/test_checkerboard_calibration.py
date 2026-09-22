import sys
import tempfile
import unittest
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.onvif.checkerboard_calibration import detect_corners, generate_pdf_target, generate_target


class CheckerboardCalibrationTests(unittest.TestCase):
    def test_generated_target_has_all_expected_corners(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "checkerboard.png"
            generate_target(target_path, square_px=80, margin_px=80)
            frame = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)

            corners = detect_corners(frame)

        self.assertIsNotNone(corners)
        self.assertEqual((54, 2), corners.shape)

    def test_generated_pdf_is_a4_vector_document(self):
        with tempfile.TemporaryDirectory() as directory:
            target_path = Path(directory) / "checkerboard.pdf"
            generate_pdf_target(target_path)
            content = target_path.read_bytes()

        self.assertTrue(content.startswith(b"%PDF-1.4"))
        self.assertIn(b"/MediaBox [0 0 595.276 841.890]", content)
        self.assertTrue(content.endswith(b"%%EOF\n"))


if __name__ == "__main__":
    unittest.main()
