from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR

parser = ArgumentParser(description="Text Mask Generation")
parser.add_argument(
    "--directory",
    type=str,
    default=".",
    help="Directory containing images to process",
)
parser.add_argument(
    "--output",
    type=str,
    default="output",
    help="Directory to save the output masks",
)

args = parser.parse_args()

input_dir = Path(args.directory)
output_dir = Path(args.output)

if not input_dir.is_dir():
    raise ValueError(
        f"Input directory {input_dir} does not exist or is not a directory."
    )

output_dir.mkdir(parents=True, exist_ok=True)

images = input_dir.glob("*.*")
images = iter(images)

for image_path in images:
    try:
        if not image_path.suffix.endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image = cv2.imread(str(image_path))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ocr = PaddleOCR(lang="en")

        results = ocr.ocr(gray, cls=True)

        mask = np.zeros_like(gray)

        for line in results:
            if not line:
                continue

            for word in line:
                if not word:
                    continue

                bbox = word[0]
                top_left = tuple(map(int, bbox[0]))
                bottom_right = tuple(map(int, bbox[2]))
                cv2.rectangle(mask, top_left, bottom_right, 255, -1)

        cv2.imwrite(f"{output_dir}/{image_path.name}", mask)
    except Exception as e:
        with open(output_dir / "error.log", "a") as log_file:
            log_file.write(f"Failed to process {image_path}: {e}\n")
