
import contextlib
import glob
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import ExifTags, Image
from tqdm import tqdm


def convert_coco_json(json_dir="../coco/annotations/", save_dir='/home/luke/data/yolo_v8/'):
    """Converts COCO JSON format to YOLO label format, with options for segments and class mapping."""
    
    # Import json
    for json_file in sorted(Path(json_dir).resolve().glob("*.json")):
        split, label = json_file.stem.replace("instances_", "").split('_')
        fn = Path(save_dir) / split / label  # folder name
        fn.mkdir(exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image dict
        images = {"%g" % x["id"]: x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)

        # Write labels file
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images["%g" % img_id]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                cls = ann["category_id"]
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

if __name__ == "__main__":
    """ This is executed when run from the command line """
    convert_coco_json('/home/luke/data/yolo_v8', save_dir='/home/luke/data/yolo_v8')