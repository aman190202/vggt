import json
import gzip

annotation_file = "apple_train.jgz"

with gzip.open(annotation_file, "r") as fin:
    annotation = json.loads(fin.read())

total_frame_num = 0
data_store = {}
import numpy as np
for seq_name, seq_data in annotation.items():
    total_frame_num += len(seq_data)
    metadata = seq_data
    ann = metadata[0]
    extri_opencv = np.array(ann["extri"])
    intri_opencv = np.array(ann["intri"])
    print(extri_opencv)
    print(intri_opencv)
    break


