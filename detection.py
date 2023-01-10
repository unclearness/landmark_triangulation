import face_alignment
from skimage import io
import cv2
import numpy as np
import os

fa = face_alignment.FaceAlignment(
    face_alignment.LandmarksType._2D, flip_input=False)

#fnames = [x for x in os.listdir("./data/") if x.endswith('png')]
fnames = [str(i).zfill(2) + ".png" for i in range(0, 22)]

preds_list = []
for fname in fnames:
    input = io.imread('./data/' + fname)
    input = input[..., :3]
    preds = fa.get_landmarks(input)

    input = np.array(input)
    # Convert to BGR for cv2
    input = input[..., ::-1].astype(np.uint8).copy()
    if preds is None:
        preds_list.append(None)
        continue
    for i, p in enumerate(preds[0]):
        p = (int(p[0]), int(p[1]))
        cv2.circle(input, p, 3, [0, 0, 255], -1)
        cv2.putText(input, str(i), p, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 0),
                    thickness=2)

    cv2.imwrite('./data/detected' + fname, input)
    preds_list.append(preds)

with open('./data/detected.txt', 'w') as f:
    for preds in preds_list:
        if preds is None:
            f.write("null\n")
            continue
        line = []
        for p in preds[0]:
            line += [str(p[0]), str(p[1])]
        line = " ".join(line)
        line += '\n'
        f.write(line)
