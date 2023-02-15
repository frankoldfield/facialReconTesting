import feat
import matplotlib
from feat import Detector
from matplotlib import pyplot as plt

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

detector
feat.detector.Detector(face_model="retinaface", landmark_model="mobilefacenet", au_model="xgb", emotion_model="resmasknet", facepose_model="img2pose")
from feat.utils.io import get_test_data_path
from feat.plotting import imshow
import os

# Helper to point to the test data folder
test_data_dir = get_test_data_path()

# Get the full path
single_face_img_path = "sample5.jpg"

# Plot it
imshow(single_face_img_path)

single_face_prediction = detector.detect_image(single_face_img_path)

# Show results
single_face_prediction
single_face_prediction.to_csv("output.csv", index=False)
# prefer to pandas read_csv
from feat.utils.io import read_feat

input_prediction = read_feat("output.csv")

# Show results
input_prediction
figs = single_face_prediction.plot_detections(poses=True)
plt.show()