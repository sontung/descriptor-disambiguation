import argparse
import itertools

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from submit_vloc_entries import view


def get_args():
    parser = argparse.ArgumentParser(description="Login to a website.")
    parser.add_argument("username", type=str, help="Your username")
    parser.add_argument("password", type=str, help="Your password")

    args = parser.parse_args()

    # Call the main function with parsed arguments
    global USERNAME, PASSWORD
    USERNAME, PASSWORD = args.username, args.password


get_args()
ENTRIES = view(USERNAME, PASSWORD)
METHOD2SCORES = {}
LOCAL_DESCRIPTORS = ["d2net", "r2d2", "dog", "superpoint"]
GLOBAL_DESCRIPTORS = ["vanilla", "mixvpr", "eigenplaces", "salad"]

MATRIX = np.zeros((len(LOCAL_DESCRIPTORS), len(GLOBAL_DESCRIPTORS)))
for data_name in ENTRIES:
    for method_name, scores in ENTRIES[data_name]:
        if method_name not in METHOD2SCORES:
            METHOD2SCORES[method_name] = [None, None, None]
        index = 0
        if "robotcar" in data_name.lower():
            index = 1
        elif "cmu" in data_name.lower():
            index = 2
        METHOD2SCORES[method_name][index] = scores

for method_name in METHOD2SCORES:
    list0 = method_name.split("_")
    i0 = LOCAL_DESCRIPTORS.index(list0[0])
    i1 = 0
    if len(list0) == 2:
        i1 = GLOBAL_DESCRIPTORS.index(list0[1])
    flattened_list = [item for sublist in METHOD2SCORES[method_name]
                      if sublist is not None for item in sublist]
    avg_score = np.mean(flattened_list)
    MATRIX[i1, i0] = avg_score
    print(list0, i0, i1, avg_score)




aachen = {
    "hloc": "89.8&96.1&99.4 & 77.0&90.6&100.0",
    "megloc": "90.5&97.3&99.8 & 77.5&92.7&100.0",
    "r2d2": "86.5 & 92.7 & 96.5 & 51.3 & 60.7 & 74.9 ",
    "sift": "86.3 & 90.3 & 94.1 & 25.7 & 29.8 & 37.2",
    "d2": "84.0 & 89.2 & 94.8 & 61.3 & 71.7 & 80.6 ",
    "superpoint": "84.5 & 91.5 & 95.8 & 50.3 & 63.9 & 75.4 ",
    "d2_mixvpr_heavy": "85.2 / 90.9 / 95.5	69.6 / 81.7 / 87.4",
    "d2_mixvpr_light": "84.6 / 89.9 / 94.3	60.2 / 72.8 / 83.8",
    "d2_eigen_heavy": "86.7 / 92.6 / 97.2	70.2 / 86.9 / 94.2",
    "d2_eigen_light": "85.3 / 91.0 / 95.9	68.6 / 81.7 / 90.6",
    "d2_salad_heavy": "84.5 / 90.5 / 95.9	67.0 / 78.5 / 85.9",
    "d2_salad_light": "84.2 / 90.0 / 95.1	59.2 / 73.8 / 83.8",
    "d2_crica_heavy": "85.2 / 90.0 / 95.4	67.0 / 78.5 / 86.4",
    "d2_crica_light": "84.0 / 89.8 / 95.1	62.3 / 73.8 / 83.8",
    "d2_salad_heavy_0.3": "86.3 / 91.5 / 97.3	69.6 / 85.9 / 93.7",
    "d2_salad_light_0.3": "	85.2 / 90.7 / 95.9	69.1 / 83.8 / 90.1",
    "glace": "8.6 / 20.8 / 64.0	1.0 / 1.0 / 17.3"

}

robotcar = {
    "r2d2": "59.9 & 88.8 & 95.5 & 4.9 & 14.7 & 23.3 ",
    "sift": "49.3 & 77.5 & 85.0 & 0.5 & 0.9 & 2.3 ",
    "d2": "60.8 & 93.8 & 99.9 & 11.9 & 33.6 & 49.7 ",
    "superpoint": "57.2 & 89.2 & 96.1 & 4.9 & 13.5 & 26.1 ",
    "d2_mixvpr_heavy": "61.6 / 94.0 / 100.0	22.4 / 53.8 / 78.3",
    "d2_mixvpr_light": "61.3 / 94.0 / 100.0	13.5 / 35.4 / 53.1",
    "d2_eigen_heavy": "60.8 / 93.7 / 100.0	17.5 / 42.0 / 54.1",
    "d2_eigen_light": "60.4 / 93.8 / 100.0	12.8 / 33.8 / 52.2",
    "d2_salad_heavy": "61.6 / 94.0 / 100.0	22.4 / 58.5 / 86.0",
    "d2_salad_light": "60.5 / 93.8 / 100.0	17.7 / 43.1 / 63.2",
    "d2_crica_heavy": "60.8 / 94.0 / 100.0	17.5 / 47.1 / 71.1",
    "topk=500": "61.3 / 93.9 / 100.0	34.5 / 83.2 / 98.8",
    "topk=400": "61.5 / 94.0 / 100.0	32.2 / 83.2 / 99.1",
    "d2_salad_heavy_0.3": "61.1 / 93.1 / 100.0	32.9 / 80.9 / 99.1",
    "d2_salad_light_0.3": "	61.4 / 94.0 / 100.0	24.0 / 62.5 / 89.7",

}

cmu = {
    "hloc": "96.9&98.9&99.3&93.3&95.4&97.1&87.0&89.5&91.6 ",
    "active_search": "81.0 & 87.3 & 92.4 & 62.6 & 70.9 & 81.0 & 45.5 & 51.6 & 62.0 ",
    "r2d2": "74.0 & 78.3 & 84.8 & 57.6 & 62.1 & 71.9 & 36.1 & 39.8 & 49.2 ",
    "sift": "53.0 & 57.8 & 64.1 & 33.4 & 38.3 & 46.9 & 17.3 & 19.7 & 25.1 ",
    "d2": "87.7 & 92.8 & 96.3 & 84.5 & 88.6 & 94.4 & 64.1 & 69.9 & 78.8 ",
    "superpoint": "76.8 & 81.9 & 87.0 & 63.4 & 68.5 & 77.2 & 40.2 & 44.6 & 53.6 ",
    "d2_mixvpr_heavy": "91.2 / 96.3 / 98.9	91.9 / 95.5 / 99.1	79.4 / 85.9 / 94.1",
    "d2_mixvpr_light": "89.6 / 94.6 / 97.7	87.9 / 91.8 / 96.7	71.6 / 78.0 / 87.4",
    "d2_eigen_heavy": "	91.1 / 96.3 / 98.8	92.3 / 95.9 / 99.3	80.8 / 86.9 / 93.9",
    "d2_eigen_light": "90.4 / 95.4 / 98.3	90.1 / 94.0 / 98.0	75.1 / 82.0 / 91.1",
    "d2_salad_heavy": "90.6 / 95.5 / 98.4	90.4 / 94.2 / 98.4	74.5 / 81.2 / 90.3",
    "d2_salad_light": "89.6 / 94.8 / 97.8	88.0 / 92.1 / 97.1	70.2 / 76.7 / 85.8",
    "d2_crica_heavy": "90.1 / 94.9 / 98.0	89.0 / 92.9 / 97.6	72.1 / 78.5 / 87.5",
    "d2_crica_light": "89.1 / 94.1 / 97.3	86.8 / 90.7 / 96.1	68.9 / 75.0 / 84.1",
    "d2_salad_heavy_0.3": "	91.1 / 96.4 / 99.0	92.2 / 96.1 / 99.5	82.2 / 88.4 / 95.4",
    "d2_salad_light_0.3": "	90.3 / 95.6 / 98.6	90.6 / 94.5 / 99.0	76.8 / 83.3 / 92.6",

}

# Generate random data
data = MATRIX

# Create a heatmap
plt.figure(figsize=(10, 6))
ax = sns.heatmap(data, annot=True, cmap="YlGnBu", vmin=0, vmax=100)
ax.set_xlabel("Local methods")
ax.set_ylabel("Global methods")

x_ticks = LOCAL_DESCRIPTORS
ax.set_xticks(np.arange(len(x_ticks)) + 0.5)  # Align tick labels to centers
ax.set_xticklabels(x_ticks)  # Rotate for readability

# Set custom y ticks
y_ticks = GLOBAL_DESCRIPTORS
ax.set_yticks(np.arange(len(y_ticks)) + 0.5)  # Align tick labels to centers
ax.set_yticklabels(y_ticks)

# Display the plot
plt.title("Heatmap of Random Data")
plt.tight_layout()

plt.savefig(
    "heatmap.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1
)
plt.close()  # This closes the plot window without displaying it
