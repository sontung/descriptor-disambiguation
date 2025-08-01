import numpy as np
from ablation_studies import find_numbers


aachen = {
    "mixvpr": "0.0 & 0.4 & 26.5 & 0.0 & 0.5 & 28.3",
    "eigenplaces": "0.0 & 0.6 & 26.6 & 0.0 & 0.5 & 27.7",
    "salad": "0.0 & 0.5 & 27.9 & 0.0 & 0.5 & 27.7",
    "crica": "0.0 & 0.4 & 26.8 & 0.0 & 0.0 & 30.4",
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
    "d2_megaloc_0.4": "86.9 / 92.8 / 97.6	72.8 / 84.8 / 93.2 ",
    "glace": "8.6 / 20.8 / 64.0	1.0 / 1.0 / 17.3",
}

robotcar = {
    "mixvpr": "11.0 & 38.3 & 99.3 & 2.3 & 9.1 & 60.4",
    "eigenplaces": "8.9 & 33.2 & 99.3 & 0.5 & 4.7 & 37.8",
    "salad": "8.2 & 31.3 & 99.4 & 4.4 & 14.2 & 93.9",
    "crica": "9.2 & 31.8 & 98.1 & 3.3 & 13.3 & 75.1",
    "hloc": "64.8 & 94.5 & 99.9 & 39.2 & 80.0 & 92.3",
    "megloc": "66.6 & 95.4 & 100.0 & 51.5 & 90.0 & 100.0 ",
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
    "d2_megaloc_0.4": "61.3 / 93.9 / 100.0	31.7 / 75.8 / 97.0",
    "topk=500": "61.3 / 93.9 / 100.0	34.5 / 83.2 / 98.8",
    "topk=400": "61.5 / 94.0 / 100.0	32.2 / 83.2 / 99.1",
    "d2_salad_heavy_0.3": "61.1 / 93.1 / 100.0	32.9 / 80.9 / 99.1",
    "d2_salad_light_0.3": "	61.4 / 94.0 / 100.0	24.0 / 62.5 / 89.7",
}

cmu = {
    "mixvpr": "15.5 & 39.5 & 97.4 & 6.0 & 22.6 & 97.8 & 6.1 & 23.1 & 92.9",
    "eigenplaces": "13.7 & 35.8 & 96.9 & 5.4 & 21.1 & 97.4 & 4.7 & 20.0 & 90.1",
    "salad": "12.0 & 32.2 & 97.8 & 5.0 & 18.3 & 97.5 & 4.5 & 18.4 & 94.4",
    "crica": "13.6 & 34.6 & 97.3 & 5.0 & 19.0 & 97.8 & 4.9 & 20.0 & 94.0",
    "hloc": "95.5 / 98.6 / 99.3	90.9 / 94.2 / 97.1	85.7 / 89.0 / 91.6	",
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
    "d2_megaloc_0.4": "	91.6 / 96.6 / 99.1	92.2 / 95.7 / 99.4	80.1 / 86.8 / 94.7",
}

method_dict = {}
for name, ds in [["aachen", aachen], ["robotcar", robotcar], ["cmu", cmu]]:

    for method_ in ds:
        count = np.array([0, 0, 0])
        scores = np.array([0, 0, 0], dtype=float)
        res = ds[method_]
        if len(res) == 0:
            continue
        if "&" not in res:
            res = find_numbers(res, True)
        else:
            res = list(map(float, res.split("&")))
        assert len(res) % 3 == 0, f"{name} {method_}"
        for i in range(0, len(res), 3):
            scores += res[i : i + 3]
            count += 1
        scores /= count
        method_dict.setdefault(method_, []).extend(scores)

for method_ in method_dict:
    res = method_dict[method_]
    avg = round(np.mean(res), 1)
    res = list(map(lambda du: round(du, 1), res))
    res.append(avg)
    res = list(map(str, res))
    print(method_)
    print(" & ".join(res))
