import numpy as np

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
}

cmu = {
    "mixvpr": "15.5 & 39.5 & 97.4 & 6.0 & 22.6 & 97.8 & 6.1 & 23.1 & 92.9",
    "eigenplaces": "13.7 & 35.8 & 96.9 & 5.4 & 21.1 & 97.4 & 4.7 & 20.0 & 90.1",
    "salad": "12.0 & 32.2 & 97.8 & 5.0 & 18.3 & 97.5 & 4.5 & 18.4 & 94.4",
    "crica": "13.6 & 34.6 & 97.3 & 5.0 & 19.0 & 97.8 & 4.9 & 20.0 & 94.0",
    "hloc": "96.9&98.9&99.3&93.3&95.4&97.1&87.0&89.5&91.6 ",
    "active_search": "81.0 & 87.3 & 92.4 & 62.6 & 70.9 & 81.0 & 45.5 & 51.6 & 62.0 ",
    "r2d2": "74.0 & 78.3 & 84.8 & 57.6 & 62.1 & 71.9 & 36.1 & 39.8 & 49.2 ",
    "sift": "53.0 & 57.8 & 64.1 & 33.4 & 38.3 & 46.9 & 17.3 & 19.7 & 25.1 ",
    "d2": "87.7 & 92.8 & 96.3 & 84.5 & 88.6 & 94.4 & 64.1 & 69.9 & 78.8 ",
    "superpoint": "76.8 & 81.9 & 87.0 & 63.4 & 68.5 & 77.2 & 40.2 & 44.6 & 53.6 ",
}

method_dict = {}
for name, ds in [["aachen", aachen], ["robotcar", robotcar], ["cmu", cmu]]:

    for method_ in ds:
        count = np.array([0, 0, 0])
        scores = np.array([0, 0, 0], dtype=float)
        res = ds[method_]
        res = list(map(float, res.split("&")))
        assert len(res) % 3 == 0, f"{name} {method_}"
        for i in range(0, len(res), 3):
            scores += res[i: i+3]
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
