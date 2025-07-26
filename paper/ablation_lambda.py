import re
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.pylab import plt

aachen = {
    "mixvpr": {
        0.1: "19.3 / 32.0 / 63.0	4.7 / 11.0 / 39.3",
        0.2: "42.5 / 52.7 / 72.5	16.2 / 26.2 / 46.1",
        0.3: "80.6 / 87.5 / 94.2	60.2 / 79.1 / 86.4",
        0.4: "85.8 / 91.9 / 96.7	69.1 / 85.9 / 92.1",
        0.5: "85.2 / 90.9 / 95.5	69.6 / 81.7 / 87.4",
        0.6: "85.0 / 90.4 / 95.0	64.4 / 76.4 / 84.8",
        0.7: "84.8 / 90.2 / 95.1	61.8 / 74.3 / 82.7",
        0.8: "85.0 / 90.4 / 94.9	60.2 / 71.2 / 81.2",
        0.9: "84.0 / 89.4 / 94.7	61.3 / 72.3 / 82.2",
        1.0: "84.0 / 89.2 / 94.8	61.3 / 71.7 / 80.6",
    },
    "eigen": {
        0.1: "19.1 / 31.8 / 61.7	5.2 / 13.6 / 41.4",
        0.2: "40.9 / 51.1 / 71.4	19.9 / 29.3 / 49.2",
        0.3: "73.3 / 82.3 / 90.9	58.1 / 70.7 / 83.8",
        0.4: "83.5 / 90.3 / 96.4	67.0 / 82.2 / 94.2",
        0.5: "86.7 / 92.6 / 97.2	70.2 / 86.9 / 94.2",
        0.6: "86.9 / 92.0 / 96.1	71.2 / 85.3 / 93.2",
        0.7: "84.7 / 90.8 / 95.5	66.0 / 78.5 / 88.0",
        0.8: "84.3 / 89.9 / 94.9	61.3 / 72.3 / 83.2",
        0.9: "84.5 / 89.9 / 94.7	62.3 / 73.3 / 82.7",
        1.0: "84.0 / 89.2 / 94.8	61.3 / 71.7 / 80.6",
    },
    "salad": {
        0.1: "20.4 / 31.1 / 62.5	5.2 / 11.5 / 36.1",
        0.2: "76.6 / 83.7 / 91.9	53.9 / 73.3 / 84.3",
        0.3: "86.3 / 91.5 / 97.3	69.6 / 85.9 / 93.7",
        0.4: "85.9 / 91.5 / 96.5	70.7 / 84.8 / 90.1",
        0.5: "84.5 / 90.5 / 95.9	67.0 / 78.5 / 85.9",
        0.6: "83.7 / 89.7 / 94.8	64.9 / 76.4 / 83.2",
        0.7: "84.3 / 89.9 / 94.7	62.3 / 72.8 / 82.2",
        0.8: "84.2 / 89.4 / 94.5	62.8 / 73.8 / 81.7",
        0.9: "84.2 / 89.6 / 94.7	61.3 / 72.8 / 81.7",
        1.0: "84.0 / 89.2 / 94.8	61.3 / 71.7 / 80.6",
    },
    "crica": {
        0.1: "19.9 / 32.0 / 61.9	2.1 / 7.9 / 39.3",
        0.2: "80.2 / 87.3 / 94.4	62.3 / 78.5 / 89.5",
        0.3: "85.7 / 91.1 / 96.7	72.3 / 85.9 / 93.7",
        0.4: "85.6 / 91.0 / 96.4	70.7 / 83.2 / 90.1",
        0.5: "85.2 / 90.0 / 95.4	67.0 / 78.5 / 86.4",
        0.6: "84.6 / 89.6 / 94.9	64.9 / 75.9 / 84.3",
        0.7: "84.6 / 89.7 / 94.8	61.3 / 72.3 / 80.1",
        0.8: "84.8 / 89.7 / 94.9	62.3 / 72.3 / 81.7",
        0.9: "84.5 / 89.3 / 94.9	61.8 / 71.7 / 81.2",
        1.0: "84.0 / 89.2 / 94.8	61.3 / 71.7 / 80.6",
    },
}

robotcar = {
    "salad": {
        0.1: "46.4 / 82.2 / 99.3	4.2 / 15.9 / 57.8",
        0.2: "60.8 / 93.1 / 100.0	26.6 / 69.9 / 94.6",
        0.3: "61.1 / 93.1 / 100.0	32.9 / 80.9 / 99.1",
        0.4: "61.3 / 93.6 / 100.0	30.1 / 74.6 / 98.6",
        0.5: "61.6 / 94.0 / 100.0	22.4 / 58.5 / 86.0",
        0.6: "60.8 / 94.0 / 100.0	15.4 / 42.0 / 65.7",
        0.7: "60.9 / 93.7 / 100.0	14.0 / 37.1 / 53.6",
        0.8: "60.8 / 93.8 / 99.9	12.1 / 34.0 / 51.5",
        0.9: "60.8 / 93.7 / 99.9	12.8 / 33.1 / 49.2",
        1.0: "60.8 / 93.8 / 99.9	11.9 / 33.6 / 49.7",
    },
    "eigen": {
        0.1: "30.5 / 65.9 / 96.9	1.2 / 8.6 / 33.6",
        0.2: "57.6 / 90.9 / 99.9	10.0 / 24.2 / 40.6",
        0.3: "60.1 / 93.3 / 100.0	12.8 / 31.5 / 46.9",
        0.4: "61.1 / 93.6 / 100.0	14.5 / 36.8 / 49.4",
        0.5: "60.8 / 93.7 / 100.0	17.5 / 42.0 / 54.1",
        0.6: "61.3 / 94.0 / 100.0	18.4 / 43.1 / 59.2",
        0.7: "60.9 / 93.8 / 99.9	14.5 / 37.8 / 56.2",
        0.8: "60.9 / 93.8 / 100.0	12.8 / 35.7 / 51.5",
        0.9: "60.6 / 93.8 / 99.9	11.9 / 34.0 / 49.7",
        1.0: "60.8 / 93.8 / 99.9	11.9 / 33.6 / 49.7",
    },
    "mixvpr": {
        0.1: "31.2 / 66.0 / 97.8	2.3 / 11.7 / 47.1",
        0.2: "59.0 / 92.1 / 99.5	13.3 / 35.7 / 62.9",
        0.3: "	60.4 / 93.4 / 99.9	17.5 / 47.6 / 68.5",
        0.4: "61.4 / 93.8 / 100.0	23.3 / 55.5 / 75.1",
        0.5: "61.6 / 94.0 / 100.0	22.4 / 53.8 / 78.3",
        0.6: "61.9 / 93.8 / 100.0	17.2 / 46.4 / 68.1",
        0.7: "61.3 / 94.0 / 100.0	14.5 / 38.0 / 56.6",
        0.8: "60.8 / 93.8 / 99.9	12.1 / 35.2 / 52.9",
        0.9: "60.8 / 93.8 / 99.9	11.2 / 33.8 / 50.3",
        1.0: "60.8 / 93.8 / 99.9	11.9 / 33.6 / 49.7",
    },
    "crica": {
        0.1: "49.0 / 85.5 / 97.9	5.6 / 18.6 / 55.2",
        0.2: "60.0 / 93.4 / 99.9	24.2 / 62.9 / 80.9",
        0.3: "61.3 / 93.7 / 100.0	30.1 / 69.7 / 86.7",
        0.4: "61.4 / 93.7 / 100.0	25.2 / 62.5 / 86.0",
        0.5: "60.8 / 94.0 / 100.0	17.5 / 47.1 / 71.1",
        0.6: "60.5 / 93.8 / 100.0	14.7 / 40.6 / 59.7",
        0.7: "61.1 / 93.8 / 99.9	12.4 / 36.6 / 55.5",
        0.8: "61.1 / 93.8 / 99.9	13.5 / 34.7 / 53.1",
        0.9: "61.1 / 93.7 / 99.9	11.7 / 33.1 / 50.3",
        1.0: "60.8 / 93.8 / 99.9	11.9 / 33.6 / 49.7",
    },
}


def find_numbers(string_):
    pattern = r"[-+]?(?:\d*\.*\d+)"
    matches = re.findall(pattern, string_)
    numbers = list(map(float, matches))
    avg = sum(numbers) / len(matches)
    return avg


def main():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.sans-serif": ["Helvetica"],
            "font.size": 12,  # Set the global font size
            "text.latex.preamble": r"\usepackage{amsmath}",
        }
    )

    plt.figure(figsize=(5, 7))

    plt.subplot(211)
    ds = aachen
    plt.ylim(0, 100)
    plt.xticks(np.arange(1, 11) / 10)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("\% successfully localized images")
    markers = {
        "mixvpr": "o",
        "eigen": "d",
        "crica": "v",
        "salad": "s",
    }
    tableau_colors = plt.get_cmap("tab10")

    colors = {
        "mixvpr": tableau_colors(0),
        "eigen": tableau_colors(1),
        "crica": tableau_colors(2),
        "salad": tableau_colors(4),
    }

    plt.title("Aachen Day/Night v1.1")
    plt.axhline(y=92.1, color="r", linestyle="--", label="hloc")
    plt.axhline(y=80.3, color="b", linestyle="--", label="vanilla")

    for method_ in ds:
        all_numbers = []
        for param_ in ds[method_]:
            res = ds[method_][param_]
            avg_res = find_numbers(res)
            all_numbers.append(avg_res)
        print(method_, max(all_numbers))
        plt.plot(
            np.arange(1, 11) / 10,
            all_numbers,
            marker=markers[method_],
            color=colors[method_],
            label=method_,
        )
    # plt.legend(loc=4, fontsize=9, ncol=1)

    plt.subplot(212)

    ds = robotcar
    plt.ylim(0, 100)
    plt.xticks(np.arange(1, 11) / 10)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("\% successfully localized images")
    plt.title("RobotCar Seasons v2")
    hloc_line = plt.axhline(y=78.5, color="r", linestyle="--", label="hloc")
    vanilla_line = plt.axhline(y=58.3, color="b", linestyle="--", label="vanilla")

    lines = []
    labels = []
    for method_ in ds:
        all_numbers = []
        for param_ in ds[method_]:
            res = ds[method_][param_]
            avg_res = find_numbers(res)
            all_numbers.append(avg_res)
        print(method_, max(all_numbers))
        line, = plt.plot(
            np.arange(1, 11) / 10,
            all_numbers,
            marker=markers[method_],
            color=colors[method_],
            label=method_,
        )
        lines.append(line)
        labels.append(method_)

    # lines = lines + [hloc_line, vanilla_line]
    # labels = labels + ["hloc", "vanilla"]
    ax = plt.gca()

    # Plot legend for second subplot only
    legend = ax.legend(
        lines + [hloc_line, vanilla_line],
        labels + ["hloc", "vanilla"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.3),  # Centered below the subplot
        ncol=3,
        fontsize=9,
        frameon=False,
    )

    # Make room at bottom
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Add vertical space

    plt.savefig(
        "ablation.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1
    )


if __name__ == "__main__":
    main()
