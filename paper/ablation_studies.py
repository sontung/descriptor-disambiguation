import re
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib.pylab import plt


aachen = {
    "random-0": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
    "first": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
    "last": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
    "central": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
}


robotcar = {
    "random-0": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
    "first": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
    "last": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
    "central": {
        "eigenplaces": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "mixvpr": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "crica": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
        "salad": {
            0.1: "",
            0.2: "",
            0.3: "",
            0.4: "",
            0.5: "",
            0.6: "",
            0.7: "",
            0.8: "",
            0.9: "",
            1.0: "",
        },
    },
}


def find_numbers(string_):
    pattern = r"[-+]?(?:\d*\.*\d+)"
    # res = "53.2 / 85.8 / 95.3	3.5 / 11.7 / 25.2"
    matches = re.findall(pattern, string_)
    numbers = list(map(float, matches))
    avg = sum(numbers) / len(matches)
    return avg


def main():
    plt.figure(figsize=(6, 10))

    plt.subplot(211)
    ds = aachen
    plt.ylim(0, 100)
    plt.xticks(np.arange(1, 11) / 10)
    plt.xlabel("lambda")
    plt.ylabel("% successfully localized images")
    markers = {
        "mixvpr": "o",
        "eigen": "d",
        "crica": "v",
        "salad": "s",
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
        print(method_, all_numbers)
        plt.plot(
            np.arange(1, 11) / 10, all_numbers, marker=markers[method_], label=method_
        )
    plt.legend(loc=4)

    plt.subplot(212)

    ds = robotcar
    plt.ylim(0, 100)
    plt.xticks(np.arange(1, 11) / 10)
    plt.xlabel("lambda")
    plt.ylabel("% successfully localized images")
    markers = {
        "mixvpr": "o",
        "eigen": "d",
        "crica": "v",
        "salad": "s",
    }
    plt.title("RobotCar Seasons v2")
    plt.axhline(y=78.5, color="r", linestyle="--", label="hloc")
    plt.axhline(y=58.3, color="b", linestyle="--", label="vanilla")

    for method_ in ds:
        all_numbers = []
        for param_ in ds[method_]:
            res = ds[method_][param_]
            avg_res = find_numbers(res)
            all_numbers.append(avg_res)
        print(method_, all_numbers)
        plt.plot(
            np.arange(1, 11) / 10, all_numbers, marker=markers[method_], label=method_
        )
    plt.legend(loc=4)
    plt.tight_layout()

    plt.savefig(
        "ablation.pdf", format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.1
    )


if __name__ == "__main__":
    main()
