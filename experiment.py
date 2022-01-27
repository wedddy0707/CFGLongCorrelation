from typing import List, Literal, Sequence, Dict, Any, Tuple
from collections import Counter
import pathlib
import yaml
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


def generate_sequence(
    a: float,
    e: float,
    length: int,
    init_symbol: Literal["Xr", "Yr"] = "Xr",
    seed: int = 1,
):
    """
    X -> XX [a^2(1-e)] | XY [a(1-a)(1-e)] | YX [a(1-a)(1-e)] | YY [(1-a)^2(1-e)] | 0 (e)
    Y -> XX [(1-a)^2(1-e)] | XY [a(1-a)(1-e)] | YX [a(1-a)(1-e)] | YY [a^2(1-e)] | 1 (e)
    """
    assert 0 <= a <= 1, a
    assert 0 <= e <= 1, e
    assert length > 0, length

    random_state = np.random.RandomState(seed)
    count_right_most_node = 0
    sequence: List[int] = []
    queue = [init_symbol]
    while len(queue) > 0 and len(sequence) < length:
        x = queue.pop(-1)
        right_most = x.endswith("r")
        if right_most:
            count_right_most_node += 1
        if x.startswith("X"):
            p = np.asarray([a * a, a * (1 - a), a * (1 - a), (1 - a) * (1 - a)])
            if not right_most:
                p = np.concatenate([p * (1 - e), np.asarray([e])])
            choice = random_state.choice(range(len(p)), p=p)
            # 深さ優先探索のため，extendする順番が左右逆であることに注意
            if choice == 0:
                queue.extend([("Xr" if right_most else "X"), "X"])
            elif choice == 1:
                queue.extend([("Yr" if right_most else "Y"), "X"])
            elif choice == 2:
                queue.extend([("Xr" if right_most else "X"), "Y"])
            elif choice == 3:
                queue.extend([("Yr" if right_most else "Y"), "Y"])
            else:
                sequence.append(0)
        else:
            p = np.asarray([(1 - a) * (1 - a), a * (1 - a), a * (1 - a), a * a])
            if not right_most:
                p = np.concatenate([p * (1 - e), np.asarray([e])])
            choice = random_state.choice(range(len(p)), p=p)
            # 深さ優先探索のため，extendする順番が左右逆であることに注意
            if choice == 0:
                queue.extend([("Xr" if right_most else "X"), "X"])
            elif choice == 1:
                queue.extend([("Yr" if right_most else "Y"), "X"])
            elif choice == 2:
                queue.extend([("Xr" if right_most else "X"), "Y"])
            elif choice == 3:
                queue.extend([("Yr" if right_most else "Y"), "Y"])
            else:
                sequence.append(1)
    print(f"Met {count_right_most_node} right-most nodes.")
    return sequence


def mutual_information(
    sequence: Sequence[int],
    distance: int
) -> float:
    cooccur_count = Counter(
        (sequence[idx], sequence[idx + distance])
        for idx in range(len(sequence) - distance)
    )
    x_count = Counter(
        sequence[idx]
        for idx in range(len(sequence) - distance)
    )
    y_count = Counter(
        sequence[idx + distance]
        for idx in range(len(sequence) - distance)
    )
    normalizer = len(sequence) - distance  # == sum(cooccur_count.values())
    log2_normalizer = np.log2(normalizer)
    mi = sum(
        sorted(  # 数値計算の安定性のため気休め程度に
            n * (
                np.log2(n)
                - np.log2(x_count[x])
                - np.log2(y_count[y])
            )
            for (x, y), n in cooccur_count.most_common()
            if n > 0
        )
    ) / normalizer + log2_normalizer
    return mi


def calc_mi_data_over_distances(
    sequence: Sequence[int],
    distances: Sequence[int],
) -> Sequence[float]:
    return [mutual_information(sequence, d) for d in distances]


def plot(
    mutual_infos: Dict[Tuple[float, float], Sequence[float]],
    distances: Dict[Tuple[float, float], Sequence[int]],
    img_dir: pathlib.Path,
    label_format: str = "$(\\alpha, \\epsilon)={}$"
) -> None:
    fig, ax = plt.subplots(
        1,
        1,
        sharex=True,
        sharey=True,
    )
    for i, key in enumerate(distances.keys()):
        x, y = zip(*((x, y) for x, y in zip(distances[key], mutual_infos[key]) if x > 0 and y > 0))
        ax.scatter(
            x,
            y,
            label=label_format.format(key),
            marker=dict(enumerate(["o", "*", "h", "x", "D", "s"]))[i % 6],  # type: ignore
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Mutual Information")
    ax.set_ylim(10 ** -6, 10 ** 0)
    ax.legend()
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / "mi_dist.png", bbox_inches="tight")


def get_params(params: Sequence[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="")
    args = parser.parse_args(params)

    with open(args.config, "r") as fileobj:
        yamlobj: Dict[str, Any] = yaml.safe_load(fileobj)
    for k, v in yamlobj.items():
        setattr(args, k, v)

    return args


def main(argv: Sequence[str]):
    print("Parsing Arguments...")
    opts = get_params(argv)
    if isinstance(opts.a, float):
        a_list: List[float] = [opts.a] 
    else:
        a_list: List[float] = opts.a
        assert isinstance(a_list, list)
        assert all(isinstance(x, float) for x in a_list)
    if isinstance(opts.e, float):
        e_list: List[float] = [opts.e] 
    else:
        e_list: List[float] = opts.e
        assert isinstance(e_list, list)
        assert all(isinstance(x, float) for x in e_list)
    ae_to_distances: Dict[Tuple[float, float], Sequence[int]] = dict()
    ae_to_mutual_infos: Dict[Tuple[float, float], Sequence[float]] = dict()
    for a in a_list:
        for e in e_list:
            print("Generating a Sequence")
            sequence = generate_sequence(
                a=a,
                e=e,
                length=opts.length
            )
            print("Calculating Mutual Information...")
            distances = list(range(1, opts.max_distance))
            mutual_infos = calc_mi_data_over_distances(sequence, distances)
            ae_to_distances[a, e] = distances
            ae_to_mutual_infos[a, e] = mutual_infos
    print("Plotting a Figure...")
    plot(ae_to_mutual_infos, ae_to_distances, pathlib.Path(opts.img_dir))
    print("End")


if __name__ == "__main__":
    main(sys.argv[1:])
