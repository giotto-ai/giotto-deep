import argparse
import enum
import string

import benchmark


class GPUs(enum.Enum):
    a100 = enum.auto()
    v100 = enum.auto()
    t4 = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return GPUs[s]
        except KeyError:
            raise ValueError(f"Unknown GPU {s}")

    def fullname(self) -> str:
        if self is GPUs.a100:
            return "nvidia-tesla-a100"
        elif self is GPUs.v100:
            return "nvidia-tesla-v100"
        elif self is GPUs.t4:
            return "nvidia-tesla-t4"
        else:
            raise Exception(f"fullname missing for {self.name}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    parser.add_argument("-i", "--image", required=True, help="Container image")
    parser.add_argument("-b", "--bucket", required=True, help="Storage bucket")
    parser.add_argument("-s", "--ksa", required=True, help="Kubernetes Service Account")
    parser.add_argument(
        "-d", "--dir", default="", help="Subdirectory in the bucket to store/read data"
    )

    parser_run = subparsers.add_parser("run", help="Generate pods to run benchmarks")
    parser_run.set_defaults(func=main_run)
    parser_run.add_argument(
        "-c", "--gpu-count", required=True, type=int, nargs="*", help="GPU count"
    )
    parser_run.add_argument(
        "-g",
        "--gpu-model",
        required=True,
        type=GPUs.from_string,
        choices=[x for x in GPUs],
        help="GPU model",
    )
    parser_run.add_argument(
        "-m",
        "--model",
        required=True,
        type=benchmark.Models.from_string,
        choices=[x for x in benchmark.Models],
        help="Model",
    )
    parser_run.add_argument(
        "-p",
        "--parallel",
        required=True,
        type=benchmark.Parallelism.from_string,
        choices=[x for x in benchmark.Parallelism],
        nargs="+",
        help="Parallelism type(s)",
    )
    parser_run.add_argument(
        "-z",
        "--batch-size",
        type=int,
        nargs=2,
        choices=benchmark.BATCH_SIZE_VALUES,
        metavar=("MINVAL", "MAXVAL"),
        help=f"Batch size range; possible values are {benchmark.BATCH_SIZE_VALUES}",
    )

    parser_plot = subparsers.add_parser(
        "plot", help="Generate pods to plot benchmarks results"
    )
    parser_plot.set_defaults(func=main_plot)

    args = parser.parse_args()
    args.func(args)


def main_run(args):
    values = {
        "image": args.image,
        "bucket": args.bucket,
        "ksa": args.ksa,
        "subdir": args.dir,
        "gpu_count": 0,
        "gpu_model": args.gpu_model.fullname(),
        "gpu_name": str(args.gpu_model),
        "model": str(args.model),
        "parallel": ", ".join([f'"{x}"' for x in args.parallel]),
        "batch_size": "",
    }

    with open("pod-template-run.yml", "r") as fp:
        ymlt = string.Template(fp.read())

    if args.batch_size is not None:
        args.batch_size.insert(0, "-b")
        values["batch_size"] = ", " + ", ".join([f'"{x}"' for x in args.batch_size])

    for gpu_count in args.gpu_count:
        values["gpu_count"] = gpu_count
        ymlv = ymlt.substitute(values)
        filename = f"run-{args.model}-{args.gpu_model}-{gpu_count}.yml"
        with open(filename, "w") as fp:
            fp.write(ymlv)
        print(f"kubectl apply -f {filename}")


def main_plot(args):
    values = {
        "image": args.image,
        "bucket": args.bucket,
        "ksa": args.ksa,
        "subdir": args.dir,
    }

    with open("pod-template-plot.yml", "r") as fp:
        ymlt = string.Template(fp.read())
    ymlv = ymlt.substitute(values)
    filename = "plot.yml"
    with open(filename, "w") as fp:
        fp.write(ymlv)
    print(f"kubectl apply -f {filename}")


if __name__ == "__main__":
    main()
