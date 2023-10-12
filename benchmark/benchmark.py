import argparse
import csv
import dataclasses
import datetime
import enum
import math
import matplotlib.pyplot as plt
import multiprocessing as pmp
import pathlib
import sys
import torch
import typing

from gdeep.trainer.trainer import ParallelismType
from gdeep.utility_examples.fsdp import ShardingStrategyEx

sys.path.append("../examples")
from examples import orbit_5k_big


class Parallelism(enum.Enum):
    none = enum.auto()
    fsdp_full_shard = enum.auto()
    fsdp_shard_grad_op = enum.auto()
    fsdp_no_shard = enum.auto()
    pipeline = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return Parallelism[s]
        except KeyError:
            raise ValueError()

    def to_text(self):
        if self is Parallelism.none:
            return "None"
        elif self is Parallelism.fsdp_full_shard:
            return "FSDP Full Shard"
        elif self is Parallelism.fsdp_shard_grad_op:
            return "FSDP Shard Grad Op"
        elif self is Parallelism.fsdp_no_shard:
            return "FSDP No Shard"
        elif self is Parallelism.pipeline:
            return "Pipeline"
        else:
            return "?"

    def to_pt(self) -> ParallelismType:
        if self is Parallelism.none:
            return ParallelismType._NONE
        elif self in (Parallelism.fsdp_full_shard, Parallelism.fsdp_shard_grad_op, Parallelism.fsdp_no_shard):
            return ParallelismType.FSDP
        elif self is Parallelism.pipeline:
            return ParallelismType.PIPELINE
        else:
            raise ValueError(f"Unknown {self}")

    def to_ss(self) -> ShardingStrategyEx:
        if self is Parallelism.fsdp_full_shard:
            return ShardingStrategyEx.FULL_SHARD
        elif self is Parallelism.fsdp_shard_grad_op:
            return ShardingStrategyEx.SHARD_GRAD_OP
        elif self is Parallelism.fsdp_no_shard:
            return ShardingStrategyEx.NO_SHARD
        else:
            return ShardingStrategyEx.SHARD_GRAD_OP

    def colour(self) -> str:
        # https://matplotlib.org/stable/gallery/color/named_colors.html#css-colors
        if self is Parallelism.none:
            return "blue"
        elif self is Parallelism.fsdp_full_shard:
            return "darkorange"
        elif self is Parallelism.fsdp_shard_grad_op:
            return "green"
        elif self is Parallelism.fsdp_no_shard:
            return "magenta"
        elif self is Parallelism.pipeline:
            return "red"
        else:
            return "black"


class Models(enum.Enum):
    none = enum.auto()
    orbit5k = enum.auto()
    orbit5kbig = enum.auto()

    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s: str):
        try:
            return Models[s]
        except KeyError:
            raise ValueError()


class RunData:

    CSV_FIELDS = ["start_time", "end_time", "run_time", "model", "parallel", "epochs", "batch_size", "loss", "accuracy", "gpu_count", "gpu_model"]

    def __init__(self, start_time: datetime.datetime, end_time: datetime.datetime, model: Models, parallel: Parallelism,
                 epochs: int, batch_size: int, loss: float, accuracy: float, gpu_count: int, gpu_model: str):
        self.start_time = start_time
        self.end_time = end_time
        self.run_time = end_time - start_time
        self.model = model
        self.parallel = parallel
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.accuracy = accuracy
        self.gpu_count = gpu_count
        self.gpu_model = gpu_model

    @classmethod
    def load(cls, start_time: str, end_time: str, model: str, parallel: str, epochs: str, batch_size: str, loss: str, accuracy: str, gpu_count: str, gpu_model: str):
        return cls(
            datetime.datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f"),
            datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S.%f"),
            Models.from_string(model),
            Parallelism.from_string(parallel),
            int(epochs),
            int(batch_size),
            float(loss),
            float(accuracy),
            int(gpu_count),
            gpu_model,
        )

    def rt_mms(self):
        return self.run_time.total_seconds()

    def write_row(self, writer: csv.DictWriter):
        values = {x: None for x in RunData.CSV_FIELDS}
        values[RunData.CSV_FIELDS[0]] = str(self.start_time)
        values[RunData.CSV_FIELDS[1]] = str(self.end_time)
        values[RunData.CSV_FIELDS[2]] = str(self.rt_mms())
        values[RunData.CSV_FIELDS[3]] = self.model
        values[RunData.CSV_FIELDS[4]] = self.parallel
        values[RunData.CSV_FIELDS[5]] = self.epochs
        values[RunData.CSV_FIELDS[6]] = self.batch_size
        values[RunData.CSV_FIELDS[7]] = self.loss
        values[RunData.CSV_FIELDS[8]] = self.accuracy
        values[RunData.CSV_FIELDS[9]] = self.gpu_count
        values[RunData.CSV_FIELDS[10]] = self.gpu_model
        writer.writerow(values)

    @staticmethod
    def write_header(fp) -> csv.DictWriter:
        writer = csv.DictWriter(fp, dialect="unix", fieldnames=RunData.CSV_FIELDS)
        writer.writeheader()
        return writer

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.start_time}, {self.end_time}, {self.model}"
            f", {self.parallel}, {self.epochs}, {self.batch_size}, {self.gpu_model}, {self.gpu_count}, {self.accuracy})"
        )

    def same(self, o: "RunData") -> bool:
        return all([
            self.model == o.model,
            self.parallel == o.parallel,
            self.batch_size == o.batch_size,
            self.gpu_count == o.gpu_count,
            self.gpu_model == o.gpu_model,
        ])

    def gt(self, o: "RunData") -> bool:
        return self.end_time > o.end_time


@dataclasses.dataclass
class RunResult:
    start_time: datetime.datetime
    end_time: datetime.datetime
    loss: float
    accuracy: float


BATCH_SIZE_VALUES = (1, 2, 4, 8, 16, 32, 64)

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/marker_reference.html
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
PLOT_LINES = [
    "solid",
    "dashed",
    "dotted",
    "dashdot",
]
PLOT_MARKERS = [
    "x",
    "+",
    ".",
    "*",
    "H",
    "s",
]
PLOT_IMG_WIDTH = 10
PLOT_IMG_HEIGHT = 5
PLOT_IMG_MARGIN_LEFT = 0.1


def device_name(model: str, count: int) -> str:
    return f"{count} {model}"


def device_filename(model: str, count: int) -> str:
    return f"{model} {count}".lower().replace(" ", "-")


def nofn(args):
    return 0, 0


def wrap(fn, args, q):
    start_time = datetime.datetime.now()
    loss, accuracy = fn(args)
    end_time = datetime.datetime.now()
    q.put(RunResult(start_time, end_time, loss, accuracy))


def identity(string):
    return string


def run_training(model: Models, parallel: Parallelism, batch_size: int, epochs: int, device_name: str, device_count: int, device_model: str) -> RunData:

    args = argparse.ArgumentParser()
    args.register("type", None, identity)
    fn = nofn

    if model is Models.none:
        pass
    elif model is Models.orbit5k:
        args.batch_size = batch_size
        args.n_epochs = epochs
        args.parallel = parallel.to_pt()
        args.layer_cls = False
        args.big_model = False
        args.sharding = parallel.to_ss()
        fn = orbit_5k_big.main
    elif model is Models.orbit5kbig:
        batch_size = 4
        args.batch_size = batch_size
        args.n_epochs = epochs
        args.parallel = parallel.to_pt()
        args.layer_cls = False
        args.big_model = True
        args.sharding = parallel.to_ss()
        fn = orbit_5k_big.main

    sys.stdout.write(f"BENCHMARK RUNNING ON {device_name}... parallelism {parallel} with batch size {batch_size}...\n")
    sys.stdout.flush()

    # Spawn a new python interpreter to ensure a nice release of resources
    mp = pmp.get_context('spawn')
    rq = mp.SimpleQueue()
    process = mp.Process(target=wrap, args=(fn, args, rq), daemon=False)
    process.start()
    process.join()
    if process.exitcode != 0:
        raise Exception(f"Train process exited with exitcode {process.exitcode}")
    r = rq.get()

    #if model is not Models.none and parallel is Parallelism.pipeline:
    #    torch.distributed.rpc.shutdown()

    return RunData(r.start_time, r.end_time, model, parallel, epochs, batch_size, r.loss, r.accuracy, device_count, device_model)


def uniq(data: typing.List[RunData]):
    data2 = []
    idx = 0
    # parse every element in the list (unless those removed during the process)
    while idx < len(data):
        jdx = idx + 1
        keep = data[idx] # set current data as kept
        # parse every further element in the list (unless those removed during the process)
        while jdx < len(data):
            # if the currently kept element and the current element are of the same "class" ...
            if data[jdx].same(keep):
                # ... compare if the current element is greater than the kept one ...
                if data[jdx].gt(keep):
                    # ... and keep and remove the current element if it is greater
                    keep = data.pop(jdx)
                else:
                    # ... or only remove the current element if it is not greater
                    del data[jdx]
            else:
                jdx += 1
        data2.append(keep)
        idx += 1
    return data2


def plot_training(run_data: typing.List[RunData], imgfile: pathlib.Path, dev_name: str):
    plt_data = {}
    for d in run_data:
        if d.parallel not in plt_data:
            plt_data[d.parallel] = {}
        plt_data[d.parallel][d.batch_size] = d.rt_mms()

    fig = plt.figure(figsize=(PLOT_IMG_WIDTH, PLOT_IMG_HEIGHT))
    ax = fig.add_subplot(1, 1, 1)
    plots = []
    legends = []
    for parallel, v in plt_data.items():
        p, = ax.plot(v.keys(), v.values(), linestyle=PLOT_LINES[0], linewidth=1.5, color=parallel.colour(), marker=PLOT_MARKERS[0])
        plots.append(p)
        legends.append(parallel.to_text())

    ax.legend(plots, legends, loc="upper right")
    ax.set_title(f"{d.model} -- Run time per batch size -- {dev_name}")
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Run time [s]")
    fig.subplots_adjust(left=PLOT_IMG_MARGIN_LEFT)
    plt.savefig(str(imgfile))


def plot_csv(run_data: typing.List[RunData], img_dir: pathlib.Path, now: datetime.datetime):
    template = f"plot-{now.strftime('%Y-%m-%d-%H-%M-%S')}"
    data = {}
    for d in run_data:
        if d.model not in data:
            data[d.model] = {}
        if d.gpu_model not in data[d.model]:
            data[d.model][d.gpu_model] = {}
        if d.gpu_count not in data[d.model][d.gpu_model]:
            data[d.model][d.gpu_model][d.gpu_count] = {}
        if d.parallel not in data[d.model][d.gpu_model][d.gpu_count]:
            data[d.model][d.gpu_model][d.gpu_count][d.parallel] = {}
        data[d.model][d.gpu_model][d.gpu_count][d.parallel][d.batch_size] = d.rt_mms()

    # Plot parallelism for model/gpu-model/gpu-count
    for model, v_model in data.items():
        for gpu_model, v_gpu_model in v_model.items():
            for gpu_n, v_gpu_n in v_gpu_model.items():
                fig = plt.figure(figsize=(PLOT_IMG_WIDTH, PLOT_IMG_HEIGHT))
                ax = fig.add_subplot(1, 1, 1)
                plots = []
                legends = []
                for parallel, values in v_gpu_n.items():
                    p, = ax.plot(values.keys(), values.values(),
                                 linestyle=PLOT_LINES[0], linewidth=1.5,
                                 color=parallel.colour(), marker=PLOT_MARKERS[0])
                    plots.append(p)
                    legends.append(parallel.to_text())
                ax.legend(plots, legends, loc="upper right")
                ax.set_title(f"{model} -- Run time per batch size -- {gpu_n} {gpu_model}")
                ax.set_xlabel("Batch size")
                ax.set_ylabel("Run time [s]")
                fig.subplots_adjust(left=PLOT_IMG_MARGIN_LEFT)
                img_name = template + f"-{model}-{device_filename(gpu_model, gpu_n)}.png"
                plt.savefig(str(img_dir.joinpath(img_name)))

    # Plot parallelism for model/gpu-model/gpu-count
    for model, v_model in data.items():
        fig = plt.figure(figsize=(PLOT_IMG_WIDTH, PLOT_IMG_HEIGHT))
        ax = fig.add_subplot(1, 1, 1)
        plots = []
        legends = []
        for k, (gpu_model, v_gpu_model) in enumerate(v_model.items()):
            for j, (gpu_n, v_gpu_n) in enumerate(v_gpu_model.items()):
                for parallel, values in v_gpu_n.items():
                    p, = ax.plot(values.keys(), values.values(),
                                 linestyle=PLOT_LINES[j], linewidth=1.5,
                                 color=parallel.colour(), marker=PLOT_MARKERS[k])
                    plots.append(p)
                    legends.append(f"{parallel.to_text()}, {gpu_n} {gpu_model}")
        ax.legend(plots, legends, loc="upper right")
        ax.set_title(f"{model} -- Run time per batch size")
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Run time [s]")
        fig.subplots_adjust(left=PLOT_IMG_MARGIN_LEFT)
        img_name = template + f"-{model}.png"
        plt.savefig(str(img_dir.joinpath(img_name)))


def main_plot(args):
    data = []
    if args.files is not None:
        files = args.files
    else:
        files = pathlib.Path(args.csvdir).glob("*.csv")
    for csvfile in files:
        with open(csvfile, "r") as csvfp:
            skip = True
            reader = csv.reader(csvfp, dialect="unix")
            for row in reader:
                if skip:
                    skip = False
                else:
                    data.append(RunData.load(row[0], row[1], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]))
    data = uniq(data)
    plot_csv(data, pathlib.Path(csvfile).parent, datetime.datetime.now())


def main_run(args):
    # Get GPU data
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        dev_count = torch.cuda.device_count()
        dev_model = torch.cuda.get_device_name(0)
        #dev_details = f"{device_name}, mem {torch.cuda.get_device_properties(0).total_memory/1e9:3.1f} GB, CUDNN {torch.backends.cudnn.version()}"
    else:
        dev_count = 0
        dev_model = "cpu"
        #dev_details = ""
    dev_name = device_name(dev_model, dev_count)
    dev_filename = device_filename(dev_model, dev_count)

    # Parse arguments
    min_exp_batch_size = int(math.log2(min(args.batch_size)))
    max_exp_batch_size = int(math.log2(max(args.batch_size)))
    filename_template = f"benchmark-{args.model}-{dev_filename}-{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

    # Setup result files
    dir = pathlib.Path(args.csvdir)
    dir.mkdir(exist_ok=True)
    csvfile = dir.joinpath(f"{filename_template}.csv")
    pltfile = dir.joinpath(f"{filename_template}.png")
    sys.stdout.write(f"BENCHMARK RUNNING ON {dev_name}...\n")
    sys.stdout.flush()
    csvfp = open(csvfile, "w", newline="")
    csvw = RunData.write_header(csvfp)

    # Run trainings
    data = []
    for parallel in args.parallel:
        for exp_batch_size in range(min_exp_batch_size, max_exp_batch_size+1):
            try:
                run_data = run_training(args.model, parallel, int(math.pow(2, exp_batch_size)), args.n_epochs, dev_name, dev_count, dev_model)
                run_data.write_row(csvw)
                data.append(run_data)
            except Exception as e:
                sys.stdout.write(f"BENCHMARK RUN FAILED: {e}\n")
                sys.stdout.flush()

    csvfp.close()

    # Plot trainings data
    if len(data):
        plot_training(data, pltfile, dev_name)

    # End of script
    sys.stdout.write(f"BENCHMARK DONE.\nLOG FILE IS {csvfile}\nPLOT FILE IS {pltfile}\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Giotto-deep benchmark tool")

    subparsers = parser.add_subparsers(required=True)

    parser_run = subparsers.add_parser("run", help="Run a benchmark")
    parser_run.set_defaults(func=main_run)
    parser_run.add_argument("-m", "--model",
                            required=True,
                            type=Models.from_string,
                            choices=[x for x in Models],
                            help="Model to run")
    parser_run.add_argument("-p", "--parallel",
                            type=Parallelism.from_string,
                            choices=[x for x in Parallelism],
                            nargs="+",
                            default=Parallelism.none,
                            help="Parallelism type(s); default is %(default)s")
    parser_run.add_argument("-b", "--batch-size",
                            type=int,
                            nargs=2,
                            choices=BATCH_SIZE_VALUES,
                            default=(4, 4),
                            metavar=("MINVAL", "MAXVAL"),
                            help=f"Batch size range; possible values are {BATCH_SIZE_VALUES}; default is %(default)s")
    parser_run.add_argument("-n", "--n-epochs",
                            type=int,
                            default=3,
                            metavar="N",
                            help="Number of epochs; default is %(default)s")
    parser_run.add_argument("-d", "--csvdir",
                            required=True,
                            help="CSV files directory")

    parser_plot = subparsers.add_parser("plot", help="Plot benchmark results")
    parser_plot.set_defaults(func=main_plot)
    grp1 = parser_plot.add_mutually_exclusive_group(required=True)
    grp1.add_argument("-d", "--csvdir",
                             help="CSV files directory; use all files in the directory")
    grp1.add_argument("-f", "--files",
                             nargs="*",
                             help="CSV files to plot; use these specific files; require full path")

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
