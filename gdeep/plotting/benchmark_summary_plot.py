import os
from PIL import Image
import matplotlib.pyplot as plt
from typing import List
from attrdict import AttrDict
from pathlib import Path


def save_run_summary(
    train_losses: List[float],
    val_losses: List[float],
    train_accuracies: List[float],
    val_accuracies: List[float],
    config_run: AttrDict,
    config: AttrDict,
    run: int
    ) -> None:


    # Generate summary text
    max_val_acc = "{:2.2%}".format(max(val_accuracies) / 100)
    meta_data = f"max_val_acc: {max_val_acc}\n"
    for k, v in config_run.items():
        if k != 'optimizer':
            meta_data += k + ": " + str(v) + "\n"
    meta_data = meta_data[:-1]

    # Generate filename
    filename = f"max_val_acc_{max_val_acc}"
    for k, v in config_run.items():
        if k != 'optimizer':
            filename += k + "_" + str(v) + "_"
    filename = filename[:-1]

    plt.plot(train_losses, label='train_loss')
    plt.plot([x for x in val_losses], label='val_loss')
    plt.legend()
    plt.title("Losses " + config.dataset_name +
        " extended persistence features only")

    plt.savefig("losses.png")
    plt.show()

    # plot accuracies
    plt.plot(train_accuracies, label='train_acc')
    plt.plot(val_accuracies, label='val_acc')
    plt.legend()
    plt.title("Accuracies " + config.dataset_name +
        " extended persistence features only")

    plt.savefig("accuracies.png")
    plt.show()

  
    # plot metadata
    plt.text(0.2, 0.1,
             meta_data,
             fontsize='x-large')
    plt.axis('off')

    plt.savefig("metadata.png")
    plt.show()

    image_files = ['losses.png', 'accuracies.png', 'metadata.png']
    images = [Image.open(x) for x in image_files]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    max_val_acc = '{:2.2%}'.format(max(val_accuracies)/100)

    benchmark_dir = os.path.join(
        "Benchmark_PersFormer_graph",
        f"""{config.dataset_name}_Benchmark""")
    Path(benchmark_dir)\
    .mkdir(parents=True, exist_ok=True)

    new_im.save(
        os.path.join(
            benchmark_dir,
            f"{filename}_run_{run}.png"
        )
    )

    # delete temporary images
    for image in image_files:
        os.remove(image)