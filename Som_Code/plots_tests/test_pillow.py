import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from tins_dots.Plots_Generator import PlotsGenerator

matplotlib.use('TkAgg')  # Set the backend to TkAgg

def generate_barcode(colors, width=400, height=100):
    color_indices = np.arange(len(colors))
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow([color_indices], cmap=cmap, aspect="auto", extent=(0, width, 0, height))
    ax.xaxis.set_visible(True)
    plt.close()
    return fig, ax, color_indices


def generate_barcode_grid(color_sequences, n_rows=2, n_cols=1, width=400, height=100):
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    for i in range(n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = i % n_cols
        colors = color_sequences[i % len(color_sequences)]
        _, ax, color_indices = generate_barcode(colors, width=width, height=height)
        ax.xaxis.set_visible(True)
        axs[row_idx, col_idx].axis('off')
        axs[row_idx, col_idx].imshow(color_indices.reshape(1, -1), cmap=ax.images[0].cmap, aspect='auto')
    plt.subplots_adjust(hspace=0, wspace=0)
    return fig, axs


# Define the color sequences for each barcode
color_sequences = [
    [(1, 1, 1), (0, 0, 0)],
    [(1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0)],
    [(0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1)],
    [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0, 0, 0)]
]

# Generate a grid of barcode plots
#fig, axs = generate_barcode_grid(color_sequences, n_rows=4, n_cols=1)
#plt.show()

barcodes_array = []
for seq in color_sequences:
    figure_data_tuple = PlotsGenerator.generateColorSequenceForTrialMatplotlib(seq)
    barcodes_array.append(figure_data_tuple)


PlotsGenerator.generateGridWithColorSequences(barcodes_array, n_rows=len(barcodes_array), n_cols=1, max_trial_len=5)
plt.suptitle("Plot1")

fig, axs = PlotsGenerator.generateGridWithColorSequences(barcodes_array, n_rows=len(barcodes_array), n_cols=1, max_trial_len=5)
plt.suptitle("Plot2")


plt.show()
