import matplotlib
import matplotlib.pyplot as plt

from ._plot_utils import configure_paper_font

configure_paper_font()
matplotlib.rcParams["font.size"] = 14
plt.rcParams.update({
    "figure.dpi": 150,
    "lines.linewidth": 2,
    "lines.markersize": 8,
    "savefig.dpi": 500,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.labelsize": 30,
    "xtick.labelsize": 26,
    "ytick.labelsize": 26,
    "legend.fontsize": 24,
    "legend.title_fontsize": 26,
    "axes.titlesize": 30,
})
