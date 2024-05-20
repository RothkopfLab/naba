import matplotlib as mpl
import seaborn as sns
import seaborn.objects as so

rcparams = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "legend.frameon": False,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 6,
    "grid.linestyle": "",
    "axes.labelsize": 8
}


def update():
    # update matplotlib theme
    mpl.rcParams.update(rcparams)

    # udpate seaborn theme
    theme = {**sns.axes_style("ticks"), **rcparams}
    so.Plot.config.theme.update(theme)


var_name_map = {"mu_0": r"$\mu_0$", "sigma_0": r"$\sigma_0$", "sigma_r": r"$\sigma_r$", "sigma": r"$\sigma$",
                "beta": r"$\beta$", "alpha": r"$\alpha$", "AsymmetricQuadratic": "AQC",
                "QuadraticCostQuadraticEffort": "QCQE",
                'alpha / beta': r"$\alpha | \beta$",
                "effort": "effort"}
