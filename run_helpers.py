from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import colors
from statsmodels.stats.multitest import multipletests


def ttest():
    """
    Performs a one sided t-test for model comparison including Holm-Bonferonnis correction and plots p-values on a
    greyscale plot. Note that you first need to create a results.csv file in every results/[model] directory
    containing the f1 scores for each domain you want to evaluate.
    :return: Plot showing p-values for each model and domain.
    """
    models = ['WBASC', 'SBASC', 'SBASC-FL', 'CASC', 'JASen', 'BERT', 'cos_sim_sentence', 'cos_sim']
    headers = ['WB-ASC', 'SB-ASC', 'SB-ASC w/o FL', 'CASC', 'JASen', 'BERT', 'CosSim-Sent', 'CosSim']
    bonferroni_correction = 28

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"

    for domain in ['restaurant-3', 'restaurant-5', 'laptop', 'restaurant-nl', 'supermarket']:
        # for domain in ['restaurant-3']:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f'{domain.capitalize()}', fontsize=16)

        for c, classification in enumerate(["polarity", "aspect"]):
            values = []
            ptable = np.empty([len(models), len(models)])
            pvalues = []
            for model in models:
                df = pd.read_csv(f"results/{model}/results.csv")
                df = df.loc[(df['type'] == classification) & (df['domain'] == domain), ['f1-score']]
                values.append(df['f1-score'].values.tolist())
            print(values)
            for i, model_1 in enumerate(models):
                for j, model_2 in enumerate(models):
                    print(f"{model_1} vs. {model_2}")
                    result = stats.ttest_ind(values[i], values[j], alternative="less")
                    ptable[i, j] = min(result.pvalue * bonferroni_correction, 1)
                    if i != j:
                        pvalues.append(result.pvalue)
                    print(result)
                    print()
            print(ptable)

            adj_ptable = np.reshape(multipletests(pvalues, method='holm')[1], (len(models), len(models) - 1))

            d = adj_ptable.shape[0]
            assert adj_ptable.shape[1] == d - 1
            matrix_new = np.ndarray((d, d + 1), dtype=adj_ptable.dtype)
            matrix_new[:, 0] = 1
            matrix_new[:-1, 1:] = adj_ptable.reshape((d - 1, d))
            matrix_new = matrix_new.reshape(-1)[:-d].reshape(d, d)

            plt.subplots_adjust(bottom=0.2, left=0.2)  # make room for labels

            cmap = plt.cm.gray
            bounds = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1]
            norm = colors.BoundaryNorm(bounds, cmap.N)
            heatmap = ax[c].pcolor(matrix_new, cmap=cmap, norm=norm)

            ax[c].set_title(f"{['sentiment', 'aspect'][c].capitalize()}")

            # Set ticks in center of cells
            ax[c].set_xticks(np.arange(ptable.shape[1]) + 0.5, minor=False)
            ax[c].set_yticks(np.arange(ptable.shape[0]) + 0.5, minor=False)

            # Rotate the xlabels. Set both x and y labels to headers[1:]
            ax[c].set_xticklabels(headers, rotation=90)

            if c == 0:
                ax[c].set_yticklabels(headers)
            else:
                ax[c].set_yticklabels([])

            # Minor ticks
            ax[c].set_xticks(np.arange(ptable.shape[1]), minor=True)
            ax[c].set_yticks(np.arange(ptable.shape[0]), minor=True)

            # Gridlines based on minor ticks
            ax[c].grid(which='minor', color='k', linestyle='-', linewidth=1)
            ax[c].invert_xaxis()
            ax[c].xaxis.set_ticks_position('none')
            ax[c].yaxis.set_ticks_position('none')
            ax[c].set_aspect("equal")

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.84, 0.30, 0.02, 0.5])
        fig.colorbar(heatmap, cax=cbar_ax, format='%.0e')

        plt.show()


def loss_plot():
    """
    Plot different (GCE and focal loss) functions in a graph to show difference for multiple parameter settings
    :return: plot showing different loss functions
    """

    def f_focal(n, t):
        summation = -(1 - t) ** n * np.log(t + 1e-9)
        return summation

    def f_gce(q, t):
        summation = (1 - t ** q) / q
        return summation

    ts = np.linspace(0, 1, 400)
    for n in [0, 0.5, 1, 2, 5]:
        fs = [f_focal(n, t) for t in ts]
        plt.plot(ts, fs, label=f'$\gamma={n}$')

    for n in [0.2, 0.5, 1]:
        fs = [f_gce(n, t) for t in ts]
        plt.plot(ts, fs, linestyle='--')

    plt.margins(x=0, tight=True)
    plt.xlabel('probability of ground truth class')
    plt.ylabel('loss')

    lines = plt.gca().get_lines()

    legend1 = plt.legend(title='Focal loss', ncol=1)
    legend2 = plt.legend([lines[i] for i in [5, 6, 7]], ['$q=0.2$', '$q=0.5$', '$q=1$'], title='GCE loss', ncol=1,
                         bbox_to_anchor=(0.85, 1))

    plt.gca().add_artist(legend1)
    plt.ylim([0, 5])
    plt.show()
