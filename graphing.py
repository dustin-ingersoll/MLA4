import matplotlib.pyplot as plt
import numpy as np


def graph(data, title, conv_iter=None, filename=None):

    # fig.set_size_inches(15, 7)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Max V')
    # ax1.set_xlabel(x1_label)
    # ax1.set_ylabel(y1_label)
    # ax2.set_title("Runtime")
    # ax2.set_xlabel(x2_label)
    # ax2.set_ylabel(y2_label)

    plt.plot(
        data[0:, 0],
        data[0:, 1],
        color='steelblue',
    )
    if conv_iter:
        plt.axvline(x=data[conv_iter, 0], color='k', dashes=(5, 2))

    ax2 = plt.twinx()
    ax2.plot(
        data[0:, 0],
        data[0:, 2],
        color="yellowgreen",
        dashes=(5, 2),
        marker="o"
    )

    ax2.set_ylabel("Error", color="yellowgreen")
    ax2.tick_params(axis='y', colors='yellowgreen')
    ax2.grid(None)

    # ax2.plot(
    #     data[0:, 0],
    #     data[0:, 3],
    #     color="darkorange"
    # )
    # if conv_iter:
    #     ax2.axvline(x=data[conv_iter, 0], color='k', dashes=(5, 2))
    # plt.savefig(filename, bbox="tight")
    if filename:
        plt.savefig("graphs/" + filename)
    plt.show()
