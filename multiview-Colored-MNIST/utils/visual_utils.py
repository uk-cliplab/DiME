import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def display_mult_images(images, rows, cols, save_fig = False, name = 'figure.pdf'):
       
    figure, ax = plt.subplots(rows,cols,figsize=(4, 4))  # array of axes
    ax = ax.flatten()

    for idx, img in enumerate(images):  # images is a list
            img = np.clip(img, 0, 1)
            ax[idx].imshow(img)
            ax[idx].set_axis_off()
    plt.subplots_adjust(wspace=0.0, hspace=0.1)
#     plt.tight_layout()
    plt.show()
    if save_fig:
        path = './figures/' + name
        figure.savefig(path, dpi=1200,bbox_inches='tight')


def display_mult_figures(images,fig,outer, rows, cols, title = None):
    ## Used to be display_walkings
    inner = gridspec.GridSpecFromSubplotSpec(rows, cols,subplot_spec=outer, wspace=0.1, hspace=0.1)
    if title:
        ax = plt.Subplot(fig, outer)
        ax.set_title(title)
        ax.axis('off')
        fig.add_subplot(ax)   

    for idx, img in enumerate(images):  # images is a list
            ax = plt.Subplot(fig,inner[idx]) 
            ax.imshow(img)
            ax.set_axis_off()
            fig.add_subplot(ax)

    return fig