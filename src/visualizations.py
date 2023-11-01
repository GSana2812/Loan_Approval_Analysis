import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def draw_facetgrid(data: pd.DataFrame, col: str, map: str)-> None:

    grid = sns.FacetGrid(data, col=col, height = 2,  aspect=1.6)
    grid.map(sns.countplot, map)
    plt.show()

def draw_countplots(data: pd.DataFrame, col: str, hue: str)-> None:

    plt.figure(figsize=(15, 5))
    sns.countplot(data=data, x = col, hue=hue)
    plt.show()

def draw_distplots(data: pd.DataFrame)-> None:
    index = 0

    n_cols = [x for x in data.columns if data.dtypes[x] != 'object']
    length_ncols_numbers = len(n_cols)
    fig, ax = plt.subplots(nrows=1, ncols=length_ncols_numbers, figsize=(20, 5))

    for x in data.columns:
        if (data.dtypes[x] != 'object'):
            sns.distplot(data[x], ax=ax[index])
            index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


def box_dist_plots(data: pd.DataFrame, col_name: str)-> None:
    plt.figure(figsize=(20, 4))

    plt.subplot(1, 2, 1)
    sns.boxplot(data[col_name])
    plt.title(f'Box Plot of {col_name}')

    plt.subplot(1, 2, 2)
    sns.distplot(data[col_name])
    plt.title(f'Distribution Plot of {col_name}')

    plt.show()