import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def parse_filename(filename):
    """ Parse filename into implementation, precision, optimization, and precision mode """
    match = re.match(r'^(.*)_(sp|dp)$', filename)
    if match:
        implementation, precision = match.groups()
    else:
        implementation, precision = None, None
    return implementation, precision


def fix_data(data):
    # Group by 'file' and 'threads' and calculate the median of 'time'
    data = data.groupby(['name', 'size'])['time'].agg(["mean", "median", "std"]).reset_index()


    # parse filenames
    data[['implementation', 'precision']] = data['name'].apply(
        lambda x: pd.Series(parse_filename(x))
    )

    return data

df = pd.read_csv("data/combined.csv")
df = fix_data(df)
#pivot_mean = df_filtered.pivot(index="implementation", columns="precision", values="median")

only_sp = df[df["precision"] == "sp"].pivot(index="implementation", columns="size", values="median")
only_dp = df[df["precision"] == "dp"].pivot(index="implementation", columns="size", values="median")

# Create subplots: one for each dataframe
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 8))

def plot_grouped_bar(ax, df, title):
    x = np.arange(len(df.index))  # label locations
    bar_width = 0.8              # width of the bars
    

    # Plot bars for each column
    bar0  = ax[0].bar(x, df[1000], bar_width)
    bar1 = ax[1].bar(x, df[10000], bar_width)
    
    # Add annotations for the first set of bars
    for bar in bar0:
        height = bar.get_height()
        ax[0].text(
            bar.get_x() + bar.get_width() / 2, height, f'{height:.0f}',
            ha='center', va='bottom', fontsize=10, rotation=0
        )
        
    # Add annotations for the second set of bars
    for bar in bar1:
        height = bar.get_height()
        ax[1].text(
            bar.get_x() + bar.get_width() / 2, height, f'{height:.0f}',
            ha='center', va='bottom', fontsize=10, rotation=0
        )
    
    # Customize the plot
    ax[0].set_xlabel("Implementation")
    ax[0].set_ylabel("Time in ms")
    ax[0].set_title(f"{title} Resolution: 1000 Iterations: 1000")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(df.index, rotation=45, ha="right")
    ax[0].grid(True, linestyle="--", alpha=0.6)

    ax[1].set_xlabel("Implementation")
    ax[1].set_ylabel("Time in ms")
    ax[1].set_title(f"{title} Resolution: 10000 Iterations: 10000")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(df.index, rotation=45, ha="right")
    ax[1].grid(True, linestyle="--", alpha=0.6)

# Plot the first dataframe in the first subplot
plot_grouped_bar(axes[0], only_sp, "Single Precision")

# Plot the second dataframe in the second subplot
plot_grouped_bar(axes[1], only_dp, "Double Precision")

plt.tight_layout()
plt.savefig("gpus", bbox_inches=0, pad_inches=0, dpi=1200)


