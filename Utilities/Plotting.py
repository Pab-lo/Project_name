
import numpy as np
import matplotlib.pyplot as plt



colours = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:olive', 'tab:cyan', 'tab:brown']


def plot_time_series(time_series_list, start = 0, end = -1, draw_means = False):
    
    plt.figure(figsize=(12, 6))
    
    colour_idx = 0
    
    for time_series in time_series_list:
        plt.plot(range(len(time_series[start:end])), time_series[start:end], linestyle = ':', marker = '.', color = colours[colour_idx])
        if draw_means:
            plt.hlines(np.mean(time_series), 0, len(time_series[start:end]), colors = colours[colour_idx], linestyles=':')
        colour_idx += 1

    plt.xlabel('Time step')
    plt.ylabel('Level')
    plt.grid()
    plt.show()
    
    
def plot_learning_curves(loss_by_epoch):
    
    plt.figure(figsize=(12, 9))
    
    train_loss_list = sorted(loss_by_epoch['Train_loss'].items())
    val_loss_list = sorted(loss_by_epoch['Val_loss'].items())
    
    epochs, train_losses = zip(*train_loss_list)
    _, val_losses = zip(*val_loss_list)

    plt.plot(epochs, train_losses, label = "Train_loss", linestyle = ':', marker = '.', color = colours[0])
    plt.plot(epochs, val_losses, label = "Val_loss", linestyle = ':', marker = '.', color = colours[1])
    plt.ylim(bottom = 0.)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend(loc="best")
    plt.show()
    
    
def plot_compressed_data(Z):
    
    plt.figure(figsize=(9, 9))
    
    plt.plot(Z[:, 0], Z[:, 1], '.', color='firebrick')
    plt.grid()
    plt.show()


def plot_backtest_results(strat_values_dict):
    
    plt.figure(figsize=(12, 9))
    
    for strat_idx in range(len(strat_values_dict)):
        plt.plot(
            range(strat_values_dict[list(strat_values_dict.keys())[strat_idx]].shape[0]),
            strat_values_dict[list(strat_values_dict.keys())[strat_idx]],
            label = list(strat_values_dict.keys())[strat_idx],
            linestyle = ':',
            marker = '',
            color = colours[strat_idx + 1]
        )
    plt.hlines(1., 0, strat_values_dict[list(strat_values_dict.keys())[0]].shape[0], colors = colours[0], linestyles='-')
    plt.ylim(bottom = 0.)
    plt.xlabel('Time')
    plt.ylabel('PnL')
    plt.grid()
    plt.legend(loc="best")
    plt.show()


def plot_spot(spot_array_np, df_index):
    
    plt.figure(figsize=(12, 9))
    
    plt.plot(range(len(spot_array_np)), spot_array_np, label = "Spot", linestyle = ':', marker = '', color = colours[2])
    plt.hlines(spot_array_np[0], 0, spot_array_np.shape[0], colors = colours[0], linestyles=':')
    plt.ylim(bottom = 0.)
    tick_idx = [i*145 for i in range(int(len(df_index)/145) + 1)]
    plt.xticks(
        ticks = tick_idx,
        labels = [str(df_index[j])[:10] for j in tick_idx],
        rotation = 45
    )
    plt.xlabel('Time')
    plt.ylabel('Spot')
    plt.grid()
    plt.legend(loc="best")
    plt.show()

    
def print_model_results_table(all_pairs, all_alphas, all_sharpes, caption, label):
    
    print(r"\begin{table}[h!]")
    print(r"  \centering")
    print(r"  \begin{tabular}{c|ccc|ccc}")
    print(r"  \hline")
    print(r"   Pair & \multicolumn{3}{c}{$\alpha$} & \multicolumn{3}{c}{Ratio of Sharpes} \\")
    print(r"    & Train & Val & Test & Train & Val & Test \\")
    print(r"  \hline")
    print(r"  \hline")
    for pair_idx in range(len(all_pairs)):
        print(f"    {all_pairs[pair_idx]} & ${100.*all_alphas['Train'][pair_idx]:.2f}\%$ & ${100.*all_alphas['Val'][pair_idx]:.2f}\%$ & ${100.*all_alphas['Test'][pair_idx]:.2f}\%$ & ${all_sharpes['Train'][pair_idx]:.2f}$ & ${all_sharpes['Val'][pair_idx]:.2f}$ & ${all_sharpes['Test'][pair_idx]:.2f}$" + r" \\")
    print(r"  \hline")
    print(f"    Mean & ${100.*np.mean(all_alphas['Train']):.2f}\%$ & ${100.*np.mean(all_alphas['Val']):.2f}\%$ & ${100.*np.mean(all_alphas['Test']):.2f}\%$ & ${np.mean(all_sharpes['Train']):.2f}$ & ${np.mean(all_sharpes['Val']):.2f}$ & ${np.mean(all_sharpes['Test']):.2f}$" + r" \\")
    print(r"  \hline")
    print(r"\end{tabular}")
    print(r"  \caption{" + caption + "}")
    print(r"  \label{" + label + "}")
    print(r"\end{table}")
    