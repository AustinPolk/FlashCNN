import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import ConfusionMatrixDisplay

# classifications
HTN = 'HTN' # higher than normal
NRM = 'NRM' # normal
LTN = 'LTN' # lower than normal

def plot_confusion(variable, ax, model_name):    
    if variable == 'PRCP':
        ax.set_title('Precipitation Classification')    
    if variable == 'TAVG':
        ax.set_title('Temperature Classification')    
    if variable == 'AWND':
        ax.set_title('Wind Speed Classification')
    
    predicted_labels = pd.read_csv(fr"C:\Users\Austin\Desktop\Desktop Projects\flash\evaluation\{model_name}_forecast_{variable}_classes.csv")
    predicted_labels.set_index('FORECAST_START', inplace=True)
    actual_labels = pd.read_csv(fr"C:\Users\Austin\Desktop\Desktop Projects\flash\evaluation\{model_name}_observations_{variable}_classes.csv")
    actual_labels.set_index('FORECAST_START', inplace=True)

    all_predicted_labels = []
    all_actual_labels = []

    for i in range(len(predicted_labels.columns)):
        all_predicted_labels.extend(predicted_labels.iloc[:, i].values)
        all_actual_labels.extend(actual_labels.iloc[:, i].values)

    ConfusionMatrixDisplay.from_predictions(all_actual_labels, all_predicted_labels, ax=ax, colorbar=False, labels=[HTN, NRM, LTN])

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True, help = "model name")
    args = parser.parse_args()
    
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    plot_confusion('PRCP', ax[0], args.model_name)
    plot_confusion('TAVG', ax[1], args.model_name)
    plot_confusion('AWND', ax[2], args.model_name)
    plt.tight_layout()
    plt.show()