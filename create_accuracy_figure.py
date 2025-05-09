import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# classifications
HTN = 'HTN' # higher than normal
NRM = 'NRM' # normal
LTN = 'LTN' # lower than normal

def get_forecast_period_pct(actual, period, classification):
    actual_period = list(actual.iloc[:, period].values)
    return actual_period.count(classification) / len(actual_period)

def get_forecast_period_accuracy(predicted, actual, period):
    correctly_classified = 0
    total = 0
    
    predicted_period = predicted.iloc[:, period].values
    actual_period = actual.iloc[:, period].values
    
    for predicted, actual in zip(predicted_period, actual_period):
        if predicted == actual:
            correctly_classified += 1
        total += 1

    return correctly_classified / total

def get_accuracies(variable, model_name):    
    predicted_labels = pd.read_csv(fr"C:\Users\Austin\Desktop\Desktop Projects\flash\evaluation\{model_name}_forecast_{variable}_classes.csv")
    predicted_labels.set_index('FORECAST_START', inplace=True)
    actual_labels = pd.read_csv(fr"C:\Users\Austin\Desktop\Desktop Projects\flash\evaluation\{model_name}_observations_{variable}_classes.csv")
    actual_labels.set_index('FORECAST_START', inplace=True)
    
    accuracy_by_forecast_period = {}
    class_occurrences = {
        LTN: {},
        NRM: {},
        HTN: {},
    }
    
    for i in range(len(predicted_labels.columns)):
        accuracy_by_forecast_period[i] = get_forecast_period_accuracy(predicted_labels, actual_labels, i)
        for classification in [LTN, NRM, HTN]:
            class_occurrences[classification][i] = get_forecast_period_pct(actual_labels, i, classification)

    return list(x+1 for x in accuracy_by_forecast_period.keys()), list(accuracy_by_forecast_period.values()), class_occurrences

def display_accuracies(v, ax, model_name):
    x, y, class_occurrences = get_accuracies(v, model_name)

    print(f'{v} accuracy:', sum(y)/len(y))
    print(f'{v} accuracy (4 week):', sum(y[:4])/len(y[:4]))
    print(f'{v} accuracy (8 week):', sum(y[:8])/len(y[:8]))
    print(f'{v} accuracy (12 week):', sum(y[:12])/len(y[:12]))
    print(f'{v} LTN frequency:', sum(class_occurrences[LTN].values())/len(class_occurrences[LTN].values()))
    print(f'{v} NRM frequency:', sum(class_occurrences[NRM].values())/len(class_occurrences[NRM].values()))
    print(f'{v} HTN frequency:', sum(class_occurrences[HTN].values())/len(class_occurrences[HTN].values()))

    week_cutoff = 12
    ax.plot(x[:week_cutoff], y[:week_cutoff], label='Prediction', color='black')
    ax.plot(x[:week_cutoff], list(class_occurrences[LTN].values())[:week_cutoff], label=LTN, linestyle='--', color='orange')
    ax.plot(x[:week_cutoff], list(class_occurrences[NRM].values())[:week_cutoff], label=NRM, linestyle='--', color='g')
    ax.plot(x[:week_cutoff], list(class_occurrences[HTN].values())[:week_cutoff], label=HTN, linestyle='--', color='r')
    if v == 'PRCP':
        ax.set_title('Precipitation Classification Accuracy By Period')
        ax.set_ylabel('Accuracy')
        ax.legend(loc='upper right', bbox_to_anchor=(1.23, 1))
    if v == 'AWND':
        ax.set_title('Wind Speed Classification Accuracy By Period')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Forecast Period')
    if v == 'TAVG':
        ax.set_title('Temperature Classification Accuracy By Period')
        ax.set_ylabel('Accuracy')

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True, help = "model name")
    args = parser.parse_args()
    
    fig, ax = plt.subplots(3, 1, figsize=(8, 8))
    display_accuracies('PRCP', ax[0], args.model_name)
    display_accuracies('TAVG', ax[1], args.model_name)
    display_accuracies('AWND', ax[2], args.model_name)
    plt.tight_layout()
    plt.show()