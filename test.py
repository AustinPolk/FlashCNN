import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from argparse import ArgumentParser

def get_above_average_days(predicted, actual, normal):
    predicted_above = []
    actual_above = []

    for i in range(len(predicted)):
        if predicted.iloc[i] > normal.iloc[i]:
            predicted_above.append(1)
        else:
            predicted_above.append(0)
        if actual.iloc[i] > normal.iloc[i]:
            actual_above.append(1)
        else:
            actual_above.append(0)

    return predicted_above, actual_above

def get_variable_totals(predicted, actual, normal):
    return sum(predicted.values), sum(actual.values), sum(normal.values)

def assess_risk(total_degrees, normal_degrees, total_winds, normal_winds, total_rainfall, normal_rainfall):
    risk_score = 0

    # heat and wind increase the risk
    risk_score += total_degrees / normal_degrees 
    risk_score += total_winds / normal_winds
    
    # rain decreases the risk
    risk_score -= total_rainfall / normal_rainfall

    return risk_score

def categorize_risk2(risk_score):
    if risk_score > 0:
        return 'High'
    else:
        return 'Low'

def categorize_risk3(risk_score):
    if risk_score > 1:
        return 'High'
    if risk_score > 0:
        return 'Medium'
    else:
        return 'Low'

def period_by_period_risk(predicted, actual, normal, station, periods=1, risk_labels=3):
    predicted_risk_labels = []
    actual_risk_labels = []

    num_days = len(predicted)
    period_length = num_days // periods

    for period in range(periods):
        period_begin = period * period_length
        period_end = (period + 1) * period_length

        period_predicted = predicted.iloc[period_begin:period_end]
        period_actual = actual.iloc[period_begin:period_end]

        # take sums of all variables to compare them to the normals
        predicted_degrees, actual_degrees, normal_degrees = get_variable_totals(predicted[f'PRCP_{station}'], actual[f'PRCP_{station}'], normal[f'PRCP_{station}'])
        predicted_winds, actual_winds, normal_winds = get_variable_totals(predicted[f'PRCP_{station}'], actual[f'PRCP_{station}'], normal[f'PRCP_{station}'])
        predicted_rainfall, actual_rainfall, normal_rainfall = get_variable_totals(predicted[f'PRCP_{station}'], actual[f'PRCP_{station}'], normal[f'PRCP_{station}'])

        predicted_risk_score = assess_risk(predicted_degrees, normal_degrees, predicted_winds, normal_winds, predicted_rainfall, normal_rainfall)
        actual_risk_score = assess_risk(actual_degrees, normal_degrees, actual_winds, normal_winds, actual_rainfall, normal_rainfall)

        if risk_labels == 3:
            predicted_risk_label = categorize_risk3(predicted_risk_score)
            actual_risk_label = categorize_risk3(actual_risk_score)
        else:
            predicted_risk_label = categorize_risk2(predicted_risk_score)
            actual_risk_label = categorize_risk2(actual_risk_score)

        predicted_risk_labels.append(predicted_risk_label)
        actual_risk_labels.append(actual_risk_label)

    return predicted_risk_labels, actual_risk_labels

def display_confusion_matrix(predicted_highs, actual_highs):
    cm = confusion_matrix(['High' if x > 0 else 'Not High' for x in actual_highs], ['High' if x > 0 else 'Not High' for x in predicted_highs], labels=['High', 'Not High'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['High', 'Not High'])
    disp.plot()
    plt.show()

def display_risk2_confusion_matrix(predicted_risks, actual_risks):
    cm = confusion_matrix(actual_risks, predicted_risks, labels=['Low', 'High'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['Low', 'High'])
    disp.plot()
    plt.show()

def calculate_stats(predicted, actual):
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
    
    for i in range(len(predicted)):
        p = predicted[i]
        e = actual[i]
    
        if p > 0 and e > 0:
            true_pos += 1
        elif p > 0:
            false_pos += 1
        elif e > 0:
            false_neg += 1
        else:
            true_neg += 1

    accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
    precision = (true_pos) / (true_pos + false_pos)
    recall = (true_pos) / (true_pos + false_neg)
    
    pos = true_pos + false_neg
    neg = true_neg + false_pos
    print('Total positive:', pos)
    print('Total negative:', neg)
    
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)

    return accuracy, precision, recall

def calculate_risk2_stats(predicted, actual):
    predicted_high_risk = [1 if x == 'High' else 0 for x in predicted]
    actual_high_risk = [1 if x == 'High' else 0 for x in actual]
    return calculate_stats(predicted_high_risk, actual_high_risk)
    
def display_variable_sma(predicted, actual, normal, variable, ax, up_to=500, window=1):
    station = 0
    
    predicted_tavg = predicted[f'{variable}_{station}'].iloc[:up_to]
    actual_tavg = actual[f'{variable}_{station}'].iloc[:up_to]
    normal_tavg = normal[f'{variable}_{station}'].iloc[:up_to]
    
    pma = predicted_tavg.rolling(window=window).mean()
    ama = actual_tavg.rolling(window=window).mean()
    nma = normal_tavg.rolling(window=window).mean()

    print('Correlation:', actual_tavg.corr(predicted_tavg))
    print(f'{window}-Day SMA Correlation:', ama.corr(pma))
    
    ax.plot(range(len(predicted_tavg)), pma, label='Predicted')
    ax.plot(range(len(predicted_tavg)), ama, label='Actual')
    ax.plot(range(len(predicted_tavg)), nma, label='Daily Normal')
    
    plt.xlabel('Days Since Forecast Began')
    if variable == 'TAVG':
        #ax.set_ylabel('Average Daily Temperature (deg F)')
        ax.set_title(f'{window}-Day SMA of Average Daily Temperature')
    elif variable == 'PRCP':
        #ax.set_ylabel('Total Daily Precipitation (hundredths in.)')
        ax.set_title(f'{window}-Day SMA of Total Daily Precipitation')
    elif variable == 'AWND':
        #ax.set_ylabel('Average Daily Wind Speed (mph)')
        ax.set_title(f'{window}-Day SMA of Average Daily Wind Speed')
    ax.legend()
    
    #plt.show()
    
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-p", "--predicted", type=str, required=True, help = "forecast path")
    parser.add_argument("-a", "--actual", type=str, required=True, help = "actual observations path")
    parser.add_argument("-n", "--normal", type=str, required=True, help = "daily normals path")
    args = parser.parse_args()
    
    predicted = pd.read_csv(args.predicted)
    actual = pd.read_csv(args.actual)
    normal = pd.read_csv(args.normal)

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    
    limit = 500
    display_variable_sma(predicted, actual, normal, 'TAVG', axes[0, 0], up_to=limit)
    display_variable_sma(predicted, actual, normal, 'TAVG', axes[1, 0], window=7, up_to=limit)

    display_variable_sma(predicted, actual, normal, 'PRCP', axes[0, 1], up_to=limit)
    display_variable_sma(predicted, actual, normal, 'PRCP', axes[1, 1], window=7, up_to=limit)

    display_variable_sma(predicted, actual, normal, 'AWND', axes[0, 2], up_to=limit)
    display_variable_sma(predicted, actual, normal, 'AWND', axes[1, 2], window=7, up_to=limit)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
    
    station = 0
    predicted_high_temps, actual_high_temps = get_above_average_days(predicted[f'TAVG_{station}'], actual[f'TAVG_{station}'], normal[f'TAVG_{station}'])
    predicted_high_winds, actual_high_winds = get_above_average_days(predicted[f'AWND_{station}'], actual[f'AWND_{station}'], normal[f'AWND_{station}'])

    stats_limit = 112
    print('High Temp Stats:')
    calculate_stats(predicted_high_temps, actual_high_temps)
    display_confusion_matrix(predicted_high_temps, actual_high_temps)
    print(f'High Temp Stats (first {stats_limit} days):')
    calculate_stats(predicted_high_temps[:stats_limit], actual_high_temps[:stats_limit])
    display_confusion_matrix(predicted_high_temps[:stats_limit], actual_high_temps[:stats_limit])
    print('High Wind Stats:')
    calculate_stats(predicted_high_winds, actual_high_winds)
    display_confusion_matrix(predicted_high_winds, actual_high_winds)
    print(f'High Wind Stats (first {stats_limit} days):')
    calculate_stats(predicted_high_winds[:stats_limit], actual_high_winds[:stats_limit])
    display_confusion_matrix(predicted_high_winds[:stats_limit], actual_high_winds[:stats_limit])

    predicted_risk_labels, actual_risk_labels = period_by_period_risk(predicted, actual, normal, 0, periods=64, risk_labels=2)

    print('High Risk Stats:')
    calculate_risk2_stats(predicted_risk_labels, actual_risk_labels)
    display_risk2_confusion_matrix(predicted_risk_labels, actual_risk_labels)
    print(f'High Risk Stats (first {stats_limit//7} weeks):')
    calculate_risk2_stats(predicted_risk_labels[:stats_limit//7], actual_risk_labels[:stats_limit//7])
    display_risk2_confusion_matrix(predicted_risk_labels[:stats_limit//7], actual_risk_labels[:stats_limit//7])