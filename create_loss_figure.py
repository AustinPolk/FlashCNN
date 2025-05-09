import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-m", "--model_name", type=str, required=True, help = "model name")
    args = parser.parse_args()
    
    with open(fr"C:\Users\Austin\Desktop\Desktop Projects\flash\models\loss_p{args.model_name}.pkl", 'rb') as f:
        loaded = pickle.load(f)
    
    _, train, test = loaded
    
    plt.plot(range(len(train)), train, label='Training Loss')
    plt.plot(range(len(train)), test, label='Validation Loss')
    plt.ylabel('MSE Loss')
    plt.xlabel('Epochs')
    plt.title('Loss Observed During Training')
    plt.legend(loc='upper right')
    plt.show()