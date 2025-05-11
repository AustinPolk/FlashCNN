from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from functools import partial
import threading

def store_filepath_in_entry(entry):
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Comma-separated files", "*.csv"), ("All files", "*.*")]
    )
    if file_path:
        entry.insert(0, file_path)
        
def start(window, entries, labels, progress_bars):
    
    historical_path 
    
    
    progress_bar["value"] = 0  # Reset
    window.update_idletasks()
    for i in range(101):
        progress_bar["value"] = i
        window.update_idletasks()
        window.after(20)  # Small delay to simulate work

if __name__ == '__main__':

    main_window = Tk()
    main_window.title = 'Model Training and Evaluation'

    app_row = 0
    Label(main_window, text='Data:').grid(row=app_row, columnspan=3)
    
    app_row += 1
    Label(main_window, text='Historical Data File:').grid(row=app_row, sticky='w')
    historical_entry = Entry(main_window, width=50)
    historical_entry.grid(row=app_row, column=1)
    historical_button = Button(main_window, text='...', command=partial(store_filepath_in_entry, historical_entry))
    historical_button.grid(row=app_row, column=2)
    
    app_row += 1
    Label(main_window, text='Normal Data File:').grid(row=app_row, sticky='w')
    normal_entry = Entry(main_window, width=50)
    normal_entry.grid(row=app_row, column=1)
    normal_button = Button(main_window, text='...', command=partial(store_filepath_in_entry, normal_entry))
    normal_button.grid(row=app_row, column=2)
    
    app_row += 1
    Label(main_window, text='Parameters:').grid(row=app_row, columnspan=3)
    
    app_row += 1
    Label(main_window, text='Lookback:').grid(row=app_row, sticky='w')
    lookback_entry = Entry(main_window, width=50)
    lookback_entry.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    Label(main_window, text='Kernel Size:').grid(row=app_row, sticky='w')
    kernel_size_entry = Entry(main_window, width=50)
    kernel_size_entry.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    Label(main_window, text='Model:').grid(row=app_row, columnspan=3)
    
    app_row += 1
    Label(main_window, text='ID:').grid(row=app_row, sticky='w')
    model_id_entry = Entry(main_window, width=50)
    model_id_entry.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    Label(main_window, text='Training Epochs:').grid(row=app_row, sticky='w')
    epochs_entry = Entry(main_window, width=50)
    epochs_entry.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    go_button = Button(main_window, text='Go', width=20)
    go_button.grid(row=app_row, columnspan=2, pady=(5, 0))
    
    app_row += 1
    Label(main_window, text='Training:').grid(row=app_row, columnspan=3)
    
    app_row += 1
    Label(main_window, text='Final Loss:').grid(row=app_row, sticky='w')
    final_loss_label = Label(main_window)
    final_loss_label.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    training_progress_bar = ttk.Progressbar(main_window, orient='horizontal', length=300, mode='determinate')
    training_progress_bar.grid(row=app_row, columnspan=3, pady=(10, 10))
    
    app_row += 1
    Label(main_window, text='Evaluation:').grid(row=app_row, columnspan=3)
    
    app_row += 1
    Label(main_window, text='PRCP Accuracy:').grid(row=app_row, sticky='w')
    prcp_accuracy_label = Label(main_window)
    prcp_accuracy_label.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    Label(main_window, text='TAVG Accuracy:').grid(row=app_row, sticky='w')
    tavg_accuracy_label = Label(main_window)
    tavg_accuracy_label.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    Label(main_window, text='AWND Accuracy:').grid(row=app_row, sticky='w')
    awnd_accuracy_label = Label(main_window)
    awnd_accuracy_label.grid(row=app_row, column=1, sticky='w')
    
    app_row += 1
    evaluation_progress_bar = ttk.Progressbar(main_window, orient='horizontal', length=300, mode='determinate')
    evaluation_progress_bar.grid(row=app_row, columnspan=3, pady=(10, 10))
    
    all_entries = {
        'Historical': historical_entry, 
        'Normal': normal_entry, 
        'Lookback': lookback_entry, 
        'Kernel Size': kernel_size_entry, 
        'ID': model_id_entry,
        'Training Epochs': epochs_entry,
    }
    all_output_labels = {
        'Final Loss': final_loss_label,
        'PRCP Accuracy': prcp_accuracy_label,
        'TAVG Accuracy': tavg_accuracy_label,
        'AWND Accuracy': awnd_accuracy_label,
    }
    all_progress_bars = {
        'Training': training_progress_bar,
        'Evaluation': evaluation_progress_bar
    }
    
    go_button.configure(command=partial(start, main_window, all_entries, all_output_labels, all_progress_bars))
    
    main_window.mainloop()