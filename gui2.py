import tkinter as tk
from tkinter import ttk, filedialog
from tkcalendar import DateEntry
from functools import partial
import pickle

def store_filepath_in_entry(entry):
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("Comma-separated values", "*.csv"), ("All files", "*.*")]
    )
    if file_path:
        entry.insert(0, file_path)

class MainWindow:
    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.title("FlashGUI")
        self.main_window.resizable(True, True)

        self.tab_control = ttk.Notebook(self.main_window)

        self.data_manager = DataManagerTab(self, self.tab_control)
        self.model_manager = ModelManagerTab(self, self.tab_control)
        self.training_manager = TrainingManagerTab(self, self.tab_control)
        self.evaluation_manager = EvaluationManagerTab(self, self.tab_control)
        self.tab_control.grid(row=0, column=0, sticky='nsew')

        self.main_window.grid_rowconfigure(0, weight=1)
        self.main_window.grid_columnconfigure(0, weight=1)

    def go(self):
        self.main_window.mainloop()

class DataManagerTab:
    def __init__(self, parent, tab_control):
        self.tab = ttk.Frame(tab_control)
        self.parent = parent
        tab_control.add(self.tab, text='Data')
        self.labels = {}
        self.entries = {}
        self.buttons = {}
        self.dates = {}
        self.create()
        self.arrange()

    def create(self):
        self.labels['Historical Data'] = ttk.Label(self.tab, text='Historical Data:')
        self.labels['Daily Normal Data'] = ttk.Label(self.tab, text='Daily Normal Data:')
        self.labels['Training Date'] = ttk.Label(self.tab, text='Training Date:')
        self.labels['Validation Date'] = ttk.Label(self.tab, text='Validation Date:')
        self.labels['Test Date'] = ttk.Label(self.tab, text='Test Date:')

        self.entries['Historical Data'] = ttk.Entry(self.tab, width=23)
        self.entries['Daily Normal Data'] = ttk.Entry(self.tab, width=23)
        # self.entries['Training Date'] = ttk.Entry(self.tab, width=20)
        # self.entries['Validation Date'] = ttk.Entry(self.tab, width=20)
        # self.entries['Test Date'] = ttk.Entry(self.tab, width=20)

        self.buttons['Historical Data'] = ttk.Button(self.tab, text='...', command=partial(store_filepath_in_entry, self.entries['Historical Data']), width=5)
        self.buttons['Daily Normal Data'] = ttk.Button(self.tab, text='...', command=partial(store_filepath_in_entry, self.entries['Daily Normal Data']), width=5)

        self.dates['Training Date'] = DateEntry(self.tab, width=20, date_patternstr='y-mm-dd', yearint=2000)
        self.dates['Validation Date'] = DateEntry(self.tab, width=20, date_patternstr='y-mm-dd', yearint=2000)
        self.dates['Test Date'] = DateEntry(self.tab, width=20, date_patternstr='y-mm-dd', yearint=2000)

    def arrange(self):
        self.labels['Historical Data'].grid(row=0, column=0, sticky='w')
        self.labels['Daily Normal Data'].grid(row=1, column=0, sticky='w')
        self.labels['Training Date'].grid(row=2, column=0, sticky='w')
        self.labels['Validation Date'].grid(row=3, column=0, sticky='w')
        self.labels['Test Date'].grid(row=4, column=0, sticky='w')

        self.entries['Historical Data'].grid(row=0, column=1)
        self.entries['Daily Normal Data'].grid(row=1, column=1)
        # self.entries['Training Date'].grid(row=2, column=1)
        # self.entries['Validation Date'].grid(row=3, column=1)
        # self.entries['Test Date'].grid(row=4, column=1)

        self.buttons['Historical Data'].grid(row=0, column=2)
        self.buttons['Daily Normal Data'].grid(row=1, column=2)

        self.dates['Training Date'].grid(row=2, column=1)
        self.dates['Validation Date'].grid(row=3, column=1)
        self.dates['Test Date'].grid(row=4, column=1)

class ModelManagerTab:
    def __init__(self, parent, tab_control):
        self.parent = parent
        self.tab = ttk.Frame(tab_control)
        tab_control.add(self.tab, text='Model')

class TrainingManagerTab:
    def __init__(self, parent, tab_control):
        self.parent = parent
        self.tab = ttk.Frame(tab_control)
        tab_control.add(self.tab, text='Training')

class EvaluationManagerTab:
    def __init__(self, parent, tab_control):
        self.parent = parent
        self.tab = ttk.Frame(tab_control)
        tab_control.add(self.tab, text='Evaluation')

if __name__ == '__main__':
    MainWindow().go()
    print('exit')

