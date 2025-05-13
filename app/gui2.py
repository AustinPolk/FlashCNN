import tkinter as tk
from tkinter import ttk, filedialog
from tkcalendar import DateEntry
from functools import partial
from helpers import *

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
        self.forecast_manager = ForecastManagerTab(self, self.tab_control)
        self.tab_control.grid(row=0, column=0, sticky='nsew')

        self.main_window.grid_rowconfigure(0, weight=1)
        self.main_window.grid_columnconfigure(0, weight=1)

    def update(self):
        self.data_manager.update()
        self.model_manager.update()
        self.training_manager.update()
        self.evaluation_manager.update()
        self.forecast_manager.update()
        self.main_window.after(50, self.update)

    def go(self):
        self.update()
        self.main_window.mainloop()

class ManagerTab:
    def __init__(self, parent, tab_control):
        self.tab = ttk.Frame(tab_control)
        self.parent = parent
        self.components = {}
        self.callback_returns = {}
        self.create()
        self.arrange()
        self.bind()
        self.update()
    def create(self):
        pass
    def arrange(self):
        pass
    def bind(self):
        pass
    def update(self):
        pass

class DataManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Data')

    def create(self):
        self.components['Labels'] = {}
        self.components['Labels']['Historical Data'] = ttk.Label(self.tab, text='Historical Data:')
        self.components['Labels']['Stations'] = ttk.Label(self.tab, text='Available Stations:')
        self.components['Labels']['Variables'] = ttk.Label(self.tab, text='Available Variables:')

        self.components['Entries'] = {}
        self.components['Entries']['Historical Data'] = ttk.Entry(self.tab, width=23)

        self.components['Buttons'] = {}
        self.components['Buttons']['Historical Data'] = ttk.Button(self.tab, text='...', command=self.handle_historical_file_selection, width=5)

        self.components['Listboxes'] = {}
        self.components['Listboxes']['Stations'] = tk.Listbox(self.tab, selectmode='extended', exportselection=False)
        self.components['Listboxes']['Variables'] = tk.Listbox(self.tab, selectmode='extended', exportselection=False)

    def arrange(self):
        self.components['Labels']['Historical Data'].grid(row=0, column=0, sticky='w')

        self.components['Entries']['Historical Data'].grid(row=0, column=1)

        self.components['Buttons']['Historical Data'].grid(row=0, column=2)

        self.components['Labels']['Stations'].grid(row=4, column=0)
        self.components['Labels']['Variables'].grid(row=4, column=1)
        self.components['Listboxes']['Stations'].grid(row=5, column=0, rowspan=15)
        self.components['Listboxes']['Variables'].grid(row=5, column=1, rowspan=15)

    def handle_historical_file_selection(self):
        store_filepath_in_entry(self.components['Entries']['Historical Data'], [("Comma-separated values", "*.csv"), ("All files", "*.*")])
        stations, variables = get_stations_and_variables_from_historical(self.components['Entries']['Historical Data'].get())
        update_listbox(stations, self.components['Listboxes']['Stations'])
        update_listbox(variables, self.components['Listboxes']['Variables'])

class ModelManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Model')

    def create(self):
        self.components['Labels'] = {}
        self.components['Labels']['Lookback'] = ttk.Label(self.tab, text='Lookback:')
        self.components['Labels']['Kernel Size'] = ttk.Label(self.tab, text='Kernel Size:')
        self.components['Labels']['Pool Size'] = ttk.Label(self.tab, text='Pool Size:')
        self.components['Labels']['Channel Multiplier'] = ttk.Label(self.tab, text='Channel Multiplier:')
        self.components['Labels']['Convolutional Layers'] = ttk.Label(self.tab, text='Convolutional Layers:')
        self.components['Labels']['Dropout Rate'] = ttk.Label(self.tab, text='Dropout Rate:')
        self.components['Labels']['Activation Function'] = ttk.Label(self.tab, text='Activation Function:')

        self.components['Entries'] = {}
        self.components['Entries']['Lookback'] = ttk.Entry(self.tab, width=20)
        self.components['Entries']['Kernel Size'] = ttk.Entry(self.tab, width=20)
        self.components['Entries']['Pool Size'] = ttk.Entry(self.tab, width=20)
        self.components['Entries']['Channel Multiplier'] = ttk.Entry(self.tab, width=20)
        self.components['Entries']['Convolutional Layers'] = ttk.Entry(self.tab, width=20)
        self.components['Entries']['Dropout Rate'] = ttk.Entry(self.tab, width=20)

        self.components['Checkboxes'] = {}
        self.components['Checkboxes']['Means Channel'] = tk.Checkbutton(self.tab, text='Use Means Channel')
        self.components['Checkboxes']['H-Means Channel'] = tk.Checkbutton(self.tab, text='Use H-Means Channel')
        self.components['Checkboxes']['H-Means2 Channel'] = tk.Checkbutton(self.tab, text='Use H-Means2 Channel')
        self.components['Checkboxes']['ReLU Convolutional'] = tk.Checkbutton(self.tab, text='Use ReLU in Convolutional Layers')
        self.components['Checkboxes']['Average Pooling'] = tk.Checkbutton(self.tab, text='Use Average Pooling instead of Max Pooling')
        self.components['Checkboxes']['Output Size'] = tk.Checkbutton(self.tab, text='Add Output Size to Fully Connected Layer Sizes')

        self.components['Comboboxes'] = {}
        self.components['Comboboxes']['Fully Connected Activation'] = ttk.Combobox(self.tab, width=15, values=('ReLU', 'Leaky ReLU'))

    def arrange(self):
        self.components['Labels']['Lookback'].grid(row=0, column=0, sticky='w')
        self.components['Labels']['Kernel Size'].grid(row=1, column=0, sticky='w')
        self.components['Labels']['Pool Size'].grid(row=2, column=0, sticky='w')
        self.components['Labels']['Channel Multiplier'].grid(row=3, column=0, sticky='w')
        self.components['Labels']['Convolutional Layers'].grid(row=4, column=0, sticky='w')
        self.components['Labels']['Dropout Rate'].grid(row=5, column=0, sticky='w')

        self.components['Entries']['Lookback'].grid(row=0, column=1)
        self.components['Entries']['Kernel Size'].grid(row=1, column=1)
        self.components['Entries']['Pool Size'].grid(row=2, column=1)
        self.components['Entries']['Channel Multiplier'].grid(row=3, column=1)
        self.components['Entries']['Convolutional Layers'].grid(row=4, column=1)
        self.components['Entries']['Dropout Rate'].grid(row=5, column=1)

        self.components['Checkboxes']['Means Channel'].grid(row=6, column=0, columnspan=2, sticky='w')
        self.components['Checkboxes']['H-Means Channel'].grid(row=7, column=0, columnspan=2, sticky='w')
        self.components['Checkboxes']['H-Means2 Channel'].grid(row=8, column=0, columnspan=2, sticky='w')
        self.components['Checkboxes']['ReLU Convolutional'].grid(row=9, column=0, columnspan=2, sticky='w')
        self.components['Checkboxes']['Average Pooling'].grid(row=10, column=0, columnspan=2, sticky='w')
        self.components['Checkboxes']['Output Size'].grid(row=11, column=0, columnspan=2, sticky='w')

        self.components['Labels']['Activation Function'].grid(row=12, column=0, sticky='w')
        self.components['Comboboxes']['Fully Connected Activation'].grid(row=12, column=1, sticky='w')

class TrainingManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Training')

    def create(self):
        self.components['Labels'] = {}
        self.components['Labels']['Training Start Date'] = ttk.Label(self.tab, text='Training Start Date:')
        self.components['Labels']['Training End Date'] = ttk.Label(self.tab, text='Training End Date:')
        self.components['Labels']['Validation Start Date'] = ttk.Label(self.tab, text='Validation Start Date:')
        self.components['Labels']['Validation End Date'] = ttk.Label(self.tab, text='Validation End Date:')

        self.components['Dates'] = {}
        self.components['Dates']['Training Start Date'] = DateEntry(self.tab, width=20, date_patternstr='yyyy-mm-dd', yearint=2000)
        self.components['Dates']['Training End Date'] = DateEntry(self.tab, width=20, date_patternstr='yyyy-mm-dd', yearint=2000)
        self.components['Dates']['Validation Start Date'] = DateEntry(self.tab, width=20, date_patternstr='yyyy-mm-dd', yearint=2000)
        self.components['Dates']['Validation End Date'] = DateEntry(self.tab, width=20, date_patternstr='yyyy-mm-dd', yearint=2000)

    def arrange(self):
        self.components['Labels']['Training Start Date'].grid(row=0, column=0, sticky='w')
        self.components['Labels']['Training End Date'].grid(row=1, column=0, sticky='w')
        self.components['Labels']['Validation Start Date'].grid(row=2, column=0, sticky='w')
        self.components['Labels']['Validation End Date'].grid(row=3, column=0, sticky='w')

        self.components['Dates']['Training Start Date'].grid(row=0, column=1)
        self.components['Dates']['Training End Date'].grid(row=1, column=1)
        self.components['Dates']['Validation Start Date'].grid(row=2, column=1)
        self.components['Dates']['Validation End Date'].grid(row=3, column=1)

class EvaluationManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Evaluation')

class ForecastManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Forecast')

if __name__ == '__main__':
    MainWindow().go()
    print('exit')

