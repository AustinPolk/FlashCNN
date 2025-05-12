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
    # def flag_component_update(self, component):
    #     pass

class DataManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Data')

    def create(self):
        self.components['Labels'] = {}
        self.components['Labels']['Historical Data'] = ttk.Label(self.tab, text='Historical Data:')
        self.components['Labels']['Training Date'] = ttk.Label(self.tab, text='Training Date:')
        self.components['Labels']['Validation Date'] = ttk.Label(self.tab, text='Validation Date:')
        self.components['Labels']['Test Date'] = ttk.Label(self.tab, text='Test Date:')
        self.components['Labels']['Stations'] = ttk.Label(self.tab, text='Available Stations:')
        self.components['Labels']['Variables'] = ttk.Label(self.tab, text='Available Variables:')

        self.components['Entries'] = {}
        self.components['Entries']['Historical Data'] = ttk.Entry(self.tab, width=23)

        self.components['Buttons'] = {}
        self.components['Buttons']['Historical Data'] = ttk.Button(self.tab, text='...', command=self.handle_historical_file_selection, width=5)

        self.components['Dates'] = {}
        self.components['Dates']['Training Date'] = DateEntry(self.tab, width=20, date_patternstr='yyyy-mm-dd', yearint=2000)
        self.components['Dates']['Validation Date'] = DateEntry(self.tab, width=20, date_patternstr='yyyy-mm-dd', yearint=2000)
        self.components['Dates']['Test Date'] = DateEntry(self.tab, width=20, date_patternstr='yyyy-mm-dd', yearint=2000)

        self.components['Listboxes'] = {}
        self.components['Listboxes']['Stations'] = tk.Listbox(self.tab, selectmode='extended', exportselection=False)
        self.components['Listboxes']['Variables'] = tk.Listbox(self.tab, selectmode='extended', exportselection=False)

    def arrange(self):
        self.components['Labels']['Historical Data'].grid(row=0, column=0, sticky='w')
        self.components['Labels']['Training Date'].grid(row=1, column=0, sticky='w')
        self.components['Labels']['Validation Date'].grid(row=2, column=0, sticky='w')
        self.components['Labels']['Test Date'].grid(row=3, column=0, sticky='w')

        self.components['Entries']['Historical Data'].grid(row=0, column=1)

        self.components['Buttons']['Historical Data'].grid(row=0, column=2)

        self.components['Dates']['Training Date'].grid(row=1, column=1)
        self.components['Dates']['Validation Date'].grid(row=2, column=1)
        self.components['Dates']['Test Date'].grid(row=3, column=1)

        self.components['Labels']['Stations'].grid(row=4, column=0)
        self.components['Labels']['Variables'].grid(row=4, column=1)
        self.components['Listboxes']['Stations'].grid(row=5, column=0)
        self.components['Listboxes']['Variables'].grid(row=5, column=1)

    def update(self):
        pass

    def handle_historical_file_selection(self):
        store_filepath_in_entry(self.components['Entries']['Historical Data'], [("Comma-separated values", "*.csv"), ("All files", "*.*")])
        stations, variables = get_stations_and_variables_from_historical(self.components['Entries']['Historical Data'].get())
        update_listbox(stations, self.components['Listboxes']['Stations'])
        update_listbox(variables, self.components['Listboxes']['Variables'])

class ModelManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Model')

class TrainingManagerTab(ManagerTab):
    def __init__(self, parent, tab_control):
        super().__init__(parent, tab_control)
        tab_control.add(self.tab, text='Training')

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

