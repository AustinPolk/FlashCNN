import tkinter as tk
from tkinter import ttk, filedialog
from tkcalendar import DateEntry
from functools import partial
import pandas as pd

def store_filepath_in_entry(entry, filetypes):
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=filetypes
    )
    if file_path:
        entry.insert(0, file_path)

def update_listbox(values, listbox):
    listbox.delete(0, tk.END)  # Clear existing items
    for item in values:
        listbox.insert(tk.END, item)  # Add new items

def get_stations_and_variables_from_historical(filepath):
    df = pd.read_csv(filepath)
    variables = [x for x in df.columns if x not in ['STATION', 'NAME', 'DATE']]
    stations = [x for x in df['STATION'].unique()]
    return stations, variables