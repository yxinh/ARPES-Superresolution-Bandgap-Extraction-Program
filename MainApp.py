"""Launch the three-tab ARPES gap-extraction suite."""

import tkinter as tk
from tkinter import ttk

from step1_band_extraction import Step1BandExtraction
from step2_sc_gap_fitting import Step2GapFitting
from step3_temperature_dependence import Step3TemperatureDependence


class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ARPES Superconducting-Gap Extraction")
        self.root.geometry("1350x850")

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.step1_module = Step1BandExtraction(self.notebook, controller=self)
        self.notebook.add(self.step1_module, text=" Step 1: Band Structure Extraction ")

        self.step2_module = Step2GapFitting(self.notebook, controller=self)
        self.notebook.add(self.step2_module, text=" Step 2: SC Gap Fitting & F-Test ")

        self.step3_module = Step3TemperatureDependence(self.notebook, controller=self)
        self.notebook.add(self.step3_module, text=" Step 3: Temperature Dependence ")


if __name__ == "__main__":
    root = tk.Tk()
    MainApp(root)
    root.mainloop()
