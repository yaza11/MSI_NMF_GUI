"""Setp and run GUI to plot MSI images."""
from PyQt5 import QtWidgets, uic, QtCore, QtGui
import qdarktheme

from mfe.feature import repeated_nmf

import sys
from dataclasses import dataclass
from typing import Iterable

import pickle
import pandas as pd
import numpy as np
import re
import os

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MaxAbsScaler

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib
# Ensure using PyQt5 backend
matplotlib.use('Qt5Agg')

ui_file = r'NMF.ui'


def check_file_integrity(
        file: str, is_file: bool = True, suffixes: list[str] = None
) -> bool:
    """Check if a given file exists and optionally is of right type."""
    if os.path.exists(file):
        if is_file != os.path.isfile(file):
            print(f'{file} is not the right type (folder instead of file or vise versa)')
            return False
        elif is_file and (suffixes is not None):
            if (suffix := os.path.splitext(file)[1]) not in suffixes:
                print(f'{file} should type should be one of {suffixes}, not {suffix}')
                return False
            else:
                print(f'{file} is okay')
                return True
        else:
            print(f'{file} is okay')
            return True
    elif file != '':
        print(f'{file} does not exist.')
    return False


@dataclass
class Options:
    is_mode_NMF: bool
    is_repeated_NMF: bool
    N_rep: int
    N_modes: float | int


class EmittingStream(QtCore.QObject):
    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass  # Implement the flush method as a no-op


class MplCanvas(FigureCanvas):

    def __init__(self):
        fig = plt.Figure()
        self.axes = fig.subplots(1, 1)
        super(MplCanvas, self).__init__(fig)

    def replace_figure_and_axes(self, new_figure, new_axes):
        # Remove the current axes from the current figure
        self.figure.clear()  # Clear the entire figure

        # Assign the new axes and figure
        self.figure = new_figure
        num_rows, num_cols = new_axes.shape
        self.axes = new_axes

        # Redraw the canvas to reflect the updated subplots
        self.draw()


class UI(QtWidgets.QMainWindow):
    def __init__(self):
        super(UI, self).__init__()
        uic.loadUi(ui_file, self)
        self.initiate_plt_area()
        self.show()
        # console output in textView
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        # load default options
        self.update_options()

        # link buttons to functions
        self.link_widgets()

        self.has_valid_mass_file = False

    def closeEvent(self, event):
        # Restore sys.stdout when the GUI is closed
        sys.stdout = sys.__stdout__
        event.accept()

    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.textEdit_console.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.textEdit_console.setTextCursor(cursor)
        self.textEdit_console.ensureCursorVisible()

    def link_widgets(self):
        self.btn_read_data.clicked.connect(self.read_data)

        self.btn_browse_data.clicked.connect(self.get_ft_from_dialog)

        self.btn_go.clicked.connect(self.go_btn)

        self.btn_load.clicked.connect(self.load_settings)
        self.btn_save.clicked.connect(self.save_settings)

    def get_ft_from_dialog(self):
        txt_file = QtWidgets.QFileDialog.getOpenFileName(self, 'File to read', 'c:\\', '*.csv')[0]
        self.lineEdit_ft_file.setText(txt_file)

    def read_data(self):
        """Activated by read-btn."""
        try:
            file = self.lineEdit_ft_file.text()
            file = re.findall(r'(?:file:///)?(.+)', file)[0]
        except:
            print('invalid input file')
            return
        if not check_file_integrity(file, suffixes=['.csv']):
            return

        ft = pd.read_csv(file, index_col=0)

        if ('x' not in ft.columns) or ('y' not in ft.columns):
            print('feature table must have "x" and "y" column')
            return

        self.ft = ft.sort_values(by=['y', 'x']).reset_index(drop=True)
        print('finished loading')

    def update_fs(self):
        SMALL_SIZE = self.horizontalScrollBar.value() / 10
        MEDIUM_SIZE = SMALL_SIZE * 3 / 2
        BIGGER_SIZE = SMALL_SIZE * 5 / 3

        plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    def update_options(self):
        self.update_fs()

        mode = self.comboBox.currentText()
        if mode == 'NMF':
            is_mode_NMF = True
        else:
            is_mode_NMF = False

        is_repeated_NMF = self.checkBox_is_repeated_NMF.isChecked()
        try:
            N_rep = int(self.lineEdit_N_rep.text())
        except:
            N_rep = 0
            is_repeated_NMF = False
        try:
            N_modes = float(self.lineEdit_n_modes.text())
            if is_mode_NMF:
                N_modes = int(N_modes)
        except:
            print('invalid mode value')
            return

        self.opts = Options(
            is_mode_NMF=is_mode_NMF,
            is_repeated_NMF=is_repeated_NMF,
            N_rep=N_rep,
            N_modes=N_modes
        )

    def initiate_plt_area(self):
        placeholder = self.findChild(QtWidgets.QWidget, 'plt_area')

        # Get the existing layout of the widget or create a new one if it doesn't have a layout
        layout = QtWidgets.QVBoxLayout()
        placeholder.setLayout(layout)

        self.canvas = MplCanvas()

        # Add the FigureCanvas to the layout
        navigation_toolbar = NavigationToolbar(self.canvas, self)
        navigation_toolbar.setStyleSheet("background-color: white;")
        layout.addWidget(navigation_toolbar)
        layout.addWidget(self.canvas)

        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        # Set stretch factor to 1 to make it expand to fill the available space
        layout.setStretchFactor(self.canvas, 1)

        self.canvas.show()

    def updata_plt_area(self):
        self.canvas.replace_figure_and_axes(self.fig, self.axs)
        self.canvas.draw()

    def get_data_columns(self):
        if ('ft' not in self.__dict__) or (self.ft is None):
            return None
        columns = self.ft
        columns_valid = []

        columns_valid = [col for col in columns if str(
            col).replace('.', '', 1).isdigit()]

        data_columns = np.array(columns_valid)
        return data_columns

    def get_xy(self):
        return self.ft.loc[:, ['x', 'y']]

    def go_btn(self):
        self.update_options()
        if not hasattr(self, 'ft'):
            print('read in some data')
            return

        if self.opts.is_mode_NMF:
            self.analyzing_NMF()
            self.plt_NMF()
        else:
            self.analyzing_PCA()
            self.plt_PCA()

    def analyzing_NMF(self):

        use_repeated_NMF = self.opts.is_repeated_NMF
        k = self.opts.N_modes
        N_rep = self.opts.N_rep

        columns = self.get_data_columns()
        data = self.ft.loc[:, list(columns)].copy()
        # exclude rows containing only nans, mask is True for valid rows
        mask_valid_rows = ~np.isnan(data).all(axis=1)
        data = data.loc[mask_valid_rows, :].copy()

        self.nmf_xy = self.get_xy()[mask_valid_rows]

        if np.any(data.to_numpy() < 0):
            print(f'Warning: found values smaller than 0 in NMF: \
{np.min(np.min(data))}. Clipping negative values to 0.')
            data[data < 0] = 0
        # fill remaining nans with zeros
        data = data.fillna(0)

        FT_s = MaxAbsScaler().fit_transform(data)

        if use_repeated_NMF:
            S = repeated_nmf(FT_s, k, N_rep, max_iter=10_000)
            self.W = S.matrix_w_accum
            self.H = S.matrix_h_accum
        else:
            model = NMF(n_components=k, max_iter=100_000, init='nndsvd')

            self.W = model.fit_transform(FT_s)
            self.H = model.components_

    def analyzing_PCA(self):
        print('PCA is not implemented yet')
        # TODO

    def plt_PCA(self):
        # TODO
        pass

    @QtCore.pyqtSlot()
    def plt_NMF(self):
        self.update_options()
        print('calculating NMF')
        self.analyzing_NMF()

        k = self.opts.N_modes

        # put in df
        W_df = pd.DataFrame(self.W, index=self.nmf_xy.index)

        W_df[['x', 'y']] = self.nmf_xy

        self.fig = plt.Figure(
            figsize=self.canvas.figure.get_size_inches(),
            dpi=self.canvas.figure.get_dpi(),
            layout='compressed'
        )
        self.axs = self.fig.subplots(
            nrows=k,
            ncols=2,
            sharex='col'
        )

        x_vals = self.get_data_columns().astype(float)
        for i in range(k):
            values = W_df.pivot(index='y', columns='x', values=i)
            self.axs[i, 0].imshow(values,
                                  aspect='equal',
                                  interpolation='none')

            if self.H.shape[1] == len(x_vals):
                self.axs[i, 1].stem(
                    np.array(x_vals),
                    self.H[i, :],
                    markerfmt=' ',
                    linefmt='blue'
                )
            else:
                self.axs[i, 1].stem(
                    range(self.H.shape[1]),
                    self.H[i, :],
                    markerfmt=' ',
                    linefmt='blue'
                )

        self.updata_plt_area()

    def save_settings(self):
        fields = [
            'lineEdit_ft_file', 'comboBox', 'checkBox_is_repeated_NMF', 'lineEdit_N_rep',
            'lineEdit_n_modes', 'horizontalScrollBar',
        ]

        entries = []
        for field in fields:
            widget_type = field.split('_')[0]
            if widget_type == 'lineEdit':
                e = self.findChild(QtWidgets.QLineEdit, field).text()
            elif widget_type == 'comboBox':
                e = self.findChild(QtWidgets.QComboBox, field).currentText()
            elif widget_type == 'checkBox':
                e = self.findChild(QtWidgets.QCheckBox, field).isChecked()
            elif widget_type == 'horizontalScrollBar':
                e = self.findChild(QtWidgets.QScrollBar, field).value()
            else:
                raise NotImplementedError
            entries.append(e)
        d = dict(zip(fields, entries))
        with open('gui_settings_nmf.pickle', 'wb') as handle:
            pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('saved settings')

    def load_settings(self):
        print('loading settings')
        with open('gui_settings_nmf.pickle', 'rb') as handle:
            d = pickle.load(handle)

        for field, entry in d.items():
            try:
                widget_type = field.split('_')[0]
                if widget_type == 'lineEdit':
                    e = self.findChild(QtWidgets.QLineEdit, field)
                    e.setText(entry)
                elif widget_type == 'comboBox':
                    e = self.findChild(QtWidgets.QComboBox, field)
                    e.setCurrentText(entry)
                elif widget_type == 'checkBox':
                    e = self.findChild(QtWidgets.QCheckBox, field)
                    e.setChecked(entry)
            except:
                pass

        self.read_data()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    qdarktheme.setup_theme()
    window = UI()
    sys.exit(app.exec_())
