"""
matplotlibplugin.py
"""

import os

from matplotlib import rcParams
from matplotlibwidget import MatplotlibWidget
from PyQt6.QtDesigner import QPyDesignerCustomWidgetPlugin
from PyQt6.QtGui import QIcon

rcParams["font.size"] = 9


class MatplotlibPlugin(QPyDesignerCustomWidgetPlugin):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._initialized = False

    def initialize(self, editor):
        self._initialized = True

    def isInitialized(self):
        return self._initialized

    def createWidget(self, parent):
        return MatplotlibWidget(parent)

    def name(self):
        return "MatplotlibWidget"

    def group(self):
        return "PyQt"

    def icon(self):
        return QIcon(os.path.join(rcParams["datapath"], "images", "matplotlib.png"))

    def toolTip(self):
        return ""

    def whatsThis(self):
        return ""

    def isContainer(self):
        return False

    def domXml(self):
        return '<widget class="MatplotlibWidget" name="mplwidget">\n</widget>\n'

    def includeFile(self):
        return "matplotlibwidget"
