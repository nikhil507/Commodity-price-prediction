#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 4.14
# In conjunction with Tcl version 8.6
#    Jul 11, 2018 05:57:41 PM


import sys

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

'''def arima_long():
    print('find_support.arima_long')
    sys.stdout.flush()
def arima_short():
    print('find_support.arima_short')
    sys.stdout.flush()
def browse_button():
    print('find_support.browse_button')
    sys.stdout.flush()
def find_best_price():
    print('find_support.find_best_price')
    sys.stdout.flush()
'''
def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import find
find.vp_start_gui()