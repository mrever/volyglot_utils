py3 << EOL
import numpy as np
import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import bokeh
import pickle
import os
import re
import sys
import pandas as pd
from glob import glob
from time import time as now, sleep
import inspect
import pyperclip as pp
import plumbum as pb

from numpy import array, linspace, exp, amin, amax, sum as Σ, mean as μ, sin, cos, tan, log, log10, meshgrid, zeros, ones, append, dot, pi, pi as π, sqrt, arange
from numpy.linalg import norm
# inspect.getmodule(zeros)

from matplotlib.pyplot import figure, plot, imshow, show, grid, xlabel, ylabel, title, draw, legend

from shutil import copyfile as cp

cmd = pb.cmd

def ls(*args):
    return [r for r in pb.cmd.ls(args).split('\n') if len(r) > 0]

def mshow(*args, **kwargs):
    h = figure()
    imshow(*args, **kwargs)
    if '__nvim__' not in globals() or __nvim__ == False:
        show(block=False)
    else:
        show()
    return h

def mplot(*args, **kwargs):
    h = figure()
    plot(*args, **kwargs)
    grid()
    if '__nvim__' not in globals() or __nvim__ == False:
        show(block=False)
    else:
        show()
    return h

def pdump(fname, dat):
    with open(fname, 'wb') as f:
        pickle.dump(dat, f)

def pload(fname):
    with open(fname, 'rb') as f:
        dat = pickle.load(f)
    return dat

def toarray(*args):
    if len(args) == 1 and type(args[0]) is list:
        return np.array(args[0])
    return np.array(args)
 
def Copy(dat):
    pp.copy(str(dat))

def Paste():
    return pp.paste()


def rotate_image(img, angle):
    size_reverse = np.array(img.shape[1::-1]) # swap x with y
    M = cv.getRotationMatrix2D(tuple(size_reverse / 2.), angle, 1.)
    MM = np.absolute(M[:,:2])
    size_new = MM @ size_reverse
    M[:,-1] += (size_new - size_reverse) / 2.
    return cv.warpAffine(img, M, tuple(size_new.astype(int)))

def loadenvvariables():
    for envvar in list(os.environ.keys()):
        if envvar:
            globals()[envvar] = os.environ[envvar]


import bokeh.plotting as bplotting
import bokeh.models.tools as btools
def bplot(x=None, y=None, outfile=None, title='', xlab='', ylab='', legend=None,
         linew=2, pwidth=1280, pheight=720):
    if not outfile:
        outfile = f"{vyth.hometmp}{np.random.randint(10000000)}.html"
    if not y:
        y = np.array(x).copy()
        x = np.arange(len(x))
    bplotting.output_file(outfile)
    p = bplotting.figure(title=title, x_axis_label=xlab, y_axis_label=ylab)
    if legend:
        p.line(x,y, legend=legend, line_width=linew)
    else:
        p.line(x,y, line_width=linew)
    p.add_tools(btools.HoverTool())
    p.toolbar.logo = None #don't show Bokeh icon/link
    bplotting.save(p)
    vyth.writebuffhtml(f'<object data="{outfile}" width="{pwidth}" height="{pheight}"></object>',
                       br=3, shownum=True)
    
def bimagesc(data, outfile=None, title='', xlab='', ylab='', pwidth=1280, pheight=720):
    if not outfile:
        outfile = f"{vyth.hometmp}{np.random.randint(10000000)}.html"
    bplotting.output_file(outfile)
    p = bplotting.figure(title=title, x_axis_label=xlab, y_axis_label=ylab)
    p.image(image=[data], x=0, y=0, dw=data.shape[1], dh=data.shape[0], palette='Spectral11')
    p.add_tools(btools.HoverTool())
    p.toolbar.logo = None #don't show Bokeh icon/link
    bplotting.save(p)
    vyth.writebuffhtml(f'<object data="{outfile}" width="{pwidth}" height="{pheight}"></object>',
                       br=3, shownum=True)

def bimshow(img_orig, scale=1, **figure_kwargs):
    outfile = f"{vyth.hometmp}{np.random.randint(10000000)}.html"
    bplotting.output_file(outfile)
    img = img_orig.astype(np.uint8)
    if img.ndim == 2:  # gray input
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGBA)
    elif img.ndim == 3:  # rgb input
        img = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
    img = np.ascontiguousarray(np.flipud(img))
    img2 = img.view('uint32').reshape(img.shape[:2])
    p = bplotting.figure(
        x_range=(0, img.shape[1]),
        y_range=(img.shape[0], 0),
        frame_width=img.shape[1]*scale,
        frame_height=img.shape[0]*scale,
        **figure_kwargs)
    p.add_tools(btools.HoverTool(
        tooltips=[
            ("(x, y)", "($x, $y)"),
            ("RGB", "(@R, @G, @B)")]))
    p.toolbar.logo = None #don't show Bokeh icon/link
    source = bplotting.ColumnDataSource(data=dict(
        img=[img2], x=[0], y=[img.shape[0]],
        dw=[img.shape[1]], dh=[img.shape[0]],
        R=[img[::-1, :, 0]], G=[img[::-1, :, 1]], B=[img[::-1, :, 2]]))
    p.image_rgba(source=source, image='img', x='x', y='y', dw='dw', dh='dh')
    bplotting.save(p)  # open a browser
    vyth.writebuffhtml(f'<object data="{outfile}" width="{img.shape[1]+100}" height="{img.shape[0]+100}"></object>',  br=3, shownum=True)

EOL

