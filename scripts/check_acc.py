import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import sys

# Get input file from command line argument
input_file = sys.argv[1]
data = pd.read_csv(input_file)

val = data.dropna(subset = ['val_loss'])
train = data.dropna(subset = ['train_loss'])
epoch = train["epoch"]  
train_loss = train["train_loss"]
val_loss = val["val_loss"]
train_acc = train["train_pcc"]
val_acc = val["val_pcc"]

fig = plt.figure(figsize = (7,6)) 
p1 = pl.plot(epoch, train_loss,'r-', label = u'train_loss')
p2 = pl.plot(epoch,val_loss, 'b-', label = u'val_loss')
pl.legend()
pl.xlabel(u'Epoch')
pl.ylabel(u'MSE loss')

output_file = input_file + '_loss.png'
plt.savefig(output_file)
plt.close()

fig = plt.figure(figsize = (7,6)) 
p3 = pl.plot(epoch, train_acc,'r-', label = u'train_pcc')
p4 = pl.plot(epoch,val_acc, 'b-', label = u'val_pcc')
pl.legend()
pl.xlabel(u'Epoch')
pl.ylabel(u'PCC')
pl.ylim(0.2, 1)

output_file = input_file + '_pcc.png'
plt.savefig(output_file)
plt.close()