'''

use the regular expression to extract the loss in order to plot them on the same figure

'''
import os
import re
import matplotlib.pyplot as plt

clf_pattern = re.compile(r'Classification\s*loss\:\s*(\d+\.\d+)')
reg_pattern = re.compile(r'Regression\s*loss\:\s*(\d+\.\d+)')

with open('/Users/zhangyunping/PycharmProjects/3Ddetection/experiment/out_dec_continue.log','r') as f:
    lines = f.readlines()
    classification_err = []
    regression_err = []
    for line in lines:
        if clf_pattern.findall(line):
            classification_err.append(float(clf_pattern.findall(line)[0]))
        if reg_pattern.findall(line):
            regression_err.append(float(reg_pattern.findall(line)[0]))

with open('/Users/zhangyunping/PycharmProjects/3Ddetection/experiment/out_dec_continue2.log','r') as f:
    lines = f.readlines()
    for line in lines:
        if clf_pattern.findall(line):
            classification_err.append(float(clf_pattern.findall(line)[0]))
        if reg_pattern.findall(line):
            regression_err.append(float(reg_pattern.findall(line)[0]))

with open('/Users/zhangyunping/PycharmProjects/3Ddetection/experiment/out_Jan.log','r') as f:
    lines = f.readlines()
    for line in lines:
        if clf_pattern.findall(line):
            classification_err.append(float(clf_pattern.findall(line)[0]))
        if reg_pattern.findall(line):
            regression_err.append(float(reg_pattern.findall(line)[0]))

plt.plot(classification_err,color='green',label='classification_error')
plt.plot(regression_err,color='red',label = 'regression_error')
plt.legend()
plt.show()