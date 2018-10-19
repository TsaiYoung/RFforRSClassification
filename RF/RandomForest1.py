# -*- coding: utf-8 -*-
print(__doc__)

import sys, os
import numpy as np
import pandas as pd
from osgeo import gdal
from osgeo.gdalconst import *
from sklearn.ensemble import RandomForestClassifier
from Tkinter import *
import tkMessageBox
import threading
import copy
import Queue
import types


# 相关函数(写在使用之前)
# 事件1
def carryout(v1, v2, v4, v5):
    if v1.get() == "":
        tkMessageBox.showerror("warning", "请输入样本数据")
    elif v2.get() == "":
        tkMessageBox.showerror("warning", "请输入影像数据")
    elif v4.get() == 0:
        tkMessageBox.showerror("warning", "请输入分类数量")
    elif v5.get() == 0:
        tkMessageBox.showerror("warning", "请输入波段数量")
    root.destroy()


# 事件2
def cancel():
    root.destroy()
    sys.exit(1)


# GUI
root = Tk()  # create
root.title("Random Forest")
root.geometry("520x320")
pass
frame1 = Frame(root)
label1 = Label(frame1, text="输入样本（ROI）:")
label1.pack(padx=10, pady=10, side=LEFT)
var1 = StringVar()
textbox1 = Entry(frame1, textvariable=var1)
var1.set("E:\\mzy\\python\\data\\roi.txt")
textbox1.pack(pady=10, side=LEFT)
frame1.pack(side=TOP)
pass
frame2 = Frame(root)
label2 = Label(frame2, text="输入影像数据：")
label2.pack(padx=10, pady=10, side=LEFT)
var2 = StringVar()
textbox2 = Entry(frame2, textvariable=var2)
var2.set("E:\\mzy\\python\\data\\test.tif")
textbox2.pack(pady=10, side=LEFT)
frame2.pack(side=TOP)
pass
frame3 = Frame(root)
label3 = Label(frame3, text="输入精度验证数据：")
label3.pack(padx=10, pady=10, side=LEFT)
var3 = StringVar()
textbox3 = Entry(frame3, textvariable=var3)
textbox3.pack(pady=10, side=LEFT)
frame3.pack(side=TOP)
pass
frame4 = Frame(root)
label4 = Label(frame4, text="分类数量：")
label4.pack(padx=10, pady=10, side=LEFT)
var4 = IntVar()
textbox4 = Entry(frame4, textvariable=var4)
var4.set(12)
textbox4.pack(pady=10, side=LEFT)
frame4.pack(side=TOP)
pass
frame5 = Frame(root)
label5 = Label(frame5, text="波段数量：")
label5.pack(padx=10, pady=10, side=LEFT)
var5 = IntVar()
textbox5 = Entry(frame5, textvariable=var5)
var5.set(64)
textbox5.pack(pady=10, side=LEFT)
frame5.pack(side=TOP)
pass
frame7 = Frame(root)
label7 = Label(frame7, text="输出影像数据：")
label7.pack(padx=10, pady=10, side=LEFT)
var7 = StringVar()
textbox7 = Entry(frame7, textvariable=var7)
var7.set("E:\\mzy\\python\\data\\result.tif")
textbox7.pack(pady=10, side=LEFT)
frame7.pack(side=TOP)
pass
frame6 = Frame(root)
button4 = Button(frame6, text="确定", command=lambda: carryout(var1, var2, var4, var5))  # 需要插入事件1
button4.pack(padx=80, pady=10, side=LEFT)
button5 = Button(frame6, text="取消", command=cancel)  # 需要插入事件2
button5.pack(padx=80, pady=10, side=LEFT)
frame6.pack(side=TOP)
pass
root.protocol("WM_DELETE_WINDOW", cancel)
root.mainloop()  # 消息循环

# path1 = "E:\\mzy\\python\\data\\roi.txt"
path1 = var1.get()
f = open(path1)
classNum = var4.get()
bandNum = var5.get()

# 读取样本各类别像素
pixCount = []
for i in range(0, classNum + 3):
    line = f.readline()
    if i in range(3, classNum + 3):
        pixCount.append(int(line[7:]))

# 定义target，Y
Y = []
for i in range(0, len(pixCount)):
    for j in range(0, pixCount[i]):
        Y.append(i + 1)

# 定义predictor,X
# 预读所有DN值
data = []
band = []
line = f.readline()
while line:
    if line[0:12] == 'Histogram\tDN':
        if len(band) > 0:
            data.append(band)
            band = []  # 不能使用clear方法，会改变之前添加的值
        pass  # next line
        line = f.readline()
        index1 = line.find('\t')
        index2 = line.find('\t', index1 + 1)
        index3 = line.find('\t', index2 + 1)
        if (index2 - index1) > 1:
            for i in range(0, int(line[index2:index3])):
                band.append(int(line[index1:index2]))
        pass  # next next line
        line = f.readline()
        index1 = line.find('\t')
        index2 = line.find('\t', index1 + 1)
        index3 = line.find('\t', index2 + 1)
        if (index2 - index1) > 1:
            for i in range(0, int(line[index2:index3])):
                band.append(int(line[index1:index2]))
        line = f.readline()
    elif line[0] == '\t':
        index1 = line.find('\t')
        index2 = line.find('\t', index1 + 1)
        index3 = line.find('\t', index2 + 1)
        if (index2 - index1) > 1:
            for i in range(0, int(line[index2:index3])):
                band.append(int(line[index1:index2]))
        line = f.readline()
    else:
        line = f.readline()
if len(band) > 0:
    data.append(band)
    band = []

# 赋值到X
sample = []
X = []
summary = 0
for i in range(0, len(pixCount)):
    summary = i * bandNum
    for j in range(0, pixCount[i]):
        for k in range(0, bandNum):
            sample.append(data[k + summary][j])
        X.append(sample)
        sample = []

# list （转 numpy.array） 转 DataFrame
# arrayX = np.array(X)
train = pd.DataFrame(X, columns=[i + 1 for i in range(0, bandNum)])
# arrayY = np.array(Y)
target = pd.DataFrame(Y)
y, _ = pd.factorize(target[0])

# 清理内存
del data, band, sample, X, Y

# 读取遥感影像
RSdata = []  # 直接读取数据
test = []
# path2 = "E:\\mzy\\python\\data\\test.tif"
path2 = var2.get()
dataset = gdal.Open(path2, GA_ReadOnly)
if dataset is None:
    print "can not open" + path2
    sys.exit(1)
samples = dataset.RasterXSize  # 影像的列
lines = dataset.RasterYSize  # 影像的行
# 读取各波段影像数据
for i in range(0, bandNum):
    band = dataset.GetRasterBand(i + 1)
    band2Array = band.ReadAsArray()
    band1Array = band2Array.flatten()
    RSdata.append(band1Array)
RSdata_1 = np.array(RSdata)
RSdata_2 = np.transpose(RSdata_1)

for i in range(0, lines):
    test1 = RSdata_2[i * samples:i * samples + samples, ]
    test.append(pd.DataFrame(test1, columns=[i + 1 for i in range(0, bandNum)]))

# 清理内存
del RSdata, RSdata_1, RSdata_2

# 建立随机森林
RF = RandomForestClassifier(n_estimators=500)
RF.fit(train, y)

# 随机森林分类（多线程并行）
# 多线程
q = Queue.Queue()


def predicting(t_data, rf, id):
    result = [id]
    for n in range(0, len(t_data)):
        predict = rf.predict(t_data[n])
        result.append(predict)
        if id == 4:
            progress = n * 100.0 / len(t_data)
            print ("当前分类进度为： %.2f%%" % progress)
    q.put(result)


cursor = lines / 4
thread_data = []
for i in range(0, 4):
    start = i * cursor
    end = (i + 1) * cursor
    if i == 3:
        end = lines
    testdata = []
    for j in range(start, end):
        testdata.append(test[j])
    thread_data.append(testdata)
testdata1 = thread_data[0]
RF1 = copy.copy(RF)
t1 = threading.Thread(target=predicting, args=(testdata1, RF1, 1))
testdata2 = thread_data[1]
RF2 = copy.copy(RF)
t2 = threading.Thread(target=predicting, args=(testdata2, RF2, 2))
testdata3 = thread_data[2]
RF3 = copy.copy(RF)
t3 = threading.Thread(target=predicting, args=(testdata3, RF3, 3))
testdata4 = thread_data[3]
RF4 = copy.copy(RF)
t4 = threading.Thread(target=predicting, args=(testdata4, RF4, 4))
t1.start()
t2.start()
t3.start()
t4.start()
t1.join()
t2.join()
t3.join()
t4.join()

# 获取多线程返回值
thread_result = list()
imagedata = list()
while not q.empty():
    thread_result.append(q.get())
for item in thread_result:
    if type(item[0]) is types.IntType and item[0] == 1:
        print item[0]
        del item[0]
        imagedata = imagedata + item
        break
for i in range(1, 4):
    for item in thread_result:
        if type(item[0]) is types.IntType and item[0] == i + 1:
            print item[0]
            del item[0]
            imagedata = imagedata + item
            break

a = np.array(imagedata)
# np.savetxt("E:\\mzy\\python\\data\\image.txt", a, delimiter=',', fmt='%d')

# 结果输出
path3 = var7.get()
dirver = dataset.GetDriver()
outdataset = dirver.Create(path3, samples, lines, 1, GDT_Float32)
outBand = outdataset.GetRasterBand(1)
outBand.WriteArray(a, 0, 0)
# 设置坐标参照系
geoTransform = dataset.GetGeoTransform()
outdataset.SetGeoTransform(geoTransform)
proj = dataset.GetProjection()
outdataset.SetProjection(proj)
print ("分类完成！/n")