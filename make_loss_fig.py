import matplotlib.pyplot as plt

import numpy as np
import torch


def get_mean(l):
    leng = len(l)
    lnew = []
    a = 0
    for i in range(leng):
        a += l[i]
        lnew.append(a/(i+1))
    return lnew


def sum_list(l1, l2):
    l = []
    for i in range(len(l1)):
        a = l1[i] + l2[i]
        l.append(a)

    return l

def make_loss_s_edge(a=0,b=50):
    y1_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1for_loss_fig\loss_s.pt')
    y3_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1for_loss_fig_edge\loss_s.pt')

    y1_l = y1_l[a:24500:b]
    y3_l = y3_l[a:24500:b]
    y1_l = get_mean(y1_l)
    y3_l = get_mean(y3_l)

    x = np.arange(a,24500,b)

    fig = plt.figure(figsize=(6,4), dpi=250)
    axes = plt.subplot(111)
    axes.plot(x, y1_l, c='red', linewidth=1, label = 'Full model')
    axes.plot(x, y3_l, c='blue',linewidth=1, label = 'w/o EIS Module')
    plt.legend(loc=1)
    plt.xlabel('Iteration', fontdict={ "size":10})
    plt.ylabel('Style Loss', fontdict={"size":10})
    plt.grid()

    plt.xticks(np.linspace(0,25000,6))
    plt.yticks(np.linspace(0,350,8))
    plt.show()


def make_loss_c_edge(a=0,b=50):
    y1_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1for_loss_fig\loss_c.pt')
    y3_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1for_loss_fig_edge\loss_c.pt')

    y1_l = y1_l[a:24500:b]
    y3_l = y3_l[a:24500:b]
    y1_l = get_mean(y1_l)
    y3_l = get_mean(y3_l)

    x = np.arange(a,24500,b)

    fig = plt.figure(figsize=(6,4), dpi=250)
    axes = plt.subplot(111)
    axes.plot(x, y1_l, c='red', linewidth=1, label = 'Full model')
    axes.plot(x, y3_l, c='blue',linewidth=1,label = 'w/o EIS Module')
    plt.legend(loc=1)
    plt.xlabel('Iteration', fontdict={ "size":10})
    plt.ylabel('Content Loss', fontdict={"size":10})
    plt.grid()

    plt.xticks(np.linspace(0,25000,6))
    plt.yticks(np.linspace(0,20,11))

    plt.show()


def make_loss_s_sa():
    y1_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1for_loss_fig\loss_s.pt')
    y2_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1wosanet2\loss_s.pt')

    y1_l = y1_l[0:24500:50]
    y2_l = y2_l[0:24500:50]
    y1_l = get_mean(y1_l)
    y2_l = get_mean(y2_l)
    x = np.arange(0,24500,50)

    fig = plt.figure(figsize=(6,4), dpi=250)
    axes = plt.subplot(111)
    axes.plot(x, y1_l, c='red', linewidth=1, label = 'Full model')
    axes.plot(x, y2_l, c='blue',linewidth=1, label = 'w/o Base Style Transfer')
    plt.legend(loc=1)
    plt.xlabel('Iteration', fontdict={ "size":10})
    plt.ylabel('Style Loss', fontdict={"size":10})
    plt.grid()

    plt.xticks(np.linspace(0,25000,6))
    plt.yticks(np.linspace(0,800,9))



    plt.show()


def make_loss_c_sa():
    y1_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1for_loss_fig\loss_c.pt')
    y2_l = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1for_loss_fig_sanet\loss_c.pt')

    y1_l = y1_l[0:24500:50]
    y2_l = y2_l[0:24500:50]
    y1_l = get_mean(y1_l)
    y2_l = get_mean(y2_l)

    x = np.arange(0,24500,50)



    fig = plt.figure(figsize=(6,4), dpi=250)
    axes = plt.subplot(111)
    axes.plot(x, y1_l, c='red', linewidth=1, label = 'Full model')
    axes.plot(x, y2_l, c='blue',linewidth=1,label = 'w/o Base Style Transfer')

    plt.legend(loc=1)
    plt.xlabel('Iteration', fontdict={ "size":10})
    plt.ylabel('Content Loss', fontdict={"size":10})
    plt.grid()

    plt.xticks(np.linspace(0,25000,6))
    plt.yticks(np.linspace(0,20,11))

    plt.show()

def make_loss_total():
    y1_l_c = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1e2\loss_c.pt')
    y2_l_c = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1wosanet2\loss_c.pt')
    y3_l_c = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1wooutedge1\loss_c.pt')

    y1_l_s = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1e2\loss_s.pt')
    y2_l_s = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1wosanet2\loss_s.pt')
    y3_l_s = torch.load(r'F:\pyproject\lap_cin\saved_loss_wosanet\026_1wooutedge1\loss_s.pt')

    y1_l = sum_list(y1_l_s, y1_l_c)
    y2_l = sum_list(y2_l_s, y2_l_c)
    y3_l = sum_list(y3_l_s, y3_l_c)

    y1_l = y1_l[0:24500:5]
    y2_l = y2_l[0:24500:5]
    y3_l = y3_l[0:24500:5]

    y1_l = get_mean(y1_l)
    y2_l = get_mean(y2_l)
    y3_l = get_mean(y3_l)



    x = np.arange(0,24500,5)



    fig = plt.figure(figsize=(6,4), dpi=250)
    axes = plt.subplot(111)
    axes.plot(x, y1_l, c='red', linewidth=1, label = 'Full model')
    axes.plot(x, y2_l, c='blue',linewidth=1,label = 'w/o Base Network')
    axes.plot(x, y3_l, c='green',linewidth=1,label = 'w/o EIS Module')
    plt.legend(loc=1)
    plt.xlabel('Iteration', fontdict={ "size":10})
    plt.ylabel('Total Loss', fontdict={"size":10})
    plt.grid()

    plt.xticks(np.linspace(0,24500,5))
    plt.yticks(np.linspace(0,1000,11))
    plt.show()

#make_loss_s_edge()
#make_loss_c_edge()
#make_loss_s_sa()
make_loss_c_sa()


