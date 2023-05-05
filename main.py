#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 23:36:42 2022
@author: kurup
"""
import numpy as np
import math
from matplotlib import pyplot as plt
###############################################################################
def get_losses(no_ite):
    temp=[]
    for i in range(no_ite):
        temp.append(np.array([np.random.normal(loc=1,scale=1,size=3)]).T)
    return temp
###############################################################################
def projection_unit(vec):
    if np.linalg.norm(vec)>1:
        vec=vec/np.linalg.norm(vec)
    return vec

def take_step(W_t,eta,loss_func1,loss_func2,is_lazy):
    W_t_1=W_t-eta*loss_func2
    proj_W_t_1=projection_unit(W_t_1) 
    loss_val=proj_W_t_1.T[0]@loss_func1.T[0]
    if not is_lazy:  # goes inside for Active and copies projection value to W_t_1
        W_t_1=proj_W_t_1
    return loss_val,W_t_1

def onl_grad_dec(W_t,loss_func,T,update_eta=True,is_lazy=True):
    eta=0.01
    loss_vals=[0.0] #considering W_1 as[[0,0,0]]
    indi_loss=[]
    for i in range(2,T+1):
        if update_eta:
            eta=1/(math.sqrt(i))
        immi_loss,W_t=take_step(W_t,eta,loss_func[i-1],loss_func[i-2],is_lazy)
        loss_vals.append(loss_vals[-1]+immi_loss)
        indi_loss.append(immi_loss)
    return loss_vals,indi_loss
###############################################################################
def plot_graph(x_len,err_list):
    L_C={
        0:['Lazy + Variable eta','r'],
        1:['Lazy + Const eta','g'],
        2:['Active + Variable eta','b'],
        3:['Active + Const eta','y']}
    for i in range(len(err_list)):
        plt.plot(np.arange(x_len),err_list[i],label=L_C[i][0],color=L_C[i][1])
    plt.legend(loc='best')
    plt.show()
###############################################################################
def main():
    is_lazy=True
    update_eta=True
    no_ite=10000
    loss_func=get_losses(no_ite)
    W_t_init=np.array([[0],[0],[0]])
    err_list=[]
    choices=[[update_eta,is_lazy],
                 [not update_eta,is_lazy],
                 [update_eta,not is_lazy],
                 [not update_eta,not is_lazy]]
    for i in choices:
        err_list.append(onl_grad_dec(W_t_init,loss_func,no_ite,i[0],i[1])[0])
    x_len=len(err_list[0])
    plot_graph(x_len,err_list)

if __name__ == '__main__':
  main()