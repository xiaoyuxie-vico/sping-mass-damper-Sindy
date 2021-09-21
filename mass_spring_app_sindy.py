# -*- coding: utf-8 -*-

'''
@author: Xiaoyu Xie
@email: xiaoyuxie2020@u.northwestern.edu
@date: Sep, 2021
'''

import io

from derivative import dxdt
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
from numpy import linalg as LA
import pandas as pd
from scipy.linalg import svd
import streamlit as st
from sklearn.linear_model import Lasso


np.set_printoptions(precision=2)

matplotlib.use('agg')


def main():
    apptitle = 'Mass-Spring-Damper-ODE'
    st.set_page_config(
        page_title=apptitle,
        page_icon=':eyeglasses:',
        # layout='wide'
    )
    st.title('Discover spring constant and damping coefficient for spring-mass-damping system')

    # level 1 font
    st.markdown("""
        <style>
        .L1 {
            font-size:40px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # level 2 font
    st.markdown("""
        <style>
        .L2 {
            font-size:20px !important;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )
    #########################Objectives#########################

    # st.markdown('<p class="L1">Objectives</p>', unsafe_allow_html=True)

    list = np.loadtxt(open("src/k8.csv","rb"),delimiter=",",skiprows=1)
    X=list[:,2]
    t=list[:,0]
    #print(X)

    Xavg=np.mean(X, axis=0)
    B=X - Xavg

    st.markdown('<p class="L1">Recover ordinary differential equation (ODE)</p>', unsafe_allow_html=True)
    #recover ODE
    z=B/1000
    Z = np.stack((z), axis=-1)  # First column is x, second is y
    #print(X)

    #plot x

    # plt.rcParams["font.family"] = "Arial"
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_subplot(111)
    # ax.plot(t, z, '-', color='blue')
    # ax.set_xlabel("Time (s)", fontsize=20)
    # ax.set_ylabel("z", fontsize=20)
    # ax.tick_params(labelsize=20)
    # #ax.set_xlim(-0.05, 1.05)
    # #ax.set_ylim(-6.5, 2.5)

    # st.pyplot(fig)

    st.markdown('<p class="L2">Videos:</p>', unsafe_allow_html=True)
    str_1 = """[1. Camera 1 (k=8 N/m)](https://drive.google.com/file/d/1WcL1O7lfkz8xi4GQlT8x0rMrPIORAhw7/view?usp=sharing)"""
    st.markdown(str_1)
    str_2 = """[2. Camera 2 (k=20 N/m)](https://drive.google.com/file/d/10OhbvmiKqNfuLQmIpkJRmHm0lnsjMpu3/view?usp=sharing)"""
    st.markdown(str_2)

    global ratio
    st.markdown('<p class="L2">Select a ratio to show data:</p>',
                unsafe_allow_html=True)
    ratio = st.slider('', 0.01, 1.0, 1.0)
    show_length = int(t.shape[0] * ratio)-1
    fig = plt.figure()
    plt.plot(t[:show_length], X[:show_length])
    plt.xlabel('t', fontsize=30)
    plt.ylabel(f'z', fontsize=30)
    # plt.title(f'x (Camera {chosen_line})', fontsize=34)
    plt.tick_params(labelsize=26)
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


    st.markdown('<p class="L1">Calculate derivatives</p>', unsafe_allow_html=True)

    z_dot= dxdt(z, t, kind="finite_difference", k=1)
    #z_dot=dxdt(z, t, kind="trend_filtered", order=3, alpha=0.1)
    z_2dot=dxdt(z_dot, t, kind="finite_difference", k=1)

    Z_2dot = np.stack((z_2dot), axis=-1)  # First column is x, second is y
    #print(Z_dot)

    col1, col2 = st.beta_columns(2)
    with col1:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(t[:show_length], z_dot[:show_length], '-', color='blue')
        ax.set_xlabel("Time (s)",fontsize=20)
        ax.set_ylabel(r"$\dot{z} $", fontsize=30)
        ax.tick_params(labelsize=20)
        #ax.set_xlim(-0.05, 1.05)
        #ax.set_ylim(-6.5, 2.5)
        st.pyplot(fig, clear_figure=True)

    with col2:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        ax.plot(t[:show_length], z_2dot[:show_length], '-', color='blue')
        ax.set_xlabel("Time (s)",fontsize=20)
        ax.set_ylabel(r"$\ddot{z} $", fontsize=30)
        ax.tick_params(labelsize=20)
        #ax.set_xlim(-0.05, 1.05)
        #ax.set_ylim(-6.5, 2.5)
        st.pyplot(fig, clear_figure=True)


    theta1 = z
    theta2 = z_dot

    THETA = np.stack((theta1,theta2), axis=-1)  # First column is x, second is y
    # print(THETA)

    model = Lasso(alpha=1e-8, max_iter=200, fit_intercept=False)
    model.fit(THETA, Z_2dot)

    st.markdown('<p class="L1">Sparse identification of dynamical systems</p>', unsafe_allow_html=True)

    r_sq = model.score(THETA, Z_2dot)
    st.markdown('<p class="L2">Fitting performance: the coefficient of determination is {}</p>'.format(
    round(r_sq, 2)), unsafe_allow_html=True)

    m=0.1
    k=-model.coef_[0]*m
    c=-model.coef_[1]*m

    # print('Spring constant k=',k, 'Damping coefficient c=',c)

    st.markdown('<p class="L2">Recovered spring constant k: {} N/m</p>'.format(round(k, 4)), unsafe_allow_html=True)
    st.markdown('<p class="L2">Recovered damping coefficient c: {} Ns/m</p>'.format(
        round(c, 4)), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
