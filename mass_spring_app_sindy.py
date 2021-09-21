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
    st.title('Discover spring constant and damping coefficient for spring-mass-damper system')

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

    st.markdown('<p class="L2">developed by Xiaoyu Xie and Zhengtao Gan.</p>', unsafe_allow_html=True)
    #########################Objectives#########################

    st.markdown('<p class="L1">Motion data:</p>', unsafe_allow_html=True)
    st.markdown('<p class="L2">Videos:</p>', unsafe_allow_html=True)
    str_1 = """[1. Video 1 (k=8 N/m)](https://drive.google.com/file/d/1WcL1O7lfkz8xi4GQlT8x0rMrPIORAhw7/view?usp=sharing)"""
    st.markdown(str_1)
    str_2 = """[2. Video 2 (k=20 N/m)](https://drive.google.com/file/d/10OhbvmiKqNfuLQmIpkJRmHm0lnsjMpu3/view?usp=sharing)"""
    st.markdown(str_2)

    st.markdown('<p class="L2">Extracted motion data:</p>', unsafe_allow_html=True)
    str_3 = """[1. Motion data from Video 1 (k=8 N/m)](https://drive.google.com/file/d/1DLkB0IbYkyA1jkjjBpqtte_5aln2qJnK/view?usp=sharing)"""
    st.markdown(str_3)
    str_3 = """[2. Motion data from Video 2 (k=20 N/m)](https://drive.google.com/file/d/1DLkB0IbYkyA1jkjjBpqtte_5aln2qJnK/view?usp=sharing)"""
    st.markdown(str_3)



    st.markdown('<p class="L1">Load motion data:</p>', unsafe_allow_html=True)
    flag = ['Dataset from Video 1 (k=8 N/m)', 'Dataset from Video 2 (k=20 N/m)', 'New dataset']
    st.markdown('<p class="L2">Chosse a new dataset or use default dataset:</p>',
                unsafe_allow_html=True)
    use_new_data = st.selectbox('', flag, 1)

    # load dataset
    if use_new_data == 'New dataset':
        uploaded_file = st.file_uploader(
            'Choose a CSV file', accept_multiple_files=False)

    # button_dataset = st.button('Click once you have selected a dataset')
    # if button_dataset:
    # load dataset
    if use_new_data == 'New dataset':
        data = io.BytesIO(uploaded_file.getbuffer())
        df = pd.read_csv(data)
    elif use_new_data == 'Dataset from Video 1 (k=8 N/m)':
        file_path = 'src/k8.csv'
        # df = pd.read_csv(file_path)
        list = np.loadtxt(open(file_path,"rb"),delimiter=",",skiprows=1)
    elif use_new_data == 'Dataset from Video 2 (k=20 N/m)':
        file_path = 'src/k20.csv'
        # df = pd.read_csv(file_path)
        list = np.loadtxt(open(file_path,"rb"),delimiter=",",skiprows=1)


    # list = np.loadtxt(open("src/k8.csv","rb"),delimiter=",",skiprows=1)
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
    
    global ratio
    st.markdown('<p class="L2">Select a ratio to show data:</p>',
                unsafe_allow_html=True)
    ratio = st.slider('', 0.01, 1.0, 1.0)
    show_length = int(t.shape[0] * ratio)-1
    fig = plt.figure()
    plt.plot(t[:show_length], X[:show_length])
    plt.xlabel('t', fontsize=30)
    plt.ylabel(f'z', fontsize=30)
    # plt.title(f'x (Video {chosen_line})', fontsize=34)
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

    st.markdown('<p class="L2">Set the mass:</p>', unsafe_allow_html=True)
    m = float(st.text_input('', 0.1))
    k=-model.coef_[0]*m
    c=-model.coef_[1]*m

    # print('Spring constant k=',k, 'Damping coefficient c=',c)

    st.markdown('<p class="L2">Recovered spring constant k: {} N/m</p>'.format(round(k, 4)), unsafe_allow_html=True)
    st.markdown('<p class="L2">Recovered damping coefficient c: {} Ns/m</p>'.format(
        round(c, 4)), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
