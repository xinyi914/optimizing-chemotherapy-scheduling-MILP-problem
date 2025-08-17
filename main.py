import gurobipy as gp
from gurobipy import GRB
import scipy.io
import numpy as np
import math
from collections import defaultdict
from openpyxl import Workbook, load_workbook
import subprocess
import os
import time
import matplotlib.pyplot as plt

import utils
from necessary_data import u,v,days,data



B = 14
# get mu, variances, and covariance of the service time
mu = data["means"][0]
mu = mu.T
variances = data["variances"][0]
cov = np.diag(variances)

# get the number of time slots of each bed
mb = [len(u[0]),len(u[1]),len(u[2]),len(u[3]),len(u[4]),len(u[5]),len(u[6]),len(u[7]),len(u[8]),len(u[9]),
      len(u[10]),len(u[11]),len(u[12]),len(u[13])]

T = []
# for i in range(len(u)):
#     m.append(len(u[i]))
for i in range(len(v)):
    T.append(v[i][-1])
T = np.array(T)

M = []
for i in range(B):
    M.append(10000)

# train first step
def train_first_step(gb_no_set,fs_no_set,determined=False):
    if not determined:
        for gb_set in gb_no_set:
            for fs_set in fs_no_set:
                print(f"first step param_set: {gb_set}, {fs_set}")
                gb_par_set,fs_par_set,sc_par_set = utils.get_par_set(gb_set,fs_set,2)

                for i in range(len(days)):
                    instances = f"day{i+1}"
                    f ,beta ,theta ,delta = utils.first_step(instances,days[i],v,u,
                                                            gb_par_set['mu'],gb_par_set['cov'],
                                                            fs_par_set['alpha'],fs_par_set['weight_delta'],
                                                            fs_par_set['weight_f'], fs_par_set['weight_delta_kb'],
                                                            gb_par_set['gb_set_no'],fs_par_set['fs_set_no'])
    else:
        for gb_set in gb_no_set:
            for fs_set in fs_no_set:
                print(f"first step param_set: {gb_set}, {fs_set}")
                gb_par_set,fs_par_set,sc_par_set = utils.get_par_set(gb_set,fs_set,2)

                for i in range(len(days)):
                    instances = f"day{i+1}"
                    f ,beta ,theta ,delta = utils.first_step_determined(instances,days[i],v,u,
                                                            gb_par_set['mu'],gb_par_set['cov'],
                                                            fs_par_set['alpha'],fs_par_set['weight_delta'],
                                                            fs_par_set['weight_f'], fs_par_set['weight_delta_kb'],
                                                            gb_par_set['gb_set_no'],fs_par_set['fs_set_no'])



# train second step
def train_second_step(gb_no_set,fs_no_set,sc_no_set,determined=False):
    for gb_set in gb_no_set:
        for fs_set in fs_no_set:
            for sc_set in sc_no_set:
                print(f"param_set: {gb_set}, {fs_set}, {sc_set}")
                gb_par_set,fs_par_set,sc_par_set = utils.get_par_set(gb_set,fs_set,sc_set)

                ins = ["day1","day2","day3","day4","day5","day6","day7","day8","day9","day10",
                    "day11","day12","day13","day14","day15","day16","day17","day18","day19","day20",
                    "day21","day22"]
                
                for i in range(len(ins)):
                    instances = ins[i]
                    if not determined:
                        fs_path = f"model3/firstStep/result/{instances}_alpha{fs_par_set['alpha']}_wd{fs_par_set['weight_delta']}_wf{fs_par_set['weight_f']}_wdk{fs_par_set['weight_delta_kb']}_set{gb_par_set['gb_set_no']}-{fs_par_set['fs_set_no']}.npz"
                    else:
                        fs_path = f"model3/firstStep/result/{instances}_alpha{fs_par_set['alpha']}_wd{fs_par_set['weight_delta']}_wf{fs_par_set['weight_f']}_wdk{fs_par_set['weight_delta_kb']}_set{1}-{1}.npz"

                    fs_result = np.load(fs_path)
                    f_arr = fs_result["f"]
                    bed_appos = f_arr.T

                    for b in range(gb_par_set["B"]):
                        print("bed ",b)
                        if np.sum(bed_appos[b]) != 0:
                            utils.second_step(instances,b,bed_appos[b],mb[b],T[b],u[b],v[b],M[b],
                                        gb_par_set["mu"],gb_par_set["variances"],
                                        sc_par_set["wc"],sc_par_set["ic"],sc_par_set["wc_out"],
                                        sc_par_set["lb"],sc_par_set["ub"],
                                        gb_par_set["gb_set_no"],fs_par_set["fs_set_no"],sc_par_set["sc_set_no"],
                                        sc_par_set["y_weight"],sc_par_set["z_weight"],sc_par_set["x_weight"],
                                        e_weight=sc_par_set["e_weight"],nu_lb = sc_par_set["nu_lb"],nu_ub = sc_par_set["nu_ub"],
                                        determined=determined)
                        else:
                            d = []
                            assin = []
                            s = [0]
                            t = [0]
                            x = [0]
                            y = [0]
                            z = [0]
                            if not determined:
                                np.savez(f"model3/secondStep/result/{instances}b{b}_wc{sc_par_set['wc']}_ic{sc_par_set['ic']}_wco{sc_par_set['wc_out']}_lb{sc_par_set['lb']}_ub{sc_par_set['ub']}_ewei{sc_par_set['e_weight']}_set{gb_par_set['gb_set_no']}-{fs_par_set['fs_set_no']}-{sc_par_set['fs_set_no']}_updated.npz",
                         s=s,t=t,d=d,z=z,y=y,x=x,app = assin)
                            else:
                                np.savez(f"model3/secondStep/result/{instances}b{b}_wc{sc_par_set['wc']}_ic{sc_par_set['ic']}_wco{sc_par_set['wc_out']}_lb{sc_par_set['lb']}_ub{sc_par_set['ub']}_ewei{sc_par_set['e_weight']}_set2-1-{sc_set}_updated_determined.npz",
                         s=s,t=t,d=d,z=z,y=y,x=x,app = assin)


# test the model
def test(gb_no_set,fs_no_set,sc_no_set,simulate_times=1000,rand_seed=42,out=True,determined=False):
    
    if out:
        da_30 = utils.generate_out_sample_data(mu[0],variances[0],10000,rand_seed = 42)
        da_60 = utils.generate_out_sample_data(mu[1],variances[1],10000,rand_seed = 42)
        da_120 = utils.generate_out_sample_data(mu[2],variances[2],10000,rand_seed = 42)
        da_180 = utils.generate_out_sample_data(mu[3],variances[3],10000,rand_seed = 42)
        da_240 = utils.generate_out_sample_data(mu[4],variances[4],10000,rand_seed = 42)
        da_300 = utils.generate_out_sample_data(mu[5],variances[5],10000,rand_seed = 42)
        da_360 = utils.generate_out_sample_data(mu[6],variances[6],10000,rand_seed = 42)
        sample_data = [da_30,da_60,da_120,da_180,da_240,da_300,da_360]
    else:
        da_30_in = data["data_30_new"]
        da_60_in = data["data_60_new"]
        da_120_in = data["data_120_new"]
        da_180_in = data["data_180_new"]
        da_240_in = data["data_240_new"]
        da_300_in = data["data_300_new"]
        da_360_in = data["data_360_new"]
        sample_data = [da_30_in,da_60_in,da_120_in,da_180_in,da_240_in,da_300_in,da_360_in]


    for gb_no in gb_no_set:
        for fs_no in fs_no_set:
            for sc_no in sc_no_set:
                print(f"param_set: {gb_no}, {fs_no}, {sc_no}")
                utils.get_second_step_stat(B,mb,gb_no,fs_no,sc_no,simulate_times,rand_seed,out,sample_data,determined=determined)

# load second step stat
def test_result_to_excel(filename,col,row_start,gb_no_set,fs_no_set,sc_no_set,simulate_times=1000,rand_seed = 42,out=True,determined=False):
    # col = [2,10,18,26,None,34]
    # row_start = [4,31,58,85,112,139,166,193,220,247,
    #             274,301,328,355,382,409,436,463,490,
    #             517,544,571,598,625,652,679,706,733,760,787,814] 

    # out = False
    # col_increase = 0
    for gb_no in gb_no_set:
        for fs_no in fs_no_set:
            for sc_no in sc_no_set:
                print(f"param_set: {gb_no}, {fs_no}, {sc_no}")
                # simulate_times = 1000
                # rand_seed = 42
                gb_par_set,fs_par_set,sc_par_set = utils.get_par_set(gb_no,fs_no,sc_no)
                wait_times,idle_times,over_times,override_policy1s,override_policy2s,override_policy3s = utils.load_step2_stats(simulate_times,rand_seed,gb_par_set,fs_par_set,sc_par_set,out,determined=determined)
                column_idx = col[fs_no - 1]
        #         print(column_idx)
                start_row = row_start[sc_no - 2]
                utils.stats_to_excel(filename,wait_times,idle_times,over_times,
                                    override_policy1s,override_policy2s,override_policy3s,column_idx,start_row)

if __name__ == "__main__":
    # define gb_set, fs_set, sc_set
    gb_no_set = [2]
    fs_no_set = [1]
    sc_no_set = [33]

    # if want to train first step
    # print("solving first step")
    # train_first_step(gb_no_set,fs_no_set)

    # # if want to train second step
    # print("solving second step")
    # train_second_step(gb_no_set,fs_no_set,sc_no_set)

    # if want to train second step in determinstic case
    # print("solving determined second step")
    # train_second_step(gb_no_set,fs_no_set,sc_no_set,determined=True)

    # # if want to test
    # print("testing")
    # test(gb_no_set,fs_no_set,sc_no_set,simulate_times=1000,rand_seed=42,out=True)

    # if want to test determined case
    # print("testing")
    # test(gb_no_set,fs_no_set,sc_no_set,simulate_times=1000,rand_seed=42,out=True,determined=True)

    # ## if want to write the result to the excel
    # print("writing")
    # filename = "model3/secondStep/second_step_statistics.xlsx" # for out of sample
    # col = [2,10,18,26,None,34]
    # row_start = [4,31,58,85,112,139,166,193,220,247,
    #             274,301,328,355,382,409,436,463,490,
    #             517,544,571,598,625,652,679,706,733,760,787,814,841]  
    # test_result_to_excel(filename,col,row_start,gb_no_set,fs_no_set,sc_no_set,simulate_times=1000,rand_seed = 42,out=True)

    ## write in deterministic statistics
    # col = [2,10,18,26,None,34]
    col = [11] # d=False lb=3,ub=4
    row_start = [4,31,58,85,112,None,None,None,None,None,
             None,None,None,None,None,None,None,None,None,
            None,None,None,None,None,None,None,None,None,None,787,None,816] 
    filename = "model3/secondStep/second_step_statistics_deterministic.xlsx" 
    test_result_to_excel(filename,col,row_start,gb_no_set,fs_no_set,sc_no_set,simulate_times=1000,rand_seed = 42,out=True,determined=True)

    ## if want to load second step result
    # updated = True
    # instances = "day1"
    # b = 1
    # gb_no=2
    # fs_no=1
    # sc_no=31
    # gb_par_set,fs_par_set,sc_par_set = utils.get_par_set(gb_no,fs_no,sc_no)
    # sc_result = utils.second_load_result(instances,b,sc_par_set['wc'],sc_par_set['ic'],sc_par_set['wc_out'],sc_par_set['lb'],sc_par_set['ub'],sc_par_set['e_weight'],gb_par_set['gb_set_no'],fs_par_set['fs_set_no'],sc_par_set['sc_set_no'],updated)
    # print("d ",sc_result["d"])

    ## if want to load first step result
    # gb_no=2
    # fs_no=1
    # app_list_count, app_counts, perc_whole, perc_30, perc_60, perc_120, perc_180, perc_240,perc_300,perc_360 = utils.get_fs_stat(gb_no,fs_no)
    # print(f'''first_step stats: (gb,fs): {gb_no},{fs_no}\n
    #             app_list_count: {app_list_count}\n
    #             app_counts: {app_counts}\n
    #             perc_whole: {perc_whole}\n
    #             perc_30: {perc_30}\n
    #             perc_60: {perc_60}\n
    #             perc_120: {perc_120}\n
    #             perc_180: {perc_180}\n
    #             perc_240: {perc_240}\n
    #             perc_300: {perc_300}\n
    #             perc_360: {perc_360}''')

    ## if want to load stats result
    # for gb_no in gb_no_set:
    #     for fs_no in fs_no_set:
    #         for sc_no in sc_no_set:
    #             print(f"param_set: {gb_no}, {fs_no}, {sc_no}")
    #             simulate_times = 1000
    #             rand_seed = 30
    #             out = True
    #             gb_par_set,fs_par_set,sc_par_set = utils.get_par_set(gb_no,fs_no,sc_no)
    #             wait_times,idle_times,over_times,override_policy1s,override_policy2s,override_policy3s = utils.load_step2_stats(simulate_times,rand_seed,gb_par_set,fs_par_set,sc_par_set,out)
    #             ins = ["day1","day2","day3","day4","day5","day6","day7","day8","day9","day10","day11","day12","day13","day14","day15","day16","day17","day18","day19","day20","day21","day22"]
    #             wt = []
    #             it = []
    #             ot = []
    #             for instance in ins:
    #                 wt.append(np.mean(wait_times[instance]))
    #                 it.append(np.mean(idle_times[instance]))
    #                 ot.append(np.mean(over_times[instance]))
    #             print(sum(wt)/len(wt))
    #             print(sum(it)/len(it))
    #             print(sum(ot)/len(ot))
