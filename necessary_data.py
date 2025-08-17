# This file is for getting necessary data for model
import numpy as np
import scipy

# bed_template
u1 = np.array([0,30,60,120,150,180,240,420])
v1 = np.array([30,60,120,150,180,240,420,600])
u2 = np.array([15,60,120,180,300,330,450,510])
v2 = np.array([45,120,180,300,330,450,510,540])
u3 = np.array([15,75,105,135,165,285,465])
v3 = np.array([45,105,135,165,285,465,585])
u4 = np.array([30,75,105,135,195])
v4 = np.array([60,105,135,195,555])
u5 = np.array([30,150,180,360])
v5 = np.array([150,180,360,600])
u6 = np.array([45,225,405])
v6 = np.array([225,405,525])
u7 = np.array([45,285,345,375,405])
v7 = np.array([285,345,375,405,525])
u8 = np.array([90,120,180,480])
v8 = np.array([120,180,480,510])
u9 = np.array([90,270,300,330])
v9 = np.array([270,300,330,570])
u10 = np.array([105,225,465,495])
v10 = np.array([225,465,495,525])
u11 = np.array([120,240,420])
v11 = np.array([240,420,600])
u12 = np.array([120,300])
v12 = np.array([300,540])
u13 = np.array([120,420])
v13 = np.array([420,540])
u14 = np.array([135,315])
v14 = np.array([315,555])

u = [u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14]
v = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14]

# appointments that need to assigned
day1 = [15,9,8,5,8,2,0]
day2 = [21,8,5,10,7,2,0]
day3 = [17,10,7,7,9,2,0]
day4 = [15,10,9,5,5,0,2]
day5 = [14,8,10,8,5,3,0]
day6 = [15,13,8,8,6,4,1]
day7 = [20,11,8,7,3,4,0]
day8 = [14,7,8,6,2,3,1]
day9 = [17,13,6,5,3,1,1]
day10 = [18,11,7,6,3,3,2]
day11 = [23,12,9,4,5,0,2]
day12 = [12,8,7,7,6,1,2]
day13 = [24,10,10,7,7,1,3]
day14 = [24,10,12,10,4,1,2]
day15 = [20,9,12,6,6,2,1]
day16 = [21,16,11,4,6,1,2]
day17 = [17,5,7,8,7,1,2]
day18 = [14,9,8,7,5,2,1]
day19 = [14,9,13,3,7,5,1]
day20 = [20,11,6,8,9,1,0]
day21 = [24,8,11,10,5,3,1]
day22 = [20,8,11,5,8,0,1]
days = [day1,day2,day3,day4,day5,day6,day7,day8,day9,day10,day11,day12,day13,day14,day15,day16,day17,day18,day19,day20,day21,day22]

# data related to random service time
data = scipy.io.loadmat("../clean_code/data.mat")