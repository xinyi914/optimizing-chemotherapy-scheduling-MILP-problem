# parameter sets
from necessary_data import data
import numpy as np
# global parameter sets
mu = data["means"][0]
mu = mu.T
variances = data["variances"][0]
cov = np.diag(variances)
global_par_set1 = {"K":7,"B":14,"patient_type":[30,60,120,180,240,300,360],"mu":np.array([30,60,120,180,240,300,360]),"variances":np.array([1,1,1,1,1,1,1]),"cov":np.eye(7),"gb_set_no":1}
global_par_set2 = {"K":7,"B":14,"patient_type":[30,60,120,180,240,300,360],"mu":mu,"variances":variances,"cov":cov,"gb_set_no":2}

# first-step parameter sets
fs_par_set1 = {"alpha":0.05,"weight_f":100,"weight_delta":1.1,"weight_delta_kb":10,"fs_set_no":1}
fs_par_set2 = {"alpha":0.1,"weight_f":100,"weight_delta":1.1,"weight_delta_kb":10,"fs_set_no":2}
fs_par_set3 = {"alpha":0.2,"weight_f":100,"weight_delta":1.1,"weight_delta_kb":10,"fs_set_no":3}
fs_par_set4 = {"alpha":0.5,"weight_f":100,"weight_delta":1.1,"weight_delta_kb":10,"fs_set_no":4}
fs_par_set5 = {"alpha":0.05,"weight_f":100,"weight_delta":8,"weight_delta_kb":10,"fs_set_no":5}
fs_par_set6 = {"alpha":0.8,"weight_f":100,"weight_delta":1.1,"weight_delta_kb":10,"fs_set_no":6}
fs_par_set7 = {"alpha":1.0,"weight_f":100,"weight_delta":1.1,"weight_delta_kb":10,"fs_set_no":7}
fs_par_set8 = {"alpha":0.9,"weight_f":100,"weight_delta":1.1,"weight_delta_kb":10,"fs_set_no":8}

# second-step parameter sets
sc_par_set1 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000000,"nu_lb":-1000000,"e_weight":1,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":1}

sc_par_set2 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":2}

sc_par_set3 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.05,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":3}
sc_par_set4 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.03,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":4}
sc_par_set5 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.01,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":5}
sc_par_set6 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.005,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":6}


sc_par_set7 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":1.2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":7}
sc_par_set8 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":1.01,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":8}

sc_par_set9 = {"lb":2,"ub":3,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":9}

sc_par_set10 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":2,"ic":1,"wc_out":20,
           "y_weight":4,"z_weight":1,"x_weight":50,"sc_set_no":10}

sc_par_set11 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":0.8,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":11}

sc_par_set12 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":1.0001,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":12}

sc_par_set13 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":0.5,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":13}

sc_par_set14 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":0.1,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":14}

sc_par_set15 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.1,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":15}

sc_par_set16 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.001,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":16}

sc_par_set17 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.001,"ic":2,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":17}

sc_par_set18 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.001,"ic":5,"wc_out":100,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":18}

sc_par_set19 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.001,"ic":20,"wc_out":1000,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":19}

sc_par_set20 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.001,"ic":20,"wc_out":100,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":20}

sc_par_set21 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.0001,"ic":5,"wc_out":100,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":21}

sc_par_set22 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.00001,"ic":5,"wc_out":100,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":22}

sc_par_set23 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.000001,"ic":5,"wc_out":100,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":23}

sc_par_set24 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.00001,"wc":0.001,"ic":20,"wc_out":100,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":24}

sc_par_set25 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.03,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":5,"sc_set_no":25}

sc_par_set26 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.03,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":10,"sc_set_no":26}

sc_par_set27 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.03,"wc":2,"ic":1,"wc_out":20,
           "y_weight":2,"z_weight":1,"x_weight":1,"sc_set_no":27}

sc_par_set28 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.03,"wc":2,"ic":1,"wc_out":20,
           "y_weight":5,"z_weight":1,"x_weight":5,"sc_set_no":28}

sc_par_set29 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":5,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":29}

sc_par_set30 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":1,"wc":0.001,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":30}

sc_par_set31 = {"lb":3,"ub":4,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":31}

sc_par_set32 = {"lb":2,"ub":3,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0.0001,"wc":0.001,"ic":20,"wc_out":1000,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":32}
sc_par_set33 = {"lb":2,"ub":3,"epsilon":0.05,"nu_ub":1000,"nu_lb":-1000,"e_weight":0,"wc":2,"ic":1,"wc_out":20,
           "y_weight":1,"z_weight":1,"x_weight":1,"sc_set_no":33}



gb_par_sets = [global_par_set1,global_par_set2]
fs_par_sets = [fs_par_set1,fs_par_set2,fs_par_set3,fs_par_set4,fs_par_set5,fs_par_set6]
sc_par_sets = [sc_par_set1,sc_par_set2,sc_par_set3,sc_par_set4,sc_par_set5,sc_par_set6,sc_par_set7,sc_par_set8,sc_par_set9,sc_par_set10,
               sc_par_set11,sc_par_set12,sc_par_set13,sc_par_set14,sc_par_set15,sc_par_set16,sc_par_set17,sc_par_set18,sc_par_set19,sc_par_set20,
               sc_par_set21,sc_par_set22,sc_par_set23,sc_par_set24,sc_par_set25,sc_par_set26,sc_par_set27,sc_par_set28,sc_par_set29,sc_par_set30,
               sc_par_set31,sc_par_set32,sc_par_set33]