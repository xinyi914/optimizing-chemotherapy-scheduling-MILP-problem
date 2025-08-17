% % rng(42);
% %remove lower than x/3 and higher than 4x
% n = 1000;
% pick = 900;
% type = [30;60;120;180;240;300;360];
% low_bound = 3;
% up_bound = 4;
% 
% % generate data for patient type 30
% m=56;
% v=35^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% data_30 = lognrnd(mu,sigma,[n,1]);
% data_30_new = data_30(data_30 >= 30/low_bound & data_30 <= up_bound*30);
% data_30_new = data_30_new(1:pick);
% % length(data_30)
% % length(data_30_new)
% mean_30 = mean(data_30_new)
% variance_30 = var(data_30_new)
% sd_30 = std(data_30_new)
% 
% % generate data for patient type 60
% m = 83;
% v= 44^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% data_60 = lognrnd(mu,sigma,[n,1]);
% data_60_new = data_60(data_60 >= 60/low_bound & data_60 <= up_bound*60);
% data_60_new = data_60_new(1:pick);
% 
% % length(data_60)
% % length(data_60_new)
% mean_60 = mean(data_60_new)
% variance_60 = var(data_60_new)
% sd_60 = std(data_60_new)
% 
% % generate data for patient type 120
% m = 142;
% v= 44^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% data_120 = lognrnd(mu,sigma,[n,1]);
% data_120_new = data_120(data_120 >= 120/low_bound & data_120 <= up_bound*120);
% data_120_new = data_120_new(1:pick);
% % length(data_120)
% % length(data_120_new)
% mean_120 = mean(data_120_new)
% variance_120 = var(data_120_new)
% sd_120 = std(data_120_new)
% 
% % generate data for patient type 180
% m = 194;
% v= 51^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% data_180 = lognrnd(mu,sigma,[n,1]);
% data_180_new = data_180(data_180 >= 180/low_bound & data_180 <= up_bound*180);
% data_180_new = data_180_new(1:pick);
% % length(data_180)
% % length(data_180_new)
% mean_180 = mean(data_180_new)
% variance_180 = var(data_180_new)
% sd_180 = std(data_180_new)
% 
% % generate data for patient type 240
% m = 236;
% v= 60^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% 
% data_240 = lognrnd(mu,sigma,[n,1]);
% data_240_new = data_240(data_240 >= 240/low_bound & data_240 <= up_bound*240);
% data_240_new = data_240_new(1:pick);
% 
% % length(data_240)
% % length(data_240_new)
% mean_240 = mean(data_240_new)
% variance_240 = var(data_240_new)
% sd_240 = std(data_240_new)
% 
% % generate data for patient type 300
% m = 269;
% v= 68^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% data_300 = lognrnd(mu,sigma,[n,1]);
% data_300_new = data_300(data_300 >= 300/low_bound & data_300 <= up_bound*300);
% data_300_new = data_300_new(1:pick);
% 
% % length(data_300)
% % length(data_300_new)
% mean_300 = mean(data_300_new)
% variance_300 = var(data_300_new)
% sd_300 = std(data_300_new)
% 
% % generate data for patient type 360
% m = 346;
% v= 67^2;
% mu = log((m^2)/sqrt(v+m^2));
% sigma = sqrt(log(v/(m^2)+1));
% data_360 = lognrnd(mu,sigma,[n,1]);
% data_360_new = data_360(data_360 >= 360/low_bound & data_360 <= up_bound*360);
% data_360_new = data_360_new(1:pick);
% % length(data_360)
% % length(data_360_new)
% mean_360 = mean(data_360_new)
% variance_360 = var(data_360_new)
% sd_360 = std(data_360_new)
% 
% 
% all = [data_30_new data_60_new data_120_new data_180_new data_240_new data_300_new data_360_new];
% 
% % cov_all = cov(all);
% % write generated data in to the excel sheet
% filename = "generated_data.xlsx";
% data = readtable(filename);
% data.data_30(1:n) = data_30;
% writetable(data,filename);
% 
% 
% % means = [mean_30 mean_60 mean_120 mean_180 mean_240 mean_300 mean_360]
% % variances = [variance_30 variance_60 variance_120 variance_180 variance_240 variance_300 variance_360]
% % sds = [sd_30 sd_60 sd_120 sd_180 sd_240 sd_300 sd_360]
% % head = ["data_30", "data_60", "data_120", "data_180", "data_240", "data_300", "data_360"]
% C = [head;means;variances;sds];
% C
% % writematrix(C,filename,'WriteMode','overwrite');
% 
% 
% filename = strcat('../clean_code/data.mat'); %file + # of bed + # of slots + # of appointments.mat
% save(filename)


% plot data
d = load('../clean_code/data.mat')
data30 = d.data_30_new;
data60 = d.data_60_new;
data120 = d.data_120_new;
data180 = d.data_180_new;
data240 = d.data_240_new;
data300 = d.data_300_new;
data360 = d.data_360_new;

%get median of the data
med_30 = median(data30)
med_60 = median(data60)
med_120 = median(data120)
med_180 = median(data180)
med_240 = median(data240)
med_300 = median(data300)
med_360 = median(data360)
%get the probability less than patient type
prob_30 = mean(data30 < 30)
prob_60 = mean(data60 < 60)
prob_120 = mean(data120 < 120)
prob_180 = mean(data180 < 180)
prob_240 = mean(data240 < 240)
prob_300 = mean(data300 < 300)
prob_360 = mean(data360 < 360)
% figure
% histogram(data30)
% xlabel("patient 30")
% ylabel("frequency")
% title("distribution of patient 30")
% 
% figure
% histogram(data60)
% xlabel("patient 60")
% ylabel("frequency")
% title("distribution of patient 60")
% 
% figure
% histogram(data120)
% xlabel("patient 120")
% ylabel("frequency")
% title("distribution of patient 120")
% 
% figure
% histogram(data180)
% xlabel("patient 180")
% ylabel("frequency")
% title("distribution of patient 180")
% 
% figure
% histogram(data240)
% xlabel("patient 240")
% ylabel("frequency")
% title("distribution of patient 240")
% 
% figure
% histogram(data300)
% xlabel("patient 300")
% ylabel("frequency")
% title("distribution of patient 300")
% 
% figure
% histogram(data360)
% xlabel("patient 360")
% ylabel("frequency")
% title("distribution of patient 360")
% 
% 
