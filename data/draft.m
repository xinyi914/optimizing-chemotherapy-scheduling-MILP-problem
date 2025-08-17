data = readtable('generated_data.xlsx');
data_30 = data.data_30;
data_60 = data.data_60;
data_120 = data.data_120;
data_180 = data.data_180;
data_240 = data.data_240;
data_300 = data.data_300;
data_360 = data.data_360;

mean = [str2double(data_30{1});str2double(data_60{1});str2double(data_120{1});str2double(data_180{1});str2double(data_240{1});str2double(data_300{1});str2double(data_360{1})];
mu = mean;
var = [str2double(data_30{2}),str2double(data_60{2}),str2double(data_120{2}),str2double(data_180{2}),str2double(data_240{2}),str2double(data_300{2}),str2double(data_360{2})];
cov = diag(var);
f = [1 0;2 3;2 4;1 5;5 3;4 7;1 4];
sqrt(transpose(f(:,2))*cov*f(:,2))
norm(sqrt(cov)*f(:,2))