% This file is for the first step in model 3
% **all codes for saving result to the file are commented

% record all results in txt file for one appointment
% dfile="diaryFile.txt";
% if exist(dfile,'file'); delete(dfile); end
% diary(dfile)
% diary on
tic;

% testing different list of appointments
% a=[30;30;30;30;30];
% a=[30;60;60;60;120;60;30;60;120];
% a=[30;30;30;30;30;30;30;30;30;30;30;30;30;30;30;30;30;30;30;30];
% a=[30;60;360];
% a=[120;120;180;360];
% a=[30;240;360];
% a = [120;120;180;180];
% a = [60;60;60;60;60;60;60;60;60;60];
% a = [30;30;60;30;30;120;120;180];
% a=[30;60;60;180;30];
% a=[180;180];
% a=[60;120;120;60;180]; % 5 appointments
% a=[30;60;60;60;120;60;30;60;120;30]; %10 appointments
% a=[30;30;60;180;180;120;60;30;30;60;60;120;240;120;180]; %15 apppointments
% a = [30;30;30;60;60;60;60;60;60;120;120;120;180;180;180;180;240;300;360;360]; %20 appointments
% a = [30;30;30;30;30;30;30;30;30;30;30;30;30;30;30;60;60;60;60;60;60;60;60;60;120;120;120;120;120;120;120;120;180;180;180;180;180;240;240;240;240;240;240;240;240;300;300];



% list of appointments on each day in the paper
p1 = [15;9;8;5;8;2;0];
p2 = [21;8;5;10;7;2;0];
p3 = [17;10;7;7;9;2;0];
p4 = [15;10;9;5;5;0;2];
p5 = [14;8;10;8;5;3;0];
p7 = [20;11;8;7;3;4;0];
p8 = [14;7;8;6;2;3;1];
p9 = [17;13;6;5;3;1;1];
p10 = [18;11;7;6;3;3;2];
p11 = [23;12;9;4;5;0;2];
p12 = [12;8;7;7;6;1;2];
p15 = [20;9;12;6;6;2;1];
p17 = [17;5;7;8;7;1;2];
p18 = [14;9;8;7;5;2;1];
p20 = [20;11;6;8;9;1;0];
p21 = [24;8;11;10;5;3;1];
p22 = [20;8;11;5;8;0;1];


p6 = [15;13;8;8;6;4;1];
p13 = [24;10;10;7;7;1;3];
p14 = [24;10;12;10;4;1;2];
p16 = [21;16;11;4;6;1;2];
p19 = [14;9;13;3;7;5;1];

% p_list = [p1 p2 p3 p4 p5 p6 p7 p8 p9 p10 p11 p12 p13 p14 p15 p16 p17 p18 p19 p20 p21 p22];
% p_list = [p15 p16 p17 p18 p19 p20 p22];
% p_list_str = getVarNames(p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22);
p_list = [p1];
% p_list_str = getVarNames(p13); %for saving the result of pi into the file(change when the p_list change)

% loop for each day and save results to txt file for each day
for p_idx = 1:size(p_list,2)

% str_p = sprintf('%s', p_list_str(p_idx));
% %save log file to the folder for each p
% dfile=strcat(pwd,'/variable_mat_files_deterministic/diary/',str_p,'.txt');
% if exist(dfile,'file'); delete(dfile); end
% diary(dfile)
% diary on
tic;

% transfer p to appointment list
p=p_list(:,p_idx)
num30=p(1); num60=p(2); num120=p(3); num180=p(4); num240=p(5); num300=p(6); num360=p(7);
a_30 = 30*ones(1,num30);
a_60 = 60*ones(1,num60);
a_120 = 120*ones(1,num120);
a_180 = 180*ones(1,num180);
a_240 = 240*ones(1,num240);
a_300 = 300*ones(1,num300);
a_360 = 360*ones(1,num360);
a = [a_30';a_60';a_120';a_180';a_240';a_300';a_360'];

% the time slots for each bed 
% ui is the list of the start time of each time slot on bed i
% vi is the list of the end time of each time slot on bed i
u1=[0; 30; 60; 120; 150; 180; 240; 420];
v1=[30;60; 120;150; 180; 240; 420; 600];
u2=[15;60; 120;180;300;330;450;510];
v2=[45;120;180;300;330;450;510;540];
u3=[15;75;105;135;165;285;465];
v3=[45;105;135;165;285;465;585];
u4=[30;75;105;135;195];
v4=[60;105;135;195;555];
u5=[30;150;180;360];
v5=[150;180;360;600];
u6=[45;225;405];
v6=[225;405;525];
u7=[45;285;345;375;405];
v7=[285;345;375;405;525];
u8=[90;120;180;480];
v8=[120;180;480;510];
u9=[90;270;300;330];
v9=[270;300;330;570];
u10=[105;225;465;495];
v10=[225;465;495;525];
u11=[120;240;420;];
v11=[240;420;600;];
u12=[120;300];
v12=[300;540];
u13=[120;420];
v13=[420;540];
u14=[135;315];
v14=[315;555];

% full sample
u={u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14};
v={v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14};
n=length(a);
m=[length(u{1});length(u{2});length(u{3});length(u{4});length(u{5});length(u{6});length(u{7});length(u{8});length(u{9});length(u{10});length(u{11});length(u{12});length(u{13});length(u{14})]; %need to change based on bed
low_b=min(a);
T=[v{1}(length(v{1}));v{2}(length(v{2}));v{3}(length(v{3}));v{4}(length(v{4}));v{5}(length(v{5}));v{6}(length(v{6}));v{7}(length(v{7}));v{8}(length(v{8}));v{9}(length(v{9}));v{10}(length(v{10}));v{11}(length(v{11}));v{12}(length(v{12}));v{13}(length(v{13}));v{14}(length(v{14}))]; %need to change based on bed
M = [10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000]; %need to change based on bed

% % smaller sample
% u = {u1,u2};
% v = {v1,v2};
% m=[length(u{1});length(u{2})]; %need to change based on bed
% T=[v{1}(length(v{1}));v{2}(length(v{2}))]; %need to change based on bed
% M = [10000;10000]; %need to change based on bed

B=14; % bed number used

l=[30;60;120;180;240;300;360]; % the list of the patient type
K=length(l);


gamma=zeros(K,B); % number of slot of type k on bed b
for b=1:B
    for k=1:K
        gamma(k,b)= sum(v{b}-u{b}==l(k));
    end
end

phi = zeros(K,1); % number of appointment of type k
for k=1:K
    phi(k) = numel(find(a==l(k)));
end

omega = zeros(B,1); % hours of bed b
for b=1:B
    omega(b) = sum(v{b}-u{b});
end

% weights
weight_f=100; % lambda2
weight_delta = 1.1; %rho
weight_delta_kb=10; % lambda1

cvx_begin
% cvx_solver_settings('MSK_DPAR_MIO_TOL_REL_GAP', 39e-2)
cvx_solver_settings('MSK_DPAR_OPTIMIZER_MAX_TIME',1000)
cvx_solver mosek
binary variables theta(B) betta(K,B) 
integer variables f(K,B) delta(K,B) 


td=0;
for k=1:K
    td= td+power(weight_delta,k)*sum(delta(k,:));
end

    minimize weight_delta_kb*td-weight_f*sum(sum(f))+sum(theta)-sum(sum(betta))

    for b=1:B
        for k=1:K
            delta(k,b) >= f(k,b)-gamma(k,b)*betta(k,b);
            delta(k,b) >= gamma(k,b)*betta(k,b)-f(k,b);
            f(k,b) >= 0;
            theta(b) >= betta(k,b);
            betta(k,b) <= f(k,b);
        end
        l'*f(:,b) <= omega(b)*theta(b);
 

    end

    for k=1:K
        sum(f(k,:)) <= phi(k);
    end


cvx_end
 
betta
f

for k=1:K
    disp(k)
    disp(sum(f(k,:)))
end

cvx_optval
cvx_cputime

toc
rec_toc=round(toc,4);

% diary off
% 
% % save the variable to the mat file
% str_m = sprintf('%d ', m);
% str_m = sprintf('%s', str_m); 
% filename = strcat(pwd,'/variable_mat_files_deterministic/',str_p,'.mat'); %file + # of bed + # of slots + # of appointments.mat
% save(filename)
% 
% % get optimizer time
% filename = dfile;
% str = extractFileText(filename);
% start = "Optimizer terminated. Time: ";
% fin = " ";
% optimizerTime = extractBetween(str,start,fin);
% optimizerTime_double=double(optimizerTime);
% 
% c = {str_p,optimizerTime_double,rec_toc,cvx_optval};
% writecell(c,'overridePolicy_result_multibeds_deterministic.xlsx','WriteMode','append');
end