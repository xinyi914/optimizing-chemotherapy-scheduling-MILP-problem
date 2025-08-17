% This file is for implementing the model 1 in stochastic case
% **all codes for saving result to the file are commented

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
% p_list_str = getVarNames(p15, p16, p17, p18, p19, p20, p22);
p_list = [p1];
p_list_str = getVarNames(p1); %change when the p_list change

lambda1=50;
lambda2=4;
lambda3=1;
lambda4=2;
lambda5=3;

% loop for each day and save results to txt file for each day
for p_idx = 1:size(p_list,2)

% save the information to the textfile
% str_p = sprintf('%s', p_list_str(p_idx)); 
% dfile=strcat(pwd,'/overridePolicy_stochastic_alpha=0.05/diary/',str_p,'.txt');
% if exist(dfile,'file'); delete(dfile); end
% diary(dfile)
% diary on

tic;
p=p_list(:,p_idx)

% the information about the number of each appointment type at each time for all beds:
% each row represents the appointment type, each column represents the time 
   % 1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41
c = [1  2  2  0  0  2  1  2  1  1  2  0  0  0  0  0  0  0  1  0  2  0  0  1  0  1  0  0  0  0  0  1  1  1  1  0  0  0  0  0  0;
     0  0  0  0  2  0  0  0  2  1  0  0  1  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0;
     0  0  1  0  0  0  0  1  1  0  0  1  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  2  1  0  0  1  0  0  0  0  0  0  0  0  0;
     0  0  0  1  0  0  1  0  1  1  0  0  1  0  0  1  2  0  0  1  0  0  0  0  0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0;
     0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  1  1  1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0;
     0  0  0  0  0  0  0  0  1  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0;
     0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]; 

% % condition under bednumber=1
% c = [0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0;
%      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0;
%      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0;
%      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0;
%      0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0;
%      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0;
%      0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0];    


T=[30;60;120;180;240;300;360];
time_step = 41;
patient_type = 7;
bed_number = 14;


% decides if the time slot (i',j',t) exists: j'th appointment slot for a
% type i' patient at time t
B = zeros(patient_type,2,time_step);
for i=1:size(c,1)
    for j=1:size(c,2)
        if c(i,j) == 1
            B(i,1,j) = 1;
        elseif c(i,j) == 2
            B(i,1,j) = 1;
            B(i,2,j) = 1;
        end
    end
end

% decides if the binary variable xo exists(binary variable for override policy 1) 
B_xo = zeros(patient_type,patient_type,2,time_step);
for i = 1:size(B_xo,1)
    for j = 1:size(B_xo,2)
        if i<j
            B_xo(i,j,:,:) = B(j,:,:);
        end
    end
end

% decides if the binary variable x exists(binary variable for override
% policy 1 and cases that exactly assigning the appointments to the time slot)
B_x = zeros(patient_type,patient_type,2,time_step);
for i = 1:size(B_x,1)
    for j = 1:size(B_x,2)
        if i<=j
            B_x(i,j,:,:) = B(j,:,:);
        end
    end
end

% % information about the y,z,w (binary variables for override policy 2)
% the chair/bed where time slot(i',j',t) is in
chair = zeros(patient_type,time_step,2);
% chair for patient type1 30
chair(1,1,1)=1;chair(1,2,1)=2;chair(1,2,2)=3;chair(1,3,1)=1;chair(1,3,2)=4;chair(1,6,1)=3;
chair(1,6,2)=4;chair(1,7,1)=8;chair(1,8,1)=3;chair(1,8,2)=4;chair(1,9,1)=1;chair(1,10,1)=3;
chair(1,11,1)=1;chair(1,11,2)=5;chair(1,19,1)=9;chair(1,21,1)=2;chair(1,21,2)=9;chair(1,24,1)=7;
chair(1,26,1)=7;chair(1,32,1)=10;chair(1,33,1)=8;chair(1,34,1)=10;chair(1,35,1)=2;
% chair for patient type2 60
chair(2,5,1)=1;chair(2,5,2)=2;chair(2,9,1)=2;chair(2,9,2)=8;chair(2,10,1)=4;chair(2,13,1)=1;
chair(2,20,1)=7;chair(2,31,1)=2;
% chair for patient type3 120
chair(3,3,1)=5; chair(3,8,1)=10;chair(3,9,1)=11;chair(3,12,1)=3;chair(3,13,1)=2;chair(3,23,1)=2;
chair(3,28,1)=6;chair(3,28,2)=7;chair(3,29,1)=13;chair(3,32,1)=3;
% chair for patient type4 180
chair(4,4,1)=6;chair(4,7,1)=9;chair(4,9,1)=12;chair(4,10,1)=14;chair(4,13,1)=5;chair(4,16,1)=6;
chair(4,17,1)=1;chair(4,17,2)=11;chair(4,20,1)=3;chair(4,29,1)=1;chair(4,29,2)=11;
% chair for patient type5 240
chair(5,4,1)=7;chair(5,16,1)=10;chair(5,21,1)=12;chair(5,22,1)=14;chair(5,23,1)=9;chair(5,25,1)=5;
% chair for patient type6 300
chair(6,9,1)=13;chair(6,13,1)=8;
% chair for patient type7 360
chair(7,14,1)=4;

% %  for the case under bedNumber=1
% chair = zeros(patient_type,time_step,2);
% chair(1,24,1)=7; chair(1,26,1)=7;chair(2,20,1)=7;chair(3,28,1)=7;chair(5,4,1)=7;

% decides if the two time slots (i',j',t) and (i",j",t+i'/15) on the same bed exists for
% combining
Y = zeros(patient_type,patient_type,2,2,time_step);
Z = zeros(patient_type,2,time_step);
W = zeros(patient_type,2,time_step);


for i=1:size(c,1)
    for j=1:size(c,2) % it is t
        for h = 1:size(c,1) % it is i"
            if c(i,j) ~= 0 && c(h,j+T(i)/15) ~= 0 && j+T(i)/15 <=time_step
                if chair(i,j,1) == chair(h,j+T(i)/15,1) && chair(i,j,1) ~= 0
                    Y(i,h,1,1,j) = 1;
                    Z(i,1,j) = 1;
                    W(h,1,j+T(i)/15) = 1;
                end
                if chair(i,j,1) == chair(h,j+T(i)/15,2) && chair(i,j,1) ~= 0
                    Y(i,h,1,2,j) = 1;
                    Z(i,1,j) = 1;
                    W(h,2,j+T(i)/15) = 1;
                end
                if chair(i,j,2) == chair(h,j+T(i)/15,1) && chair(i,j,2) ~= 0
                    Y(i,h,2,1,j) = 1;
                    Z(i,2,j) = 1;
                    W(h,1,j+T(i)/15) = 1;
                end
                if chair(i,j,2) == chair(h,j+T(i)/15,2) && chair(i,j,2) ~= 0
                    Y(i,h,2,2,j) = 1;
                    Z(i,2,j) = 1;
                    W(h,2,j+T(i)/15) = 1;
                end
            end
        end
    end
end

% decides if the appointment could use override policy 2, being assigned to
% a combined time slot 
B_y = zeros(patient_type,patient_type,patient_type,2,2,time_step);
B_z = zeros(patient_type,patient_type,2,time_step);
B_w = zeros(patient_type,patient_type,2,time_step);
for i = 1:size(B_y,1) %i
    for j = 1:size(B_y,2) %i'
        for h = 1:size(B_y,3) %i''
            if i>j && i>h && T(j)+T(h)>=T(i)
                B_y(i,j,h,:,:,:) = Y(j,h,:,:,:);
                B_z(i,j,:,:) = Z(j,:,:);
                B_w(i,h,:,:) = W(h,:,:);
            end
        end
    end
end


% information for override policy 3
% decides if the time slot safisfied override policy exist
V = zeros(patient_type,time_step,2);
for i=1:size(c,1)
    for j=1:size(c,2)
        if c(i,j) == 1
            V(i,j,1) = 1;
        elseif c(i,j) == 2
            V(i,j,1) = 1;
            V(i,j,2) = 1;
        end
    end
end

% decides if the apoointment i and i" could use override policy in time
% slot(i',t,j') 
B_v = zeros(patient_type,patient_type,patient_type,time_step,2);

for i = 1:size(B_v,1) %i
    for j = 1:size(B_v,2) %i"
        for h = 1:size(B_v,3) %i'
            if i<h && j<h && T(i)+T(j)<=T(h)
                B_v(i,j,h,:,:) = V(h,:,:);
            end
        end
    end
end

% index that variables are zero
zeroIndices_bx = (B_x==0);
zeroIndices_bxo = (B_xo==0);
zeroIndices_by = (B_y==0);
zeroIndices_bv = (B_v==0);
zeroIndices_bz = (B_z==0);
zeroIndices_bw = (B_w==0);

%stochastic
mu = T;
cov = eye(length(T));
alpha = 0.05; % the probability that the actual total appointment time exceeds the total duration of the time slots
L = zeros(bed_number,1); % the total ength of time slots on bed b
for i=1:patient_type
    for j=1:2
        for t = 1:time_step
            if chair(i,t,j) ~= 0
                L(chair(i,t,j)) = L(chair(i,t,j))+T(i);
            end
        end
    end
end



cvx_begin
cvx_solver mosek
binary variables x(patient_type,patient_type,2,time_step) y(patient_type,patient_type,patient_type,2,2,time_step) z(patient_type,patient_type,2,time_step) w(patient_type,patient_type,2,time_step) v(patient_type,patient_type,patient_type,time_step,2);
integer variables q(patient_type);
variables s(patient_type,bed_number)
expressions u(patient_type,time_step) u0(patient_type,time_step) y_c(patient_type,patient_type,2,time_step) %s(patient_type,bed_number);


xo_s=B_xo.*x;


% equation 8
for i=1:patient_type
    for t=1:time_step
        u(i,t) = sum(sum(x(i,:,:,t)))+sum(sum(sum(sum(y(i,:,:,:,:,t)))))+sum(sum(sum(v(i,:,:,t,:))))+sum(sum(sum(v(:,i,:,t,:))));%sum(sum(sum(v_s(i,:,:,t,:))));
    end
end

% equation 9
for i=1:patient_type
   for t=1:time_step
      u0(i,t) = sum(sum(xo_s(i,:,:,t)))+sum(sum(sum(sum(y(i,:,:,:,:,t)))))+sum(sum(sum(v(i,:,:,t,:))))+sum(sum(sum(v(:,i,:,t,:))));%sum(sum(sum(v_s(i,:,:,t,:))));
   end
end

% equation 5
y_c = cvx(zeros(patient_type,patient_type,2,time_step));
for i=1:patient_type
    for m=1:patient_type %i"
        for l=1:2 %j"
            for t=1:time_step
                for j = 1:patient_type %i'
                    if t-T(j)/15 >= 1
                        if c(j,t-T(j)/15) ~= 0
                            for n = 1:c(j,t-T(j)/15) %j'
                                y_c(i,m,l,t) = y_c(i,m,l,t)+y(i,j,m,n,l,t-T(j)/15); 
                            end
                        end
                    end
                end
            end
        end
    end
end


    minimize lambda1*sum(q)+lambda2*sum(sum(u0))+lambda3*sum(sum(sum(sum(xo_s))))+lambda4*sum(sum(sum(sum(sum(sum(y))))))+lambda5*sum(sum(sum(sum(sum(v)))))
    subject to 

        % actual total time that the appointment of type i
        for i=1:patient_type
            for b = 1:bed_number
                xs = 0;
                ys = 0;
                vs1 = 0;
                vs2 = 0;
                for ip = 1:patient_type
                    for j = 1:2
                        for t = 1:time_step
                            if chair(ip,t,j) == b
                                xs = xs+x(i,ip,j,t);
                                ys = ys+sum(sum(y(i,ip,:,j,:,t)));
                                vs1 = vs1+sum(v(i,:,ip,t,j));
                                vs2 = vs2+sum(v(:,i,ip,t,j));
                            end
                        end
                    end
                end
                s(i,b) == xs+ys+vs1+vs2;
            end
        end
        %stochastic
        for b=1:bed_number
            mu'*s(:,b) + sqrt((1-alpha)/alpha)*norm(sqrt(cov)*s(:,b)) <= L(b);
        end
        
        %constraint on the number of appointment assigned
        for i=1:patient_type % newly added
            sum(u(i,:)) <= p(i);
        end

        %remove redundant variable
        x(zeroIndices_bx) == 0;
        y(zeroIndices_by) == 0;
        v(zeroIndices_bv) == 0;
        z(zeroIndices_bz) == 0;
        w(zeroIndices_bw) == 0;

        % equation 10
        for i=1:patient_type
            q(i) >= 0;
            q(i) >= p(i)-sum(u(i,:));
        end

        % equation 1
        for i=1:patient_type %i'
            for j=1:2
                for t=1:time_step
                    if B(i,j,t) ~= 0 
                        sum(x(:,i,j,t)) + sum(z(:,i,j,t)) + sum(w(:,i,j,t)) + sum(sum(1/2.*v(:,:,i,t,j))) <= 1;
                    end
                end
            end
        end

        % equation 2,3,4
        for i =1:patient_type %i
            for j=1:patient_type %i'
                for m=1:patient_type %i"
                    for n=1:2 %j'
                        for l = 1:2 %j"
                            for t =1:time_step 
                                % if t+T(i)/15 <=time_step && chair(j,t,n) ~= 0 && chair(j,t,n)==chair(m,t+T(j)/15,l) %t %6/24 add chair equal
                                if B_y(i,j,m,n,l,t) ~= 0
                                    y(i,j,m,n,l,t) <= z(i,j,n,t);
                                    if t+T(j)/15 <= time_step
                                    y(i,j,m,n,l,t) <= w(i,m,l,t+T(j)/15);
                                    y(i,j,m,n,l,t) >= z(i,j,n,t)+w(i,m,l,t+T(j)/15)-1;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end

        %equation 5
        for i=1:patient_type
            for m=1:patient_type %i"
                for l=1:2 %j"
                    for t=1:time_step
                        if B_w(i,m,l,t) ~= 0
                        % sum(y_s(i,1,m,:,l,t-T(1)/15))+sum(y_s(i,2,m,:,l,t-T(2)/15)) <= 1;
                        y_c(i,m,l,t) <= 1;
                        end

                    end
                end
            end
        end

        %eauqtion 6
        for i=1:patient_type
            for j=1:patient_type %i'
                for n=1:2 %j'
                    for t=1:time_step
                       if B_z(i,j,n,t) ~= 0
                       sum(sum(y(i,j,:,n,:,t))) <= 1;
                       end
                    end
                end
            end
        end

        %equation 7
        for i=1:patient_type
            for t=1:time_step
                for j=1:2
                    % sum(sum(v_s(i,:,:,t,j)))<=1;
                    if V(i,t,j) ~= 0
                    sum(sum(v(:,:,i,t,j)))<=1;
                    end
                end
            end
        end

        %stochastic
        for b=1:bed_number
            mu'*s(:,b) + sqrt((1-alpha)/alpha)*norm(sqrt(cov)*s(:,b)) <= L(b);
        end

cvx_end

% present the number of each override policy
sum(sum(sum(sum(xo_s))))
sum(sum(sum(sum(sum(sum(y))))))
sum(sum(sum(sum(sum(v)))))
sum(q)



% count the number of appointments assigned
count_i = zeros(patient_type,1);

disp('x_s')
indices = find(x>1.0e-6);
[i,iprime,jprime,t] = ind2sub(size(x),indices);
for len = 1:length(i)
    count_i(i(len)) = count_i(i(len))+1;
    disp(['[i,iprime,jprime,t,chair]',num2str(len)])
    display = [num2str(i(len)),' ',num2str(iprime(len)),' ',num2str(jprime(len)),' ',num2str(t(len)),' ',num2str(chair(iprime(len),t(len),jprime(len)))];
    disp(display)
end



disp('y_s')
indices = find(y>1.0e-6);
[i,iprime,ipprime,jprime,jpprime,t] = ind2sub(size(y),indices);
for len = 1:length(i)
    count_i(i(len)) = count_i(i(len))+1;
    disp(['[i,iprime,ipprime,jprime,jpprime,t,chair]',num2str(len)])
    display = [num2str(i(len)),' ',num2str(iprime(len)),' ',num2str(ipprime(len)),' ',num2str(jprime(len)),' ',num2str(jpprime(len)),' ',num2str(t(len)),' ',num2str(chair(iprime(len),t(len),jprime(len)))];
    disp(display)
end


disp('v_s')
indices = find(v>1.0e-6);
[i,ipprime,iprime,t,jprime] = ind2sub(size(v),indices);
for len = 1:length(i)
    count_i(i(len)) = count_i(i(len))+1;
    count_i(ipprime(len)) = count_i(ipprime(len))+1;
    disp(['[i,ipprime,iprime,jprime,t,chair]',num2str(len)])
    display = [num2str(i(len)),' ',num2str(ipprime(len)),' ',num2str(iprime(len)),' ',num2str(jprime(len)),' ',num2str(t(len)),' ',num2str(chair(iprime(len),t(len),jprime(len)))];
    disp(display)
end

disp(count_i)

% %save the variables to the file
% str_p = sprintf('p%d', p_idx);
% filename = strcat(pwd,'/overridePolicy_stochastic_alpha=0.05/',str_p,'.mat');
% save(filename)
% 
% 
toc
rec_toc=round(toc,4);
% diary off
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
% writecell(c,'overridePolicy_result.xlsx','WriteMode','append');
end