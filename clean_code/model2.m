% This file is for implementing model 2 which is the integrated model
% **all codes for saving result to the file are commented

% generate 10 random instances for different cases
% for fileNumber=1:10
% dfile=strcat(pwd,'/variable_mat_files/b10a15diary/num',string(fileNumber),'.txt');
% if exist(dfile,'file'); delete(dfile); end
% diary(dfile)
% diary on
tic;
% a=[30;30;30;30;30];
a=[30;60;60;60;120;60;30;60;120];
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



% randomly generate the instances 
% patientType=[30;60;120;180;240;300;360];
% appointmentNumber = 10; %change when need different number of appointments
% a=randsample(patientType,appointmentNumber,true); % the appointment list

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
u={u1,u7}; %need to change based on bed
v={v1,v7}; %need to change based on bed
% u={u1,u2,u3,u4,u5,u6,u7,u8,u9,u10,u11,u12,u13,u14}; % for using all beds
% v={v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14}; % for using all beds

n=length(a); % the number of appointments
% smaller sample
m=[length(u{1}),length(u{2})]; % the number of time slots on each bed (need to change based on bed)
T=[v{1}(length(v{1})),v{2}(length(v{2}))]; % need to change based on bed

% full sample
% m=[length(u{1});length(u{2});length(u{3});length(u{4});length(u{5});length(u{6});length(u{7});length(u{8});length(u{9});length(u{10});length(u{11});length(u{12});length(u{13});length(u{14})]; %need to change based on bed
% T=[v{1}(length(v{1}));v{2}(length(v{2}));v{3}(length(v{3}));v{4}(length(v{4}));v{5}(length(v{5}));v{6}(length(v{6}));v{7}(length(v{7}));v{8}(length(v{8}));v{9}(length(v{9}));v{10}(length(v{10}));v{11}(length(v{11}));v{12}(length(v{12}));v{13}(length(v{13}));v{14}(length(v{14}))]; %need to change based on bed

epsilon=0.05;
M = [10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000;10000]; %need to change based on bed

B=2; % need to change based on bed!
low_b=min(a); % the shortest appointment in the appointment list

cvx_begin
% cvx_solver_settings('MSK_DPAR_MIO_TOL_REL_GAP', 20e-2) % set the minimum relative gap to 20e-2
cvx_solver_settings('MSK_DPAR_OPTIMIZER_MAX_TIME',300) % set the maxmum time to 300s

cvx_solver mosek
binary variables w(n,n,B) z(n,max(m),B) y(max(m),B,n) x(n,max(m),B) g(n,B) h(max(m),B) r(n,B) q(max(m),B,n);
variables s(n,B) t(n,B) d(n,B) p(n,max(m),B);
expressions x_o(n,max(m),B) completion_t(B);

for i=1:n    
    for b=1:B
        for j=1:m(b)
            x_o(i,j,b)=p(i,j,b)/(v{b}(j)-u{b}(j));
        end
    end
end


    minimize sum(sum(sum(y)))+sum(sum(sum(z)))-100*sum(sum(r))-sum(sum(sum(x_o)))

    for i=1:n
        for b=1:B
            s(i,b) >= u{b}(1); 
            t(i,b) >= u{b}(1); 
            s(i,b)+d(i,b) <= t(i,b);
            t(i,b) <= T(b)+M(b)*(1-r(i,b));
            if i>=2
                t(i-1,b) <= s(i,b);
            end
           sum(w(i,:,b)) <= 1; 
           if i < n
               sum(w(i,:,b)) >= sum(w(i+1,:,b));
           end
           sum(w(i,:,b)) >= r(i,b);
 
        end
        sum(sum(w(:,i,:))) == 1;         
    end
   

    for i = 1:n
        for b = 1:B
            d(i,b) == w(i,:,b)*a;
        end
    end

    for b = 1:B
        for i=1:(n-1)
            r(i,b) >= r(i+1,b);
        end
    end

    for i=1:n
        for b=1:B
            for j=1:m(b)
                p(i,j,b) >= x(i,j,b)*low_b;
                p(i,j,b) >= t(i,b)-s(i,b)+x(i,j,b)*T(b)-T(b);
                p(i,j,b) <= t(i,b)-s(i,b)+x(i,j,b)*low_b-low_b;
                p(i,j,b) <= x(i,j,b)*T(b);
            end
        end
    end


    % constraints for z
    for i=2:n 

        for b=1:B
            for j=1:m(b)
                t(i-1,b)-epsilon >= u{b}(j)-M(b)*(1-z(i,j,b));
                s(i,b)+epsilon <= v{b}(j)+M(b)*(1-z(i,j,b));
                z(i,j,b) <= r(i,b);
                z(i,j,b) <= r(i-1,b);
            end
        end
        sum(sum(z(i,:,:)) )<= 1;
    end


    for b=1:B
        for j=1:m(b)
            z(1,j,b)==0;
        end         
    end


    %constraints for y when y=1
    for i=1:n
        for b=1:B
            for j=2:m(b)
                s(i,b)+epsilon <= v{b}(j-1)+M(b)*(1-y(j,b,i)); 
                t(i,b)-epsilon >= v{b}(j-1)-M(b)*(1-y(j,b,i)); 
                y(j,b,i) <= r(i,b);
            end

        end
    end


    for b=1:B
        for j=2:m(b)
            sum(y(j,b,:)) <= 1;
        end
    end



    % constraints for y when y=0
     for i=1:n
        for b=1:B
            for j=2:m(b)
                s(i,b)+M(b)*q(j,b,i)+M(b)*y(j,b,i) >= v{b}(j-1); 
                t(i,b)-M(b)*(1-q(j,b,i))-M(b)*y(j,b,i) <= v{b}(j-1);
            end      
        end
     end

     for i=1:n
         for b=1:B
            y(1,b,i)==0;
            q(1,b,i)==0;
         end
     end

     for i=1:n
         for b=1:B
             for j=2:m(b)
                 q(j,b,i) >= y(j,b,i);
             end
         end
     end



    % constraints for g

    for i=2:n-1
        for b=1:B
            temp1=0.0;
            for j=2:m(b)
                temp1 = temp1+y(j,b,i);
            end
            g(i,b) <= sum(z(i,:,b))+sum(z(i+1,:,b))+temp1;
        end
    end

    temp2=0.0;

    for b=1:B
        temp2=0.0;
        for j=2:m(b)
            temp2=temp2+y(j,b,1);
        end
        g(1,b) <= sum(z(2,:,b))+temp2;
    end


    for b=1:B
        temp3=0.0;
        for j=2:m(b)
            temp3=temp3+y(j,b,n);
        end
        g(n,b) <= sum(z(n,:,b))+temp3;
    end

    for i=2:n
        for b=1:B
            g(i,b) >= sum(z(i,:,b));
        end
    end

    for i=1:n
        for b=1:B
            for j=2:m(b)
                g(i,b) >= y(j,b,i);
            end      
            g(i,b) >= 0;
            g(i,b) <= 1;
        end
    end

    % constraints for h
    for b=1:B
        for j=2:(m(b)-1)
            temp4=0.0;
            for i=2:n
                temp4=temp4+z(i,j,b);
            end
            h(j,b) <= temp4+sum(y(j,b,:))+sum(y(j+1,b,:));
        end
    end

    for b=1:B
        temp5=0.0;
        for i=2:n
                temp5=temp5+z(i,1,b);
        end
        h(1,b) <= temp5+sum(y(2,b,:));
    end

    for b=1:B
        temp6=0.0;
        for i=2:n
            temp6=temp6+z(i,m(b),b);
        end
        h(m(b),b) <= temp6+sum(y(m(b),b,:));
    end

    for b=1:B
        for j=1:m(b)
            for i=2:n
                h(j,b) >= z(i,j,b);
            end
            h(j,b) >= 0;
            h(j,b) <= 1;
        end
    end
    for b=1:B
        for j=2:m(b)
            h(j,b) >= sum(y(j,b,:));
        end
    end


    % constraints for x
    for i=1:n
        for b=1:B
            for j=1:m(b)
                x(i,j,b) <= 1-h(j,b);
                x(i,j,b) <= 1-g(i,b);
                x(i,j,b) <= r(i,b);
                s(i,b) >= u{b}(j)-M(b)*(1-x(i,j,b));
                t(i,b) <= v{b}(j)+M(b)*(1-x(i,j,b));
            end
            sum(x(i,:,b)) <= r(i,b);
            sum(x(i,:,b)) == r(i,b)-g(i,b);
        end
    end

    for b=1:B
        for j=1:m(b)
            sum(x(:,j,b)) <= 1;
        end
    end


cvx_end
 
s=int64(s)
t=int64(t)
d
x
cvx_optval
cvx_cputime

toc
rec_toc=round(toc,4);

% write the results into the test file and excel decomment it if want to
% record the results
% diary off
% % 
% % write the result into a text file
% filename = dfile;
% str = extractFileText(filename);
% start = "Optimizer terminated. Time: ";
% fin = " ";
% optimizerTime = extractBetween(str,start,fin);
% optimizerTime_double=double(optimizerTime);
% % start_gap = "The relative gap is ";
% % fin_gap = "(%).";
% % relativeGap = extractBetween(str,start_gap,fin_gap);
% % relativeGap = relativeGap(1,1);
% start_result = "Status:";
% fin_result = "cvx_optval =";
% result = extractBetween(str,start_result,fin_result);
% 
% % fileNumber=10;
% if isfile('overridepolicy_multipleBeds_result.txt')
%     fileID = fopen('overridepolicy_multipleBeds_result2.txt','a+');
%     fprintf(fileID,'------------------------------------------------------------\n');
%     fprintf(fileID,'bed number: %d\n',B);
%     fprintf(fileID,'number of appointments: %d\n',length(a));
%     fprintf(fileID,'number of slots: %d\n',m);
%     fprintf(fileID,'%s%d\n','a',fileNumber); %add 1 when different a with same slot and bed number
%     fprintf(fileID,'%d ',a);
%     fprintf(fileID,'\n');
%     fprintf(fileID,'\n');
%     fprintf(fileID,'Elapsed time is %.2f seconds.\n',toc);
%     fprintf(fileID,'Optimizer time is %.2f seconds.\n',optimizerTime_double);
%     % fprintf(fileID,'Relative gap is %s(%%).\n',relativeGap);
%     fprintf(fileID,'status: %s\n',result);
%     fclose(fileID);
% 
% else
%     fileID = fopen('overridepolicy_multipleBeds_result2.txt','w');
%     fprintf(fileID,'bed number: %d\n',B);
%     fprintf(fileID,'number of appointments: %d\n',length(a));
%     fprintf(fileID,'number of slots: %d\n',m);
%     fprintf(fileID,'%s%d\n','a',fileNumber); %add 1 when different a with same slot and bed number
%     fprintf(fileID,'%d ',a);
%     fprintf(fileID,'\n');
%     fprintf(fileID,'\n');
%     fprintf(fileID,'Elapsed time is %.2f seconds.\n',toc);
%     fprintf(fileID,'Optimizer time is %.2f seconds.\n',optimizerTime_double);
%     % fprintf(fileID,'Relative gap is %s(%%).\n',relativeGap);
%     fprintf(fileID,'status: %s',result);
%     fclose(fileID);
% end
% 
% % 
% % write into excel sheeet
% str_m = sprintf('%d ', m);
% str_m = sprintf('%s', str_m); 
% c = {B,length(a),str_m,rec_toc,optimizerTime_double,cvx_optval};
% writecell(c,'overridepolicy_multibeds3.xlsx','WriteMode','append');
% 
% % save the variable to the mat file
% filename = strcat(pwd,'/variable_mat_files/b',num2str(B),'a',num2str(length(a)),'s',str_m,'_',num2str(fileNumber),'.mat'); %file + # of bed + # of slots + # of appointments.mat %add 1 when different a with same slot and bed number
% save(filename)
% end
