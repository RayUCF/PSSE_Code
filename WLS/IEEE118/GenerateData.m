function GenerateData()
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Following the loadprofile to get various load
%runpf --> True measurements and Ybus
%WGN --> Actural measurements%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear
clc

%%%%%%%%%%%%%%%%%%%%%Part I%%%%%%%%%%%%%%%%%%%%%%%
% % % True measurements and Ybus
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mpc = loadcase('case118');
% Ybus1 = full(makeYbus(mpc));
mpc.branch(:,5) = 0; %line charging
mpc.branch(:,9) = 0; %tap ration
mpc.bus(:,5) = 0;   % Gs
mpc.bus(:,6) = 0;   %Bs
%%%%Combine the two circuits between two same nodes
multi_index = [];
ft = mpc.branch(:,1:2);
for i = 1:length(ft)-1
    if ft(i,1)==ft(i+1,1) && ft(i,2)==ft(i+1,2)
        disp(ft(i,:));
        multi_index = [multi_index;i+1];
        P_z = (mpc.branch(i,3) + j*mpc.branch(i,4))*(mpc.branch(i+1,3) + j*mpc.branch(i+1,4)) /(mpc.branch(i,3) + j*mpc.branch(i,4) + mpc.branch(i+1,3) + j*mpc.branch(i+1,4)) ;
        mpc.branch(i,3) = real(P_z);
        mpc.branch(i,4) = imag(P_z);
        mpc.branch(i,5) = mpc.branch(i,5) + mpc.branch(i+1,5);
    end
end

mpc.branch(multi_index,:)=[];%%%%%New Branch Data%%%%%

% %%%%%%%%%%%%%%%%%%%%%%Ybus%%%%%%%%%%%%%%%%%%%%%%%%%
Ybus = full(makeYbus(mpc));
G_ = real(Ybus);
B_ = imag(Ybus);
G = full(G_);
B = full(B_);

xlswrite('measurements.xlsx',G, 'Real');
xlswrite('measurements.xlsx',B,'Imag');

num = 118;
zdata = xlsread('measurements.xlsx','OriginalIndices');

type = zdata(:,2); % Type of measurement, Vi - 1, Pi - 2, Qi - 3, Pij - 4, Qij - 5, Iij - 6..
fbus = zdata(:,3);
tbus = zdata(:,4); % To bus..
Ri = diag(zdata(:,5)); % Measurement Error..

vi = find(type == 1); % Index of voltage magnitude measurements..
ppi = find(type == 2); % Index of real power injection measurements..
qi = find(type == 3); % Index of reactive power injection measurements..
pf = find(type == 4); % Index of real powerflow measurements..
qf = find(type == 5); % Index of reactive powerflow measurements..

nvi = length(vi); % Number of Voltage measurements..
npi = length(ppi); % Number of Real Power Injection measurements..
nqi = length(qi); % Number of Reactive Power Injection measurements..
npf = length(pf); % Number of Real Power Flow measurements..
nqf = length(qf); % Number of Reactive Power Flow measurements..

% %%%%%%%%%%%%%%%%%True Measurements%%%%%%%%%%%%%%%%%
ratio = xlsread('loadprofile-train', 'Sheet1');
[r,c] = size(ratio);

measurements =[];
true_values = [];

row=1;
NG = 0;
fail = 0;
while row <= r
    disp(row)
    disp('****************************************');
    
    true_v = [];
    true_del = [];
    tmp=[];
    ratld = ratio(row,c);
    mpc = loadcase('case118');
    mpc.gen(:,2) = mpc.gen(:,2);
    mpc.branch(:,5) = 0; % line charging
    mpc.branch(:,9) = 0; % tap ration
    mpc.bus(:,5) = 0;   % Gs
    mpc.bus(:,6) = 0;   %Bs
    %%%%%%%%%%%%%%%%%%%%%%Load Profile%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     mpc.gen(:,2) = ratld.*mpc.gen(:,2);
    mpc.bus(:,3) = ratld.*mpc.bus(:,3).*(1+(0.4*rand(num,1)-0.2));
    mpc.bus(:,4) = ratld.*mpc.bus(:,4).*(1+(0.4*rand(num,1)-0.2));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    msr = zeros(size(fbus,1),1);
%     [baseMVA, bus, gen, branch, success, et] = runpf(mpc);
    [baseMVA, bus, gen, gencost, branch, f, success, et] = runopf(mpc);

    if min(gen(:,2)) < 0
        NG = NG + 1
        continue
    else        
        Load = sum(bus(:,3));
        Gen = sum(gen(:,2));
        msr(vi) = bus(vi,8);
        
        for i = 1:npi
            if ismember(fbus(ppi(i)),gen(:,1))
                index = find(gen(:,1) == fbus(ppi(i)));
                msr(ppi(i)) = (gen(index,2) -bus(fbus(ppi(i)),3))/baseMVA;
            else
                msr(ppi(i)) = (0-bus(fbus(ppi(i)),3))/baseMVA;
            end
        end
        
        for i = 1:nqi
            if ismember(fbus(qi(i)),gen(:,1))
                index = find(gen(:,1) == fbus(qi(i)));
                msr(qi(i)) = (gen(index,3) -bus(fbus(qi(i)),4))/baseMVA;
            else
                msr(qi(i)) = (0-bus(fbus(qi(i)),4))/baseMVA;
            end
        end
        
        % find(branch(:,1)==fbus(ppi) & branch(:,2)==2 )
        for i = 1:npf
            index = find(branch(:,1)==fbus(pf(i))& branch(:,2)==tbus(pf(i)));
            if length(index) == 2
                tmp1 = branch(index(1),14)/baseMVA;
                tmp2 = branch(index(2),14)/baseMVA;
                msr(pf(i)) = tmp1 + tmp2;
            elseif length(index) == 1
                msr(pf(i)) = branch(index,14)/baseMVA;
            else
                index = find(branch(:,2)==fbus(pf(i))& branch(:,1)==tbus(pf(i)));
                msr(pf(i)) = branch(index,16)/baseMVA;
            end
        end
        
        for i = 1:nqf
            index = find(branch(:,1)==fbus(qf(i))& branch(:,2)==tbus(qf(i)));
            if length(index) == 2
                tmp1 = branch(index(1),15)/baseMVA;
                tmp2 = branch(index(2),15)/baseMVA;
                msr(qf(i)) = tmp1 + tmp2;
            elseif length(index) == 1
                msr(qf(i)) = branch(index,15)/baseMVA;
            else
                index = find(branch(:,2)==fbus(qf(i))& branch(:,1)==tbus(qf(i)));
                msr(qf(i)) = branch(index,17)/baseMVA;
            end
        end
        
        [warnmsg,msgid]=lastwarn;
        if strcmp(msgid, 'MATLAB:nearlySingularMatrix')|strcmp(msgid, 'MATLAB:singularMatrix')
            lastwarn('')
            continue;
        end
        
        if success == 1
            row = row + 1;
        else
            fail = fail + 1;
            continue
        end 
        
        del = bus(:,9)*pi/180;
        V = bus(:,8);
        measurements = horzcat(measurements, msr);
        true_v = horzcat(true_v, V);
        true_del = horzcat(true_del, del);
        tmp = vertcat(true_del,true_v);
        true_values = horzcat(true_values,tmp);
    end
end
measurements = measurements';
true_values = true_values';
xlswrite('truetraindata.xlsx',measurements, 'Sheet1');
xlswrite('truetrainstate.xlsx',true_values,'Sheet1');

[r,c] = size(measurements);
sigma = sqrt(zdata(:,5));
tt = [];
for col = 1:c
    randn(r,1);
    tmp = measurements(:,col) + randn(r,1)*sigma(col); %%%%Add Gaussian Noise
    tt = horzcat(tt,tmp);
end
xlswrite('actualtraindata.xlsx',tt,'Sheet1');
OriginalData= vertcat(zdata',tt);
xlswrite('measurements.xlsx',OriginalData,'Original');
end