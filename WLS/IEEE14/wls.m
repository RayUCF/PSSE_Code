 function wls() %Num_Set
clear all
clc

% Power System State Estimation using Weighted Least Square Method..
num = 14; % IEEE - 14 or IEEE - 30 bus system..(for IEEE-14 bus system replace 30 by 14)...
%zdata = zdatas(num); % Get Measurement data..
zdata = xlsread('measurements.xlsx','Original');
zdata = zdata';
bpq = zeros(num,num);
nbus = num;

G = xlsread('measurements.xlsx','Real');
B = xlsread('measurements.xlsx','Imag');
done =0;
nummsr = 29;
while true
    ii=100;
    while ii
        index = randperm(69);
        index(find(index==1))=[];
        newind= index(1:nummsr);
        newind= horzcat(1,newind);
        newind= sort(newind)
        msrdata = zdata(newind,:);
        
        type = msrdata(:,2); % Type of measurement, Vi - 1, Pi - 2, Qi - 3, Pij - 4, Qij - 5, Iij - 6..
        fbus = msrdata(:,3);
        tbus = msrdata(:,4); % To bus..
        Ri = diag(msrdata(:,5)); % Measurement Error..
        
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
        
        iter = 1;
        tol = 5;
        
        msrdata1 = msrdata(:,6:end);
        [r,c] = size(msrdata1);
        total=[];
        msr_rmse =[];
        
        for set=1:c
            z = msrdata1(:,set);% Measuement values..
            V = ones(nbus,1); % Initialize the bus voltages..
            del = zeros(nbus,1); % Initialize the bus angles..
            E = [del(2:end); V];   % State Vector..
            NG=0;
            flag = true;
            while(tol > 1e-5 && iter <= 200)
                if iter == 200
                    NG=NG+1;
                end
                
                %Measurement Function, h
                h1 = V(fbus(vi),1);
                h2 = zeros(npi,1);
                h3 = zeros(nqi,1);
                h4 = zeros(npf,1);
                h5 = zeros(nqf,1);
                
                for i = 1:npi
                    m = fbus(ppi(i));
                    for k = 1:nbus
                        h2(i) = h2(i) + V(m)*V(k)*(G(m,k)*cos(del(m)-del(k)) + B(m,k)*sin(del(m)-del(k)));
                    end
                end
                
                for i = 1:nqi
                    m = fbus(qi(i));
                    for k = 1:nbus
                        h3(i) = h3(i) + V(m)*V(k)*(G(m,k)*sin(del(m)-del(k)) - B(m,k)*cos(del(m)-del(k)));
                    end
                end
                
                for i = 1:npf
                    m = fbus(pf(i));
                    n = tbus(pf(i));
                    h4(i) = -V(m)^2*G(m,n) - V(m)*V(n)*(-G(m,n)*cos(del(m)-del(n)) - B(m,n)*sin(del(m)-del(n)));
                end
                
                for i = 1:nqf
                    m = fbus(qf(i));
                    n = tbus(qf(i));
                    h5(i) = -V(m)^2*(-B(m,n)+bpq(m,n)) - V(m)*V(n)*(-G(m,n)*sin(del(m)-del(n)) + B(m,n)*cos(del(m)-del(n)));
                end
                
                h = [h1; h2; h3; h4; h5];
                
                % Residue..
                r = z - h;
                
                % Jacobian..
                % H11 - Derivative of V with respect to angles.. All Zeros
                H11 = zeros(nvi,nbus-1);
                
                % H12 - Derivative of V with respect to V..
                H12 = zeros(nvi,nbus);
                for k = 1:nvi
                    for n = 1:nbus
                        if n == k
                            H12(k,n) = 1;
                        end
                    end
                end
                
                % H21 - Derivative of Real Power Injections with Angles..
                H21 = zeros(npi,nbus-1);
                for i = 1:npi
                    m = fbus(ppi(i));
                    for k = 1:(nbus-1)
                        if k+1 == m
                            for n = 1:nbus
                                H21(i,k) = H21(i,k) + V(m)* V(n)*(-G(m,n)*sin(del(m)-del(n)) + B(m,n)*cos(del(m)-del(n)));
                            end
                            H21(i,k) = H21(i,k) - V(m)^2*B(m,m);
                        else
                            H21(i,k) = V(m)* V(k+1)*(G(m,k+1)*sin(del(m)-del(k+1)) - B(m,k+1)*cos(del(m)-del(k+1)));
                        end
                    end
                end
                
                % H22 - Derivative of Real Power Injections with V..
                H22 = zeros(npi,nbus);
                for i = 1:npi
                    m = fbus(ppi(i));
                    for k = 1:(nbus)
                        if k == m
                            for n = 1:nbus
                                H22(i,k) = H22(i,k) + V(n)*(G(m,n)*cos(del(m)-del(n)) + B(m,n)*sin(del(m)-del(n)));
                            end
                            H22(i,k) = H22(i,k) + V(m)*G(m,m);
                        else
                            H22(i,k) = V(m)*(G(m,k)*cos(del(m)-del(k)) + B(m,k)*sin(del(m)-del(k)));
                        end
                    end
                end
                
                % H31 - Derivative of Reactive Power Injections with Angles..
                H31 = zeros(nqi,nbus-1);
                for i = 1:nqi
                    m = fbus(qi(i));
                    for k = 1:(nbus-1)
                        if k+1 == m
                            for n = 1:nbus
                                H31(i,k) = H31(i,k) + V(m)* V(n)*(G(m,n)*cos(del(m)-del(n)) + B(m,n)*sin(del(m)-del(n)));
                            end
                            H31(i,k) = H31(i,k) - V(m)^2*G(m,m);
                        else
                            H31(i,k) = V(m)* V(k+1)*(-G(m,k+1)*cos(del(m)-del(k+1)) - B(m,k+1)*sin(del(m)-del(k+1)));
                        end
                    end
                end
                
                % H32 - Derivative of Reactive Power Injections with V..
                H32 = zeros(nqi,nbus);
                for i = 1:nqi
                    m = fbus(qi(i));
                    for k = 1:(nbus)
                        if k == m
                            for n = 1:nbus
                                H32(i,k) = H32(i,k) + V(n)*(G(m,n)*sin(del(m)-del(n)) - B(m,n)*cos(del(m)-del(n)));
                            end
                            H32(i,k) = H32(i,k) - V(m)*B(m,m);
                        else
                            H32(i,k) = V(m)*(G(m,k)*sin(del(m)-del(k)) - B(m,k)*cos(del(m)-del(k)));
                        end
                    end
                end
                
                % H41 - Derivative of Real Power Flows with Angles..
                H41 = zeros(npf,nbus-1);
                for i = 1:npf
                    m = fbus(pf(i));
                    n = tbus(pf(i));
                    for k = 1:(nbus-1)
                        if k+1 == m
                            H41(i,k) = V(m)* V(n)*(-G(m,n)*sin(del(m)-del(n)) + B(m,n)*cos(del(m)-del(n)));
                        else if k+1 == n
                                H41(i,k) = -V(m)* V(n)*(-G(m,n)*sin(del(m)-del(n)) + B(m,n)*cos(del(m)-del(n)));
                            else
                                H41(i,k) = 0;
                            end
                        end
                    end
                end
                
                % H42 - Derivative of Real Power Flows with V..
                H42 = zeros(npf,nbus);
                for i = 1:npf
                    m = fbus(pf(i));
                    n = tbus(pf(i));
                    for k = 1:nbus
                        if k == m
                            H42(i,k) = -V(n)*(-G(m,n)*cos(del(m)-del(n)) - B(m,n)*sin(del(m)-del(n))) - 2*G(m,n)*V(m);
                        else if k == n
                                H42(i,k) = -V(m)*(-G(m,n)*cos(del(m)-del(n)) - B(m,n)*sin(del(m)-del(n)));
                            else
                                H42(i,k) = 0;
                            end
                        end
                    end
                end
                
                % H51 - Derivative of Reactive Power Flows with Angles..
                H51 = zeros(nqf,nbus-1);
                for i = 1:nqf
                    m = fbus(qf(i));
                    n = tbus(qf(i));
                    for k = 1:(nbus-1)
                        if k+1 == m
                            H51(i,k) = -V(m)* V(n)*(-G(m,n)*cos(del(m)-del(n)) - B(m,n)*sin(del(m)-del(n)));
                        else if k+1 == n
                                H51(i,k) = V(m)* V(n)*(-G(m,n)*cos(del(m)-del(n)) - B(m,n)*sin(del(m)-del(n)));
                            else
                                H51(i,k) = 0;
                            end
                        end
                    end
                end
                
                % H52 - Derivative of Reactive Power Flows with V..
                H52 = zeros(nqf,nbus);
                for i = 1:nqf
                    m = fbus(qf(i));
                    n = tbus(qf(i));
                    for k = 1:nbus
                        if k == m
                            H52(i,k) = -V(n)*(-G(m,n)*sin(del(m)-del(n)) + B(m,n)*cos(del(m)-del(n))) - 2*V(m)*(-B(m,n)+ bpq(m,n));
                        else if k == n
                                H52(i,k) = -V(m)*(-G(m,n)*sin(del(m)-del(n)) + B(m,n)*cos(del(m)-del(n)));
                            else
                                H52(i,k) = 0;
                            end
                        end
                    end
                end
                
                % Measurement Jacobian, H..
                H = [H11 H12; H21 H22; H31 H32; H41 H42; H51 H52];
                % Gain Matrix, Gm..
                Gm = H'*inv(Ri)*H;
                J = sum(inv(Ri)*r.^2);
                
                % State Vector..
                dE = inv(Gm)*(H'*inv(Ri)*r);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %%%%%%%% singular problem%%%%%%%%%%%%%%
                [warnmsg,msgid]=lastwarn;
                if strcmp(msgid, 'MATLAB:nearlySingularMatrix')|strcmp(msgid, 'MATLAB:singularMatrix')
                    lastwarn('')
                    flag = false;
                    break;
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                E = E + dE;
                del(2:end) = E(1:nbus-1);
                V = E(nbus:end);
                iter = iter + 1;
                tol = max(abs(dE));
            end
            
            total = horzcat(total, E);
            tol = 5;
            iter = 1;
            
            if flag == false
                if ii > 1
                    ii = ii-1;
                else
                    nummsr = nummsr + 1
                    ii = 100;
                end
                break;
            end
        end
        if set==c
            done = 1;
            break;
        end
    end
    if done == 1
        TT = total';
        xlswrite('measurements.xlsx',TT,'WLS_Train');
        xlswrite('measurements.xlsx',newind,'indices');
        xlswrite('measurements.xlsx',msrdata','Data');
        break;
    end
end
end



