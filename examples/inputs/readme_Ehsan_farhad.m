

%% 
% channel number = 268 : L-Band = 1:80, C-Band = 81:160, S-Band = 161:268

%%

% JPN 12, Topology

s=[1 1 2 3 3 4 5 5 6 7 7 8 9 9 10 10 11];
t=[2 4 3 4 7 5 6 7 8 8 10 9 10 11 11 12 12];
weights = 1e3.*[593.3 1256.4 351.8 47.4 366 250.7 252.2 250.8 263.8 186.6 490.7 341.6 66.2 280.7 365 1158.7 911.9];
names = {'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'};

%%  All_connections_Profile

All_connections_Profile_...{connection_counter,1} = source_counter; %  source

All_connections_Profile_...{connection_counter,2} = dest_counter; %  sdestination

All_connections_Profile_...{connection_counter,3} = link_list_allpath; % link list

All_connections_Profile_...{connection_counter,4} = totalCosts_allpath_km; % length in km

All_connections_Profile_...{connection_counter,5} = totalCosts_allpath_Nspan; % Nspan,

All_connections_Profile_...{connection_counter,6} = nodes_ShPath_list; % nodes

All_connections_Profile_...{connection_counter,7} = Num_inLine_Amp; % num inLine_Amp

All_connections_Profile_...{connection_counter,8} = don't consider!!

All_connections_Profile_...{connection_counter,9} = don't consider!!

