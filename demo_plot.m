% Faults at generator
%demo_scenario_filename = 'D:/data/training/Fault 1 Location 2/scenario_25-2.mat';
%demo_scenario_filename = 'D:/data/training/Fault 2 Location 2/scenario_25-3.mat';
%demo_scenario_filename = 'D:/data/training/Fault 3 Location 2/scenario_25-4.mat';
%demo_scenario_filename = 'D:/data/training/Fault 4 Location 2/scenario_25-5.mat';
%demo_scenario_filename = 'D:/data/training/Fault 5 Location 2/scenario_25-6.mat';
%demo_scenario_filename = 'D:/data/training/Fault 6 Location 2/scenario_25-7.mat';
%demo_scenario_filename = 'D:/data/training/Fault 7 Location 2/scenario_25-8.mat';
%demo_scenario_filename = 'D:/data/training/Fault 8 Location 2/scenario_25-9.mat';
%demo_scenario_filename = 'D:/data/training/Fault 9 Location 2/scenario_25-10.mat';
%demo_scenario_filename = 'D:/data/training/Fault 10 Location 2/scenario_25-11.mat';
%demo_scenario_filename = 'D:/data/training/Fault 11 Location 2/scenario_25-12.mat';

% Faults on bus
%demo_scenario_filename = 'D:/data/training/Fault 1 Location 3/scenario_25-13.mat';
%demo_scenario_filename = 'D:/data/training/Fault 2 Location 3/scenario_25-14.mat';
%demo_scenario_filename = 'D:/data/training/Fault 3 Location 3/scenario_25-15.mat';
%demo_scenario_filename = 'D:/data/training/Fault 4 Location 3/scenario_25-16.mat';
%demo_scenario_filename = 'D:/data/training/Fault 5 Location 3/scenario_25-17.mat';
%demo_scenario_filename = 'D:/data/training/Fault 6 Location 3/scenario_25-18.mat';
%demo_scenario_filename = 'D:/data/training/Fault 7 Location 3/scenario_25-19.mat';
%demo_scenario_filename = 'D:/data/training/Fault 8 Location 3/scenario_25-20.mat';
%demo_scenario_filename = 'D:/data/training/Fault 9 Location 3/scenario_25-21.mat';
%demo_scenario_filename = 'D:/data/training/Fault 10 Location 3/scenario_25-22.mat';
%demo_scenario_filename = 'D:/data/training/Fault 11 Location 3/scenario_25-23.mat';

% A-B faults at each location
%demo_scenario_filename = 'D:/data/training/Fault 1 Location 2/scenario_25-2.mat';
%demo_scenario_filename = 'D:/data/training/Fault 1 Location 3/scenario_25-13.mat';
%demo_scenario_filename = 'D:/data/training/Fault 1 Location 4/scenario_25-125.mat';
%demo_scenario_filename = 'D:/data/training/Fault 1 Location 5/scenario_25-88.mat';
%demo_scenario_filename = 'D:/data/training/Fault 1 Location 6/scenario_25-51.mat';

% A-B-C-G faults at each location
%demo_scenario_filename = 'D:/data/training/Fault 11 Location 2/scenario_25-251.mat';
%demo_scenario_filename = 'D:/data/training/Fault 11 Location 3/scenario_25-225.mat';
%demo_scenario_filename = 'D:/data/training/Fault 11 Location 4/scenario_25-273.mat';
%demo_scenario_filename = 'D:/data/training/Fault 11 Location 5/scenario_25-236.mat';
%demo_scenario_filename = 'D:/data/training/Fault 11 Location 6/scenario_25-199.mat';

% Load steps
%demo_scenario_filename = 'D:/data/training/Load step 1/scenario_25-24.mat';
%demo_scenario_filename = 'D:/data/training/Load step 2/scenario_25-25.mat';
%demo_scenario_filename = 'D:/data/training/Load step 3/scenario_25-26.mat';

% Prime mover, no events
%demo_scenario_filename = 'D:/data/training/Prime mover loss/scenario_25-27.mat';
%demo_scenario_filename = 'D:/data/training/No events/scenario_25-1.mat';

num_cycles = 5;

for demo_scenario_filename = [...
  "D:/data/training/Fault 1 Location 2/scenario_50-2.mat", ...
  "D:/data/training/Fault 1 Location 3/scenario_50-13.mat", ...
  "D:/data/training/Fault 1 Location 4/scenario_50-125.mat", ...
  "D:/data/training/Fault 1 Location 5/scenario_50-88.mat", ...
  "D:/data/training/Fault 1 Location 6/scenario_50-51.mat", ...
]
  
  dt_info = DTInfo.read_dt_info(demo_scenario_filename);
  
  figure('WindowState', 'maximized');
  fig_full = demo_model(model, dt_info, num_cycles);
end
