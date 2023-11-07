classdef DTInfo
  properties (Constant)
      scenario_label_map = containers.Map;
  end

  methods (Static)
    function [dt_info] = read_dt_info(filename)
    % Read DT data from a saved MAT file
    %
    % Arguments
    % filename:  Name of file to read from
    %
    % Outputs
    % dt_info:  Structure containing DT data
      filedata = matfile(filename, 'Writable', false);
      dt_info = filedata.dt_info;
    end
    
    function [dt_info] = read_sim_data(filename)
    % Read DT data from a saved simulation output
    %
    % Arguments
    % filename:  Name of file to load from
    %
    % Outputs
    % dt_info:  Structure containing DT data

      % Load saved simulation data
      filedata = load(filename, 'sim_output');

      % Get simulation info
      dt_info.name = filedata.sim_output.SimulationMetadata.UserString;
      dt_info.variables = filedata.sim_output.SimulationMetadata.UserData.Variables;
      
      % TODO: Scenario start time/configuration start time
      dt_info.sim_start_time = 15;
      
      % Get saved data
      dt_info.data = filedata.sim_output.ess_virtual_twin;
      
      % Load global parameters
      gp = global_params();
      
      % Scale values to their per-unit values
      dt_info.data(:, 1:3) = dt_info.data(:, 1:3) ./ gp.current_pu;   % iLf
      dt_info.data(:, 4:6) = dt_info.data(:, 4:6) ./ gp.voltage_pu;   % vCf
      dt_info.data(:, 7:9) = dt_info.data(:, 7:9) ./ gp.current_pu;   % iLo
      
      % Keep magnitude of error vector
      dt_info.data(:, 20:28) = abs(dt_info.data(:, 20:28));
      
      % TODO: Some kind of frequency transform?
    end
    
    function [good] = validate_sim_data(filename, data_len)
    % Validates the integrity of data in a saved simulation output file
    % Does not validate correctness of the data, only the existence of the
    % required fields, data lengths, etc.
    %
    % Arguments
    % filename:  Name of file to load from
    % data_len (optional):  Expected number of data samples, default 16001
    %
    % Outputs
    % good:  True if data seems intact, false otherwise
      arguments
        filename = [];
        data_len = 16001;
      end

      try
        % Check for file
        if ~isfile(filename)
          error('Inalid filename');
        end
        
        % Load saved simulation data
        file_data = load(filename, 'sim_output');

        if ~isfield(file_data, 'sim_output')
          error('Missing sim_output in file');
        end

        % Check structure of loaded data
        if ~isprop(file_data.sim_output, 'ess_virtual_twin')
          error('Missing ESS data');
        end
        
        if ~isprop(file_data.sim_output, 'SimulationMetadata')
          error('Missing simulation metadata');
        end
        
        if ~isprop(file_data.sim_output.SimulationMetadata, 'UserString')
          error('Missing scenario name');
        end
        
        if ~isprop(file_data.sim_output.SimulationMetadata, 'UserData')
          error('Missing UserData property');
        end
        
        if ~isprop(file_data.sim_output.SimulationMetadata.UserData, 'Variables')
          error('Missing variable override info');
        end
        
        % Check saved ESS data length
        if (size(file_data.sim_output.ess_virtual_twin, 1) ~= data_len)
          error('Invalid ESS data length');
        end

        % Done
        good = true;
        
      catch ex
        % Validation failed
        fprintf('Validation failed:  %s', ex.message);
        good = false;
      end
    end
    
    function [ilf_meas] = get_ilf_meas(dt_info)
    % Get measured filter inductor current
      ilf_meas = dt_info.data(:, 1:3);
    end
    
    function [vcf_meas] = get_vcf_meas(dt_info)
    % Get measured filter capacitor voltage
      vcf_meas = dt_info.data(:, 4:6);
    end
    
    function [ilo_meas] = get_ilo_meas(dt_info)
    % Get measured output inductor current
      ilo_meas = dt_info.data(:, 7:9);
    end
    
    function [ilf_err] = get_ilf_err(dt_info)
    % Get error vector components for filter inductor current
      ilf_err = dt_info.data(:, 20:22);
    end
    
    function [vcf_err] = get_vcf_err(dt_info)
    % Get error vector components for filter capacitor voltage
      vcf_err = dt_info.data(:, 23:25);
    end
    
    function [ilo_err] = get_ilo_err(dt_info)
    % Get error vector components for output inductor current
      ilo_err = dt_info.data(:, 26:28);
    end
        
    function [vgrid] = get_vgrid(dt_info)
    % Get grid voltages
      vgrid = dt_info.data(:, 17:19);
    end
    
    function [gates] = get_gates(dt_info)
    % Get gate drive signals
      gates = dt_info.data(:, 10:15);
    end
    
    function [meas] = get_all_meas(dt_info)
    % Get all measured values for all state signals
      meas = dt_info.data(:, 1:9);
    end
    
    function [err] = get_all_err(dt_info)
    % Get error vector components for all state signals
      err = dt_info.data(:, 20:28);
    end
    
    function [data] = get_all_data(dt_info)
    % Get measured values and error vector components
      data = dt_info.data(:, 1:28);
    end
    
    function [name] = get_scenario_name(dt_info)
    % Get the scenario name
      strs = split(dt_info.name, '/');
      name = strs(end);
    end

    function [data] = get_input_dlarray(dt_info)
    % Get the data formatted for use by neural networks with CBT format
      data = DTInfo.get_all_err(dt_info)';
      data = reshape(data, [size(data, 1), 1, size(data, 2)]);
      data = dlarray(data, 'CBT');
    end

    function initialize_scenario_labels(training_data_dir)
    % Read scenario names from training data folder and create one-hot encoded labels
      scenario_names = dir(training_data_dir);
      scenario_names = {scenario_names.name};
      scenario_names = scenario_names(~ismember(scenario_names, {'.', '..'}));
      scenario_names = string(scenario_names);

      label_map = DTInfo.scenario_label_map;
      for n = scenario_names
        label_map(n) = onehotencode(n, 1, 'ClassNames', scenario_names);
      end
    end

    function [label] = get_scenario_label(scenario_name)
    % Get the one-hot encoded label for the scenario name
      label_map = DTInfo.scenario_label_map;
      label = label_map(scenario_name);
    end
    
  end
end
