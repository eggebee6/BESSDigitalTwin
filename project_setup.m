% This script sets up project directories and paths

fprintf('Setting up project, %s\n', datetime());
try
  % Get base directory
  base_dir = pwd();
  addpath(base_dir);
  fprintf('Base directory is %s\n', base_dir);

  % Set up output directory, use environment variable if specified
  env_output_dir = 'NANOGRID_OUTPUT_DIR';
  output_dir = getenv(env_output_dir);
  if isempty(output_dir)
    output_dir = fullfile(base_dir, 'output');
  end
  clear env_output_dir;
  
  % Set up output subfolder, let user create subfolder with create_output_dir()
  output_dir_subfolder = string(datetime('now', 'Format', 'MMddHHmm'));
  output_dir = fullfile(output_dir, output_dir_subfolder);

  create_output_dir = @() create_output_dir_helper(output_dir);
  
  fprintf('Output directory is %s\n', output_dir);
  if ~exist(output_dir, 'dir')
    fprintf('Output directory has not been created\n');
  end
  
  % Set up data directory, use environment variable if specified
  env_data_dir = 'NANOGRID_DATA_DIR';
  data_dir = getenv(env_data_dir);
  if isempty(data_dir)
    data_dir = fullfile(base_dir, 'data');
  end
  clear env_data_dir;
  
  % Set data subdirectories
  configuration_data_dir = fullfile(data_dir, 'configurations');
  scenario_data_dir = fullfile(data_dir, 'scenarios');
  training_data_dir = fullfile(data_dir, 'training');
  
  % Create data subdirectories if they don't exist
  if ~exist(configuration_data_dir, 'dir')
    fprintf('Creating configuration data directory\n');
    mkdir(configuration_data_dir);
  end
  if ~exist(scenario_data_dir, 'dir')
    fprintf('Creating scenario data directory\n');
    mkdir(scenario_data_dir)
  end
  if ~exist(training_data_dir, 'dir')
    fprintf('Creating training data directory\n');
    mkdir(training_data_dir)
  end
  
  fprintf('Data directory is %s\n', data_dir);
  fprintf('Configuration data directory is %s\n', configuration_data_dir);
  fprintf('Scenario data directory is %s\n', scenario_data_dir);
  fprintf('Training data directory is %s\n', training_data_dir);

  % Set up temporary directory
  env_temp_dir = 'TEMP_DIR';
  temp_dir = getenv(env_temp_dir);
  if isempty(temp_dir)
    temp_dir = fullfile(base_dir, 'temp');
  end
  clear env_temp_dir;
  
  if ~exist(temp_dir, 'dir')
    fprintf('Creating temporary directory\n');
    mkdir(temp_dir);
  end

  fprintf('Temporary directory is %s\n', temp_dir);

  % Get cluster profile from environment variable
  env_hpc_profile = 'HPC_PROFILE_NAME';
  hpc_profile = getenv(env_hpc_profile);
  if isempty(hpc_profile)
    fprintf('No cluster profile specified\n');
  else
    fprintf('Cluster profile is %s\n', hpc_profile);
  end
  clear env_hpc_profile;
  
  % Add other paths
  utils_dir = fullfile(base_dir, 'utility');
  addpath(utils_dir);
  
  networks_dir = fullfile(base_dir, 'networks');
  addpath(networks_dir);

  % Get global parameters
  gp = global_params();
  
catch ex
  % Log exception time and rethrow
  fprintf('ERROR: Unhandled exception in project setup, %s\n', datetime());
  rethrow(ex);
end

fprintf('Project setup complete, %s\n', datetime());

function create_output_dir_helper(out_dir)
  if ~exist(out_dir, 'dir')
    fprintf('Creating output directory\n');
    mkdir(out_dir);
  end
  fprintf('Output directory is %s\n', out_dir);
end
