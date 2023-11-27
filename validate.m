%% Set up project
clear;
reset_persistent;

fprintf('Starting, %s\n', datetime());

desktop_env = ispc();

if (desktop_env)
  % Set up project for desktop environment

  % Override default data directory to local storage
  data_dir_override = 'D:/data/';
  if ~exist(data_dir_override, 'dir')
    error('Data override directory does not exist');
  end
  setenv('NANOGRID_DATA_DIR', data_dir_override);
  
  project_setup;
  
  setenv('NANOGRID_DATA_DIR');  % Clear data directory override

else
  % Set up project for console or non-interactive environment
  project_setup;
end

% Check for GPUs
gpus_available = gpuDeviceCount("available");

if (gpus_available > 0)
  fprintf('Using GPU\n');
  executionEnvironment = "gpu";
else
  fprintf('No GPUs found\n');
end

% Get global parameters
gp = global_params();


%% Set up datastore

% Initialize scenario labels
num_actions = DTInfo.initialize_scenario_labels(training_data_dir);

% Set minibatch size
if (gpus_available > 0)
  mini_batch_size = floor(gp.samples_per_cycle / gp.min_sequence_len) * 64;
else
  mini_batch_size = 3;    % TODO: This is a small value for test purposes only
end

fprintf('Initializing datastore, %s\n', datetime());
tic;

% Get training file set and create source datastore
ds_fileset = matlab.io.datastore.DsFileSet(training_data_dir, ...
  'IncludeSubfolders', true);

ds_source = fileDatastore(ds_fileset, ...
  'ReadFcn', @validation_ds_reader);

num_ds_files = size(ds_source.Files, 1);

% Create minibatch queue
% Note: pay attention to formatting!
validation_batch_queue = minibatchqueue(ds_source, ...
  'MiniBatchFormat', {'CTB', 'CTB', 'CB'}, ...
  'MiniBatchSize', mini_batch_size, ...
  'PartialMiniBatch', 'discard');

fprintf('Datastore initialized\n');
toc;


%% Validate

action_names = [...
  "Disconnect (bus fault)", ...
  "Grid form (gen fault)", ...
  "No action (IM load)", ...
  "Ride thru (load fault)", ...
  "No action", ...
  "Grid form (PM loss)", ...
];

create_output_dir();

try
  % Load model
  model_file = load('validation_model.mat', 'model');
  model = model_file.model;
  
  % Get confusion matrices, accumulate total confusion
  fprintf('Getting confusion matrices, %s\n', datetime());
  
  batch = 0;
  total_confusion = zeros(num_actions, num_actions);

  shuffle(validation_batch_queue);
  
  tic;
  while hasdata(validation_batch_queue)
    batch = batch + 1;

    [confusion_mat] = dlfeval(@get_confusion_matrix, model, validation_batch_queue);
    total_confusion = total_confusion + confusion_mat;

    if (mod(batch, 10) == 0)
      elapsed_time = toc;
      fprintf('Batch %d, time %f\n', batch, elapsed_time);

      save(fullfile(output_dir, 'confusion_matrix.mat'), 'total_confusion');

      tic;
    end
  end

  save(fullfile(output_dir, 'confusion_matrix.mat'), 'total_confusion');
  %confusionchart(total_confusion, action_names, 'Normalization', 'row-normalized');

catch ex

  % Save model and model parameters for debugging
  save(fullfile(output_dir, 'debug_confusion.mat'), 'total_confusion');
  rethrow(ex);
end
