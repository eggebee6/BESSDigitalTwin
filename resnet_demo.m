%% Set up project
clear;
reset_persistent;

fprintf('Starting, %s\n', datetime());

desktop_env = ispc();
has_gui = false;

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

  has_gui = true;
else
  % Set up project for console or non-interactive environment
  project_setup;
end

% Initialize monitor
monitor.name = 'Training monitor';
monitor.gui_monitor = [];
monitor.training_csv_file = fullfile(output_dir, 'training.csv');
monitor.validation_csv_file = fullfile(output_dir, 'validation.csv');
monitor.losses = [];
monitor.epoch = 0;
monitor.iteration = 0;
monitor.console_update_counter = 0;
monitor.console_update_iterations = 25;

% Check for GUI interface
fprintf('Has graphical interface: %s\n', mat2str(has_gui));

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

if (gpus_available > 0)
  mini_batch_size = floor(gp.samples_per_cycle / gp.min_sequence_len) * 64;
else
  mini_batch_size = 3;    % TODO: This is a small value for test purposes only
end

fprintf('Initializing datastores, %s\n', datetime());
tic;

% Get training file set and create source datastore
ds_fileset = matlab.io.datastore.DsFileSet(training_data_dir, ...
  'IncludeSubfolders', true);

ds_source = fileDatastore(ds_fileset, ...
  'ReadFcn', @dtinfo_ds_reader);

num_ds_files = size(ds_source.Files, 1);

% Create an 80/20 split for training/validation
ds_shuffle = randperm(num_ds_files);
ds_split_index = round(0.8 * num_ds_files);

training_ds = subset(ds_source, ds_shuffle(1:ds_split_index));
validation_ds = subset(ds_source, ds_shuffle(ds_split_index+1:end));

% Add transform functions to datastores
training_ds = transform(training_ds, ...
  @(ds_in) dtinfo_ds_transform(ds_in));

validation_ds = transform(validation_ds, ...
  @(ds_in) dtinfo_ds_transform(ds_in));

% Create minibatch queues
% Note: pay attention to formatting!
training_batch_queue = minibatchqueue(training_ds, ...
  'MiniBatchFormat', {'CTB', 'CTB', 'CTB'}, ...
  'MiniBatchSize', mini_batch_size, ...
  'PartialMiniBatch', 'discard');

validation_batch_queue = minibatchqueue(validation_ds, ...
  'MiniBatchFormat', {'CTB', 'CTB', 'CTB'}, ...
  'MiniBatchSize', mini_batch_size, ...
  'PartialMiniBatch', 'discard');

fprintf('Datastores initialized\n');
toc;

%% Create network
fprintf('Creating network, %s\n', datetime());
tic;

model_params = [];

if isfile('debug_model.mat')
  % Load saved model for debugging
  debug_model = load('debug_model.mat');
  model = debug_model.model;
  model_params = debug_model.model_params;
  clear debug_model;

  fprintf('[%s] WARNING: Using debug model\n', datetime());

else
  % Set model parameters
  model_params.filter_size = 3;
  
  model_params.num_res_blocks = 5;

  model_params.encoder_hidden_size = gp.num_features * 16;
  model_params.latent_dims = 6;
  %model_params.latent_dims = gp.num_features;   % TODO: Not this!

  model_params.label_count = num_actions;
  
  % Create model
  [model, training_params] = create_resnet(model_params);
end


%% Train and monitor progress
% Initialize training parameters
training_params.learn_rate = 1e-3;
training_params.monte_carlo_reps = 2;
training_params.best_validation_loss = [];

training_params.using_gpu = gpus_available > 0;

training_params.min_kl_scaling_loss = 10000;

% Initialize output, counters, etc.
create_output_dir();

checkpoint_iteration_count = 250;
checkpoint_counter = 0;

validation_iteration_count = 100;
validation_counter = 0;

% Initialize training loop values
epoch_count = 50;

epoch = 0;
iteration = 0;

% Create GUI monitor if GUI is available
if (has_gui)
  monitor.gui_monitor = create_gui_monitor();
end

% Start training
display_action_values();

fprintf('Starting training, %s\n', datetime());
try
  while epoch < epoch_count && ~stop_requested(monitor)
    % Increment epoch count and reset counters
    epoch = epoch + 1;
    epoch_iteration = 0;
  
    % Shuffle data
    shuffle(training_batch_queue);
    shuffle(validation_batch_queue);

    % Process minibatches until out of data (or stop requested)
    epoch_start_time = tic;
    while hasdata(training_batch_queue) && ~stop_requested(monitor)
      iteration = iteration + 1;
      epoch_iteration = epoch_iteration + 1;

      training_params.epoch = epoch;
      training_params.iteration = iteration;

      grads.iteration = iteration;
  
      % Evaluate and update model with a training batch
      [training_data, error_vectors, labels] = next(training_batch_queue);

      [losses, grads, training_params] = dlfeval(@evaluate_resnet, ...
        model, ...
        training_data, error_vectors, labels, ...
        training_params, grads);

      [model, training_params] = update_resnet(model, losses, grads, training_params);

      %include_reconstruction = false;
      %include_action_rec = true;

      %if (include_reconstruction)
      %  % Reconstruction phase
      %  [losses, grads, training_params] = dlfeval(@evaluate_resnet_recon, ...
      %    model, ...
      %    training_data, error_vectors, ...
      %    training_params, grads);
      %  [model, training_params] = update_resnet_recon(model, losses, grads, training_params);
  
      %  monitor_losses.recon_loss = losses.recon_loss;
      %  monitor_losses.kl_loss = losses.kl_loss;
      %else
      %  monitor_losses.recon_loss = dlarray(0);
      %  monitor_losses.kl_loss = dlarray(0);
      %end

      %if (include_action_rec)
      %  % Action recommendation phase
      %  [losses, grads, training_params] = dlfeval(@evaluate_resnet_action, ...
      %    model, ...
      %    training_data, labels, ...
      %    training_params, grads);
      %  [model, training_params] = update_resnet_action(model, losses, grads, training_params);
  
      %  monitor_losses.action_loss = losses.action_loss;
      %else
      %  monitor_losses.action_loss = dlarray(0);
      %end

      total_loss = ...
        losses.recon_loss + ...
        losses.kl_loss + ...
        losses.action_loss;
      
      monitor_losses.recon_loss = losses.recon_loss;
      monitor_losses.action_loss = losses.action_loss;
      monitor_losses.kl_loss = losses.kl_loss;

      monitor_losses.total_loss = total_loss;
    
      % Update monitor
      monitor.epoch = epoch;
      monitor.iteration = iteration;
      monitor.losses = monitor_losses;
      monitor = update_monitor(monitor);

      % Test performance on validation data
      if (validation_counter > 0)
        validation_counter = validation_counter - 1;
      else
        % Reset counter
        validation_counter = validation_iteration_count - 1;
        
        [training_data, error_vectors, labels] = next(validation_batch_queue);
        perform_validation(model, ...
          training_data, error_vectors, labels, ...
          training_params, monitor);

        % Check for best validation loss
        if isempty(training_params.best_validation_loss)
          training_params.best_validation_loss = total_loss;

        elseif (total_loss < training_params.best_validation_loss)
          % Save model
          best_validation_file = fullfile(output_dir, 'best_model.mat');
          save(best_validation_file, 'model');
      
          training_params.best_validation_loss = total_loss;
        end
      end
  
      % Perform checkpoint operations
      if (checkpoint_counter > 0)
        checkpoint_counter = checkpoint_counter - 1;
      else
        % Reset counter
        checkpoint_counter = checkpoint_iteration_count - 1;

        % Save model
        checkpoint_file = fullfile(output_dir, ...
          sprintf('checkpoint-e%d-i%d.mat', epoch, epoch_iteration));
        save(checkpoint_file, 'model');
      end
  
    end
    epoch_duration = toc(epoch_start_time);

    % Drop learning rate to 90%
    training_params.learn_rate = training_params.learn_rate * 0.9;

    % Save model after every epoch
    try
      model_filename = sprintf('res-epoch-%d.mat', epoch);
      save(fullfile(output_dir, model_filename), 'model');
    catch ex
      fprintf('[%s] ERROR: Failed to save model after epoch: %s\n%s\n', ...
          datetime(), ...
          ex.identifier, ...
          ex.message);
        % TODO: Ignore this exception or do something else with it?
    end

    % Display timing info
    fprintf('Epoch timing: %f seconds per iteration, %d iterations\n', ...
      epoch_duration / epoch_iteration, ...
      epoch_iteration);
  end

  % Done!

catch ex
  % Save model and model parameters for debugging
  save(fullfile(output_dir, 'debug_model.mat'), 'model', ...
    'model_params', 'training_params');
  rethrow(ex);

end


%% Helper functions
function monitor = create_gui_monitor()
  monitor = trainingProgressMonitor( ...
    Metrics = ["Recon", "Action", "KL"], ...
    Info = ["Epoch", "Loss"], ...
    XLabel = "Iteration");
end

function stop = stop_requested(monitor)
  if ~isempty(monitor.gui_monitor)
    % Check for stop from GUI
    stop = monitor.gui_monitor.Stop;
  else
    % TODO: Is there a way to catch an interrupt signal?
    stop = false;
  end
end

function monitor = update_monitor(monitor)
  % Update GUI if it exists
  if ~isempty(monitor.gui_monitor)
    recordMetrics(monitor.gui_monitor, ...
      monitor.iteration, ...
      'Recon', monitor.losses.recon_loss, ...
      'Action', monitor.losses.action_loss, ...
      'KL', monitor.losses.kl_loss);
  
    updateInfo(monitor.gui_monitor, ...
      'Epoch', monitor.epoch, ...
      'Loss', monitor.losses.total_loss);
  end

  % Update console periodically
  if (monitor.console_update_counter > 0)
    monitor.console_update_counter = monitor.console_update_counter - 1;
  else
    % Reset counter
    monitor.console_update_counter = monitor.console_update_iterations - 1;

    % CSV format:
    % time, epoch, iteration, total loss, recon loss, action loss, KL loss
    csv_line = sprintf('%s, %d, %d, %f, %f, %f, %f\n', ...
      datetime(), ...
      monitor.epoch, ...
      monitor.iteration, ...
      extractdata(monitor.losses.total_loss), ...
      extractdata(monitor.losses.recon_loss), ...
      extractdata(monitor.losses.action_loss), ...
      extractdata(monitor.losses.kl_loss));
    
    % Write CSV to console
    fprintf('%s', csv_line);

    % Write CSV to file
    if ~isempty(monitor.training_csv_file)
      csv_file_id = [];
      try
        csv_file_id = fopen(monitor.training_csv_file, 'a');
        fwrite(csv_file_id, csv_line);
      catch ex
        fprintf('[%s] ERROR: Failed to write to CSV file: %s\n%s\n', ...
          datetime(), ...
          ex.identifier, ...
          ex.message);
        % TODO: Ignore this exception or do something else with it?
      end
  
      if ~isempty(csv_file_id)
        fclose(csv_file_id);
      end
    end
  end
end

function [validation_losses] = perform_validation(model, training_data, error_vectors, labels, training_params, monitor)
  grads.iteration = -1;

  [validation_losses, ~, ~] = dlfeval(@evaluate_resnet, ...
    model, ...
    training_data, error_vectors, labels, ...
    training_params, grads);

  % Get losses from evaluation functions
  %[losses, ~, ~] = dlfeval(@evaluate_resnet_recon, ...
  %  model, ...
  %  training_data, error_vectors, ...
  %  training_params, grads);

  %validation_losses.recon_loss = losses.recon_loss;
  %validation_losses.kl_loss = losses.kl_loss;

  % Get losses from evaluation functions
  %[losses, ~, ~] = dlfeval(@evaluate_resnet_action, ...
  %  model, ...
  %  training_data, labels, ...
  %  training_params, grads);

  %validation_losses.action_loss = losses.action_loss;

  %validation_losses.total_loss = ...
  %  validation_losses.recon_loss + ...
  %  validation_losses.kl_loss + ...
  %  validation_losses.action_loss;

  % Display validation info
  fprintf('[%s] Validation losses: Total %f, Recon %f, Action %f, KL %f\n', ...
    datetime(), ...
    extractdata(validation_losses.total_loss), ...
    extractdata(validation_losses.recon_loss), ...
    extractdata(validation_losses.action_loss), ...
    extractdata(validation_losses.kl_loss));

  % CSV format:
  % time, epoch, iteration, total loss, recon loss, action loss, KL loss
  csv_line = sprintf('%s, %d, %d, %f, %f, %f, %f\n', ...
    datetime(), ...
    monitor.epoch, ...
    monitor.iteration, ...
    extractdata(validation_losses.total_loss), ...
    extractdata(validation_losses.recon_loss), ...
    extractdata(validation_losses.action_loss), ...
    extractdata(validation_losses.kl_loss));

  % Write CSV to file
  if ~isempty(monitor.validation_csv_file)
    csv_file_id = [];
    try
      csv_file_id = fopen(monitor.validation_csv_file, 'a');
      fwrite(csv_file_id, csv_line);
    catch ex
      fprintf('[%s] ERROR: Failed to write to CSV file: %s\n%s\n', ...
        datetime(), ...
        ex.identifier, ...
        ex.message);
      % TODO: Ignore this exception or do something else with it?
    end

    if ~isempty(csv_file_id)
      fclose(csv_file_id);
    end
  end
end

function display_action_values()
  fprintf('Labels:\n');
  for i = DTInfo.scenario_action_map.keys
    [~, action_value] = max(DTInfo.get_scenario_label(string(i)));
    fprintf('  %s: %d\n', string(i), action_value);
  end
end
