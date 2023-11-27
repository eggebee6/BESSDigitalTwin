function [fig] = demo_model(model, dt_info, num_cycles)
  arguments
    model = [];
    dt_info = [];
    num_cycles = 1;
  end
  gp = global_params();

  x_range = 1:(gp.samples_per_cycle * num_cycles);

  testing_data = DTInfo.get_model_input(dt_info);
  testing_data = testing_data(:, :, x_range);

  scenario_name = DTInfo.get_scenario_name(dt_info);
  correct_action = DTInfo.get_scenario_label(scenario_name);
  vgrid = DTInfo.get_vgrid(dt_info) ./ gp.voltage_pu;

  event_timestep = DTInfo.get_event_timestep(dt_info);
  if isempty(event_timestep)
    event_timestep = 0;
  end
  
  dim_C = 1;
  dim_B = 2;
  dim_T = 3;
  size_C = size(testing_data, dim_C);
  size_T = size(testing_data, dim_T);

  % Only use first sample from batch
  testing_data = testing_data(:, 1, :);
  size_B = 1;

  %% Forward data through model
  % Pass entire sequence through model
  [action_output, decoder_output] = model_predict(model, dt_info);

  % Get reconstruction from decoder output
  recon_data = decoder_output;

  [~, action_data] = max(action_output);
  
  % Reshape for convenience
  testing_data = reshape(testing_data, [size_C size_T]);
  recon_data = reshape(recon_data, [size(recon_data, 1) size(recon_data, 3)]);
  action_data = reshape(action_data, [size(action_data, 1) size(action_data, 3)]);

  % Truncate data ranges to shorter length
  max_len = min([size(testing_data, 2), size(recon_data, 2)]);
  testing_data = testing_data(:, 1:max_len);
  recon_data = recon_data(:, 1:max_len);
  vgrid = vgrid(1:max_len, :)';

  testing_err_vec = testing_data(1:9, :);

  %% Plot results
  % Set plot parameters
  x_range = (1:max_len) ./ gp.Fs;
  start_skip = 2;

  y_lim_ilf = max([...
    max(testing_err_vec(1:3, start_skip:end), [], 'all'), ...
    max(recon_data(1:3, start_skip:end), [], 'all'), ...
    10
  ]);
  %y_lim_ilf = [0 extractdata(y_lim_ilf)];
  y_lim_ilf = [0 20];

  y_lim_vcf = max([...
    max(testing_err_vec(4:6, start_skip:end), [], 'all'), ...
    max(recon_data(4:6, start_skip:end), [], 'all'), ...
    4
  ]);
  %y_lim_vcf = [0 extractdata(y_lim_vcf)];
  y_lim_vcf = [0 10];

  y_lim_ilo = max([...
    max(testing_err_vec(7:9, start_skip:end), [], 'all'), ...
    max(recon_data(7:9, start_skip:end), [], 'all'), ...
    4
  ]);
  %y_lim_ilo = [0 extractdata(y_lim_ilo)];
  y_lim_ilo = [0 14];

  y_lim_action = [0 model.label_count + 1];

  % Misc calculations
  [~, max_action] = max(correct_action);
  action_data_len = size(action_data, 2);

  action_event_time = max([1, floor(event_timestep / 16)]);
  correct_action = repmat(max_action, size_B, action_data_len);
  [~, no_action] = max(DTInfo.get_scenario_label("No events"));
  correct_action(:, 1:action_event_time) = no_action;

  action_data_len = min([floor(max_len/16), action_data_len]);
  action_data = action_data(:, 1:action_data_len);
  correct_action = correct_action(:, 1:action_data_len);
  action_x_range = 16*(1:action_data_len) ./ gp.Fs;

  rec_hist = histcounts(extractdata(action_data(:, action_event_time:end)), 1:model.label_count+1);
  rec_hist = rec_hist ./ sum(rec_hist);

  % Scale timestep to actual time
  event_timestep = event_timestep / gp.Fs;

  % Create plots
  fig = tiledlayout(3, 2);
  title(fig, sprintf('%s', scenario_name));

  % Plot iLf error
  nexttile();
  hold on;
  data_range = 1:3;
  plot(...
    x_range, testing_err_vec(data_range(1), :)', 'r-', ...
    x_range, testing_err_vec(data_range(2), :)', 'g-', ...
    x_range, testing_err_vec(data_range(3), :)', 'b-');
  plot(...
    x_range, recon_data(data_range(1), :)', 'r.', ...
    x_range, recon_data(data_range(2), :)', 'g.', ...
    x_range, recon_data(data_range(3), :)', 'b.');
  
  xline(event_timestep, 'k--');

  xlim('tight');
  ylim(y_lim_ilf);
  title('Lf error');
  xlabel('Time (s)');
  ylabel('Current (A)');

  % Plot grid voltage
  nexttile();
  hold on;
  plot(...
    x_range, vgrid(1, :)', 'r-', ...
    x_range, vgrid(2, :)', 'g-', ...
    x_range, vgrid(3, :)', 'b-');
  
  xline(event_timestep, 'k--');

  xlim('tight');
  ylim([-4.5 4.5]);
  title('Grid voltage p.u.');
  xlabel('Time (s)');
  ylabel('Voltage (V)');

  % Plot vCf error
  nexttile();
  hold on;
  data_range = 4:6;
  plot(...
    x_range, testing_err_vec(data_range(1), :)', 'r-', ...
    x_range, testing_err_vec(data_range(2), :)', 'g-', ...
    x_range, testing_err_vec(data_range(3), :)', 'b-');
  plot(...
    x_range, recon_data(data_range(1), :)', 'r.', ...
    x_range, recon_data(data_range(2), :)', 'g.', ...
    x_range, recon_data(data_range(3), :)', 'b.');

  xline(event_timestep, 'k--');

  xlim('tight');
  ylim(y_lim_vcf);
  title('Cf error');
  xlabel('Time (s)');
  ylabel('Voltage (V)');

  % Plot action recommendations over time
  nexttile;
  plot(...
    action_x_range, action_data(1, :)', '.', ...
    action_x_range, correct_action', '-');

  xline(event_timestep, 'k--');

  xlim('tight');
  ylim(y_lim_action);

  % TODO: Better y-axis labelling
  action_names = [...
    "Disconnect (bus fault)", ...
    "Grid form (gen fault)", ...
    "No action (IM load)", ...
    "Ride thru (load fault)", ...
    "No action", ...
    "Grid form (PM loss)", ...
  ];

  ax = gca;
  ax.YGrid = 'on';
  ax.GridLineStyle = '--';
  ax.YTick = 1:7;
  ax.YTickLabel = action_names;
  ax.YAxisLocation = 'right';

  title('Recommendations over time');
  xlabel('Time (s)');
  %ylabel('Action')

  % Plot iLo error
  nexttile();
  hold on;
  data_range = 7:9;
  plot(...
    x_range, testing_err_vec(data_range(1), :)', 'r-', ...
    x_range, testing_err_vec(data_range(2), :)', 'g-', ...
    x_range, testing_err_vec(data_range(3), :)', 'b-');
  plot(...
    x_range, recon_data(data_range(1), :)', 'r.', ...
    x_range, recon_data(data_range(2), :)', 'g.', ...
    x_range, recon_data(data_range(3), :)', 'b.');

  xline(event_timestep, 'k--');

  xlim('tight');
  ylim(y_lim_ilo);
  title('Lo error');
  xlabel('Time (s)');
  ylabel('Current (A)');

  % Plot action recommendation histograms
  nexttile;
  bar(rec_hist' .* 100);
  xline(max_action, 'k-');
  yline(max(rec_hist .* 100), 'k:');
  ylim([0 100]);
  title('Recommendations');
  xlabel('Recommendation');
  ylabel('Post-event %');
  ytickformat('percentage');
  legend('Recommended', 'Correct', ...
    'Location', 'northeastoutside');

end
