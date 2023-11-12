function [gp] = global_params()
  persistent params;
  
  if isempty(params)
    params.debug_enabled = true;    % TODO: Fancy debug toggling
    
    params.Ts = 10e-6;
    params.Fs = 1 / params.Ts;

    params.frequency = 60;
    params.cycle_time = 1 / params.frequency;
    params.samples_per_cycle = ceil(params.cycle_time * params.Fs);

    % Per-unit scales for current and voltage
    params.voltage_pu = 391.9184;  % ESS.VSC.Vbin
    params.current_pu = 378.0077;  % ESS.VSC.Ibin

    % Features:
    % err_iLf_abc, err_vCf_abc, err_iLo_abc
    % meas_iLf_abc, meas_vCf_abc, meas_iLo_abc
    % vgrid
    params.num_features = 21;
    
    params.num_err_components = 9;

    % Scale values for error vectors
    params.iLf_err_scale = 8;
    params.vCf_err_scale = 1;
    params.iLo_err_scale = 1;

    % Minimum number of samples needed for sequence processing
    params.min_sequence_len = 16 * floor(params.samples_per_cycle / 16);
    params.strides_per_sequence = 3;
    
  end
  gp = params;
end
