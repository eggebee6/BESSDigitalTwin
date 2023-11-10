function [model, training_params] = create_resnet2(model_params)
  %% Initialize parameters
  gp = global_params();

  num_res_blocks = model_params.num_res_blocks;
  encoder_hidden_size = model_params.encoder_hidden_size;
  latent_dims = model_params.latent_dims;

  model.encoder_hidden_size = encoder_hidden_size;
  model.latent_dims = latent_dims;
  model.min_sequence_len = gp.min_sequence_len;
  model.label_count = model_params.label_count;

  ar_gru_hidden_size = 16;


  %% Create encoder
  encoder_lgraph = layerGraph();

  enc_input_name = 'enc_input';

  encoder_lgraph = addLayers(encoder_lgraph, [...
    sequenceInputLayer(gp.num_features, 'Name', enc_input_name, ...
      'Normalization', 'none', ...
      'MinLength', gp.min_sequence_len)
  ]);

  % Add resnet-like blocks
  incoming_connection_name = enc_input_name;
  for i = 1:num_res_blocks
    num_res_neurons = latent_dims * (1 + num_res_blocks - i);
    res_filter_size = 1 + 2*(i-1);

    % Add convolution layers
    res_conv1_name = sprintf('enc_res%d_conv1', i);
    res_conv_act_name = sprintf('enc_res%d_conv_act', i);
    res_conv2_name = sprintf('enc_res%d_conv2', i);

    encoder_lgraph = addLayers(encoder_lgraph, [...
      convolution1dLayer(res_filter_size, num_res_neurons, 'Name', res_conv1_name, ...
        'Padding', 'same', ...
        'Stride', 1)
      tanhLayer('Name', res_conv_act_name)
      convolution1dLayer(3, num_res_neurons, 'Name', res_conv2_name, ...
        'Padding', 'same', ...
        'Stride', 1)
    ]);

    encoder_lgraph = connectLayers(encoder_lgraph, incoming_connection_name, res_conv1_name);

    % Add projection layer
    res_fc_proj_name = sprintf('enc_res%d_proj', i);

    encoder_lgraph = addLayers(encoder_lgraph, ...
      fullyConnectedLayer(num_res_neurons, 'Name', res_fc_proj_name));

    encoder_lgraph = connectLayers(encoder_lgraph, incoming_connection_name, res_fc_proj_name);

    % Add summation layers
    res_sum_name = sprintf('enc_res%d_sum', i);
    res_sum_act_name = sprintf('enc_res%d_sum_act', i);

    encoder_lgraph = addLayers(encoder_lgraph, [...
      additionLayer(2, 'Name', res_sum_name)
      tanhLayer('Name', res_sum_act_name)
    ]);

    encoder_lgraph = connectLayers(encoder_lgraph, res_conv2_name, sprintf('%s/in1', res_sum_name));
    encoder_lgraph = connectLayers(encoder_lgraph, res_fc_proj_name, sprintf('%s/in2', res_sum_name));

    % Update incoming connection name
    incoming_connection_name = res_sum_act_name;
  end

  % Add encoder hidden layers
  enc_hidden_name = 'enc_hidden1';
  enc_hidden_act_name = 'enc_hidden1_act';

  encoder_lgraph = addLayers(encoder_lgraph, [...
    fullyConnectedLayer(encoder_hidden_size, 'Name', enc_hidden_name)
    tanhLayer('Name', enc_hidden_act_name)
  ]);

  encoder_lgraph = connectLayers(encoder_lgraph, incoming_connection_name, enc_hidden_name);

  % Add mean/logvar layers and concatenate into parameters layer
  enc_latent_params_name = 'enc_latent_params';
  enc_latent_mean_name = 'enc_latent_mean';
  enc_latent_logvars_name = 'enc_latent_logvar';

  encoder_lgraph = addLayers(encoder_lgraph, ...
    concatenationLayer(1, 2, 'Name', enc_latent_params_name));

  encoder_lgraph = addLayers(encoder_lgraph, ...
    fullyConnectedLayer(latent_dims, 'Name', enc_latent_mean_name));
  encoder_lgraph = connectLayers(encoder_lgraph, enc_hidden_act_name, enc_latent_mean_name);
  encoder_lgraph = connectLayers(encoder_lgraph, enc_latent_mean_name, sprintf('%s/in1', enc_latent_params_name));

  encoder_lgraph = addLayers(encoder_lgraph, ...
    fullyConnectedLayer(latent_dims, 'Name', enc_latent_logvars_name));
  encoder_lgraph = connectLayers(encoder_lgraph, enc_hidden_act_name, enc_latent_logvars_name);  
  encoder_lgraph = connectLayers(encoder_lgraph, enc_latent_logvars_name, sprintf('%s/in2', enc_latent_params_name));


  %% Create latent sampling layers
  % Takes a Gaussian sample using the encoder mean/logvar parameters
  latent_sampler_lgraph = layerGraph();
  latent_sampler_lgraph = addLayers(latent_sampler_lgraph, [...
    sequenceInputLayer(latent_dims * 2, 'Name', 'latent_sampler_input', ...
      'Normalization', 'none')
    gaussianSamplingLayer('Name', 'latent_sample')
  ]);


  %% Create decoder
  decoder_lgraph = layerGraph();

  dec_input_name = 'dec_input';
  dec_latent_name = 'dec_latent_sample';

  decoder_lgraph = addLayers(decoder_lgraph, [...
    sequenceInputLayer(latent_dims, 'Name', dec_input_name, ...
    'Normalization', 'none', ...
    'MinLength', gp.min_sequence_len)

    fullyConnectedLayer(latent_dims, 'Name', dec_latent_name)
  ]);

  % Add hidden layers
  dec_hidden_name = 'dec_hidden';
  dec_hidden_act_name = 'dec_hidden_act';

  decoder_lgraph = addLayers(decoder_lgraph, [...
    tanhLayer('Name', dec_hidden_act_name)
    fullyConnectedLayer(encoder_hidden_size, 'Name', dec_hidden_name)
  ]);

  decoder_lgraph = connectLayers(decoder_lgraph, dec_latent_name, dec_hidden_act_name);

  % Add resnet-like blocks
  incoming_connection_name = dec_hidden_name;
  for i = num_res_blocks:-1:1
    num_res_neurons = latent_dims * (1 + num_res_blocks - i);
    res_filter_size = 1 + 2*(i-1);

    % Add initial activation function
    res_sum_act_name = sprintf('dec_res%d_sum_act', i);
    decoder_lgraph = addLayers(decoder_lgraph, ...
      tanhLayer('Name', res_sum_act_name));

    decoder_lgraph = connectLayers(decoder_lgraph, incoming_connection_name, res_sum_act_name);

    % Add projection layer
    res_fc_proj_name = sprintf('dec_res%d_proj', i);

    decoder_lgraph = addLayers(decoder_lgraph, ...
      fullyConnectedLayer(num_res_neurons, 'Name', res_fc_proj_name));

    decoder_lgraph = connectLayers(decoder_lgraph, res_sum_act_name, res_fc_proj_name);

    % Add convolution layers
    res_conv1_name = sprintf('dec_res%d_tconv1', i);
    res_conv_act_name = sprintf('dec_res%d_tconv_act', i);
    res_conv2_name = sprintf('dec_res%d_tconv2', i);

    decoder_lgraph = addLayers(decoder_lgraph, [...
      transposedConv1dLayer(3, num_res_neurons, 'Name', res_conv2_name, ...
        'Cropping', 'same', ...
        'Stride', 1)
      tanhLayer('Name', res_conv_act_name)
      transposedConv1dLayer(res_filter_size, num_res_neurons, 'Name', res_conv1_name, ...
        'Cropping', 'same', ...
        'Stride', 1)
    ]);

    decoder_lgraph = connectLayers(decoder_lgraph, res_sum_act_name, res_conv2_name);

    % Add summation layers
    res_sum_name = sprintf('dec_res%d_sum', i);

    decoder_lgraph = addLayers(decoder_lgraph, [...
      additionLayer(2, 'Name', res_sum_name)
    ]);

    decoder_lgraph = connectLayers(decoder_lgraph, res_conv1_name, sprintf('%s/in1', res_sum_name));
    decoder_lgraph = connectLayers(decoder_lgraph, res_fc_proj_name, sprintf('%s/in2', res_sum_name));

    % Update incoming connection name
    incoming_connection_name = res_sum_name;
  end

  % Add output layer
  dec_output_name = 'dec_tconv_out';

  decoder_lgraph = addLayers(decoder_lgraph, ...
    transposedConv1dLayer(3, gp.num_err_components, 'Name', dec_output_name, ...
      'Cropping', 'same', ...
      'Stride', 1));
  decoder_lgraph = connectLayers(decoder_lgraph, incoming_connection_name, dec_output_name);


  %% Add action recommender
  action_lgraph = layerGraph();

  ar_input_name = 'ar_input';

  action_lgraph = addLayers(action_lgraph, [...
    sequenceInputLayer(latent_dims, 'Name', ar_input_name, ...
      'Normalization', 'none', ...
      'MinLength', gp.min_sequence_len)
  ]);

  % Add resnet-like blocks
  incoming_connection_name = ar_input_name;
  for i = 3:-1:1
    num_res_neurons = gp.num_features * (1 + 3 - i);
    res_filter_size = 3;

    % Add convolution layers
    res_conv1_name = sprintf('ar_res%d_conv1', i);
    res_conv_act_name = sprintf('ar_res%d_conv_act', i);
    res_conv2_name = sprintf('ar_res%d_conv2', i);
    res_pool_name = sprintf('ar_res%d_pool', i);

    action_lgraph = addLayers(action_lgraph, [...
      convolution1dLayer(res_filter_size, num_res_neurons, 'Name', res_conv1_name, ...
        'Padding', 'same', ...
        'Stride', 1)
      reluLayer('Name', res_conv_act_name)
      maxPooling1dLayer(3, 'Name', res_pool_name, ...
        'Padding', 'same', ...
        'Stride', 1)
      convolution1dLayer(res_filter_size, num_res_neurons, 'Name', res_conv2_name, ...
        'Padding', 'same', ...
        'Stride', 1)
    ]);

    action_lgraph = connectLayers(action_lgraph, incoming_connection_name, res_conv1_name);

    % Add projection layer
    res_fc_proj_name = sprintf('ar_res%d_proj', i);

    action_lgraph = addLayers(action_lgraph, ...
      fullyConnectedLayer(num_res_neurons, 'Name', res_fc_proj_name));

    action_lgraph = connectLayers(action_lgraph, incoming_connection_name, res_fc_proj_name);

    % Add summation layers
    res_sum_name = sprintf('ar_res%d_sum', i);
    res_sum_act_name = sprintf('ar_res%d_sum_act', i);

    action_lgraph = addLayers(action_lgraph, [...
      additionLayer(2, 'Name', res_sum_name)
      tanhLayer('Name', res_sum_act_name)
    ]);

    action_lgraph = connectLayers(action_lgraph, res_conv2_name, sprintf('%s/in1', res_sum_name));
    action_lgraph = connectLayers(action_lgraph, res_fc_proj_name, sprintf('%s/in2', res_sum_name));

    % Update incoming connection name
    incoming_connection_name = res_sum_act_name;
  end
  res_output_name = incoming_connection_name;

  % Add hidden layers
  ar_hidden_conv_name = 'ar_hidden_conv';
  ar_hidden_conv_act_name = 'ar_hidden_conv_act';
  ar_gru_name = 'ar_hidden_gru';
  ar_gru_act_name = 'ar_hidden_gru_act';
  ar_hidden1_name = 'ar_hidden1';
  ar_hidden1_act_name = 'ar_hidden1_act';
  ar_hidden2_name = 'ar_hidden2';
  ar_hidden2_act_name = 'ar_hidden2_act';
  ar_hidden_output = 'ar_hidden_out';
  ar_output_name = 'ar_output';

  action_lgraph = addLayers(action_lgraph, [...
    convolution1dLayer(3, gp.num_features, 'Name', ar_hidden_conv_name, ...
      'Padding', 'same', ...
      'Stride', 1)
    reluLayer('Name', ar_hidden_conv_act_name)

    fullyConnectedLayer(gp.num_features, 'Name', ar_hidden1_name)
    reluLayer('Name', ar_hidden1_act_name)

    gruLayer(ar_gru_hidden_size, 'Name', ar_gru_name, ...
      'OutputMode', 'sequence')
    reluLayer('Name', ar_gru_act_name)

    fullyConnectedLayer(encoder_hidden_size, 'Name', ar_hidden2_name)
    reluLayer('Name', ar_hidden2_act_name)

    fullyConnectedLayer(model_params.label_count, 'Name', ar_hidden_output)    
    softmaxLayer('Name', ar_output_name)
  ]);

  action_lgraph = connectLayers(action_lgraph, res_output_name, ar_hidden_conv_name);


  %% Assemble the model
  % Put all the networks into the model struct
  model.encoder = dlnetwork(encoder_lgraph);
  model.latent_sampler = dlnetwork(latent_sampler_lgraph);
  model.decoder = dlnetwork(decoder_lgraph);
  model.action_recommender = dlnetwork(action_lgraph);

  %% Set default training parameters
  training_params.enc_grad_avg = [];
  training_params.enc_grad_avg2 = [];
  training_params.dec_grad_avg = [];
  training_params.dec_grad_avg2 = [];

  training_params.act_grad_avg = [];
  training_params.act_grad_avg2 = [];

  training_params.recon_loss_factor = 1;
  training_params.kl_loss_factor = 1;
  training_params.action_loss_factor = 1;

end
