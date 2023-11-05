function [model, training_params] = create_resnet(model_params)
  %% Initialize parameters
  gp = global_params();

  filter_size = model_params.filter_size;
  num_filters = model_params.num_filters;
  num_res_blocks = model_params.num_res_blocks;

  encoder_hidden_size = model_params.encoder_hidden_size;
  latent_dims = model_params.latent_dims;
  total_downsampling = 2 ^ num_res_blocks;

  model.encoder_hidden_size = encoder_hidden_size;
  model.latent_dims = latent_dims;
  model.total_downsampling = total_downsampling;
  model.min_sequence_len = gp.min_sequence_len;


  %% Create encoder
  encoder_lgraph = layerGraph();
  encoder_lgraph = addLayers(encoder_lgraph, [...
    sequenceInputLayer(gp.num_features, 'Name', 'enc_input', ...
      'Normalization', 'none', ...
      'MinLength', gp.min_sequence_len)
    convolution1dLayer(filter_size, num_filters, 'Name', 'enc_conv_in', ...
      'Padding', 'same', ...
      'Stride', 1)
  ]);

  % Add resnet blocks and downsampling layers to encoder graph
  incoming_connection_name = 'enc_conv_in';
  for i = 1:num_res_blocks
    % Add resnet block and connect incoming connection
    res_block_name = sprintf('enc_res%d', i);
    [encoder_lgraph, res_input_name, res_output_name] = add_conv_resnet(...
      encoder_lgraph, res_block_name, num_filters, ...
      'FilterSize', filter_size);
    encoder_lgraph = connectLayers(encoder_lgraph, incoming_connection_name, res_input_name);

    % Add downsampling and connect block output
    conv_name = sprintf('enc_conv_ds%d', i);
    conv_act_name = sprintf('enc_tanh_ds%d', i);
    encoder_lgraph = addLayers(encoder_lgraph, [...
      convolution1dLayer(filter_size, num_filters, 'Name', conv_name, ...
        'Padding', 'same', ...
        'Stride', 2)
      tanhLayer('Name', conv_act_name)
    ]);
    encoder_lgraph = connectLayers(encoder_lgraph, res_output_name, conv_name);

    % Update incoming connection name
    incoming_connection_name = conv_act_name;
  end
  resnet_output_name = incoming_connection_name;

  % Add fully connected layers and connect last resnet block
  encoder_lgraph = addLayers(encoder_lgraph, [...
    fullyConnectedLayer(encoder_hidden_size, 'Name', 'enc_fc1')
    tanhLayer('Name', 'enc_tanh_fc1')
    fullyConnectedLayer(encoder_hidden_size * 4, 'Name', 'enc_fc2')
    tanhLayer('Name', 'enc_tanh_fc2')
    fullyConnectedLayer(encoder_hidden_size, 'Name', 'enc_fc3')
  ]);
  encoder_lgraph = connectLayers(encoder_lgraph, resnet_output_name, 'enc_fc1');

  % Add GRU projection layer and connect last resnet block
  encoder_lgraph = addLayers(encoder_lgraph, ...
    gruLayer(encoder_hidden_size, 'Name', 'enc_fc_gru', ...
      'OutputMode', 'sequence'));
  encoder_lgraph = connectLayers(encoder_lgraph, resnet_output_name, 'enc_fc_gru');

  % Add bypass layers
  encoder_lgraph = addLayers(encoder_lgraph, [...
    gruLayer(num_filters, 'Name', 'enc_bypass_gru', ...
      'OutputMode', 'sequence')
    convolution1dLayer(total_downsampling, encoder_hidden_size, 'Name', 'enc_bypass_conv', ...
      'Padding', 'same', ...
      'Stride', total_downsampling)
  ]);
  encoder_lgraph = connectLayers(encoder_lgraph, 'enc_conv_in', 'enc_bypass_gru');

  % Add summation and encoder hidden layers
  encoder_lgraph = addLayers(encoder_lgraph, [...
    additionLayer(3, 'Name', 'enc_add')
    tanhLayer('Name', 'enc_tanh_add')
    fullyConnectedLayer(encoder_hidden_size, 'Name', 'enc_hidden')
  ]);

  % Connect fully connected and GRU layers to summation layer
  encoder_lgraph = connectLayers(encoder_lgraph, 'enc_fc3', 'enc_add/in1');
  encoder_lgraph = connectLayers(encoder_lgraph, 'enc_fc_gru', 'enc_add/in2');
  encoder_lgraph = connectLayers(encoder_lgraph, 'enc_bypass_conv', 'enc_add/in3');
  
  % Add mean/logvar layers
  encoder_lgraph = addLayers(encoder_lgraph, fullyConnectedLayer(latent_dims, 'Name', 'latent_mean'));
  encoder_lgraph = addLayers(encoder_lgraph, fullyConnectedLayer(latent_dims, 'Name', 'latent_logvars'));
  encoder_lgraph = connectLayers(encoder_lgraph, 'enc_hidden', 'latent_mean');
  encoder_lgraph = connectLayers(encoder_lgraph, 'enc_hidden', 'latent_logvars');

  encoder_lgraph = addLayers(encoder_lgraph, concatenationLayer(1, 2, 'Name', 'latent_params'));
  encoder_lgraph = connectLayers(encoder_lgraph, 'latent_mean', 'latent_params/in1');
  encoder_lgraph = connectLayers(encoder_lgraph, 'latent_logvars', 'latent_params/in2');


  %% Create latent sampling layers
  % Takes a Gaussian sample using the encoder mean/logvar as parameters
  latent_sampler_lgraph = layerGraph();
  latent_sampler_lgraph = addLayers(latent_sampler_lgraph, [...
    sequenceInputLayer(latent_dims * 2, 'Name', 'latent_sampler_input', ...
      'Normalization', 'none')
    gaussianSamplingLayer('Name', 'latent_sample')
  ]);


  %% Create decoder
  decoder_lgraph = layerGraph();
  decoder_lgraph = addLayers(decoder_lgraph, [...
    sequenceInputLayer(latent_dims, 'Name', 'dec_input', ...
      'Normalization', 'none')

    fullyConnectedLayer(encoder_hidden_size, 'Name', 'dec_fc_in')
    tanhLayer('Name', 'dec_tanh_in')
  ]);

  % Add fully connected layers and connect input
  decoder_lgraph = addLayers(decoder_lgraph, [...
    fullyConnectedLayer(encoder_hidden_size, 'Name', 'dec_fc3')
    tanhLayer('Name', 'dec_tanh_fc2')
    fullyConnectedLayer(encoder_hidden_size * 4, 'Name', 'dec_fc2')
    tanhLayer('Name', 'dec_tanh_fc1')
    fullyConnectedLayer(encoder_hidden_size, 'Name', 'dec_fc1')
  ]);
  decoder_lgraph = connectLayers(decoder_lgraph, 'dec_tanh_in', 'dec_fc3');

  % Add bypass layers
  decoder_lgraph = addLayers(decoder_lgraph, [...
    transposedConv1dLayer(total_downsampling, encoder_hidden_size, 'Name', 'dec_bypass_tconv', ...
      'Cropping', 'same', ...
      'Stride', total_downsampling)
    gruLayer(num_filters, 'Name', 'dec_bypass_gru', ...
      'OutputMode', 'sequence')
  ]);
  decoder_lgraph = connectLayers(decoder_lgraph, 'dec_tanh_in', 'dec_bypass_tconv');

  % Add GRU layer and connect input
  decoder_lgraph = addLayers(decoder_lgraph, ...
    gruLayer(encoder_hidden_size, 'Name', 'dec_fc_gru', ...
      'OutputMode', 'sequence'));
  decoder_lgraph = connectLayers(decoder_lgraph, 'dec_tanh_in', 'dec_fc_gru');

  % Add fully connected/GRU summation layer
  decoder_lgraph = addLayers(decoder_lgraph, additionLayer(2, 'Name', 'dec_add1'));
  decoder_lgraph = connectLayers(decoder_lgraph, 'dec_fc1', 'dec_add1/in1');
  decoder_lgraph = connectLayers(decoder_lgraph, 'dec_fc_gru', 'dec_add1/in2');

  % Add transposed resnet blocks and upsampling layers to decoder graph
  incoming_connection_name = 'dec_add1';
  for i = num_res_blocks:-1:1
    % Add upsampling
    conv_name = sprintf('dec_tconv_us%d', i);
    conv_act_name = sprintf('dec_tanh_us%d', i);
    decoder_lgraph = addLayers(decoder_lgraph, [...
      tanhLayer('Name', conv_act_name)
      transposedConv1dLayer(filter_size, num_filters, 'Name', conv_name, ...
        'Cropping', 'same', ...
        'Stride', 2)
    ]);
    decoder_lgraph = connectLayers(decoder_lgraph, incoming_connection_name, conv_act_name);

    % Add resnet block
    res_block_name = sprintf('dec_res%d', i);
    [decoder_lgraph, res_input_name, res_output_name] = add_tconv_resnet(...
      decoder_lgraph, res_block_name, num_filters, ...
      'FilterSize', filter_size);
    decoder_lgraph = connectLayers(decoder_lgraph, conv_name, res_input_name);

    incoming_connection_name = res_output_name;
  end
  resnet_output_name = incoming_connection_name;

  % Add bypass summation layer
  decoder_lgraph = addLayers(decoder_lgraph, additionLayer(2, 'Name', 'dec_add2'));
  decoder_lgraph = connectLayers(decoder_lgraph, resnet_output_name, 'dec_add2/in1');
  decoder_lgraph = connectLayers(decoder_lgraph, 'dec_bypass_gru', 'dec_add2/in2');

  % Add output layer
  decoder_lgraph = addLayers(decoder_lgraph, ...
    transposedConv1dLayer(filter_size, gp.num_features, 'Name', 'dec_tconv_out', ...
      'Cropping', 'same', ...
      'Stride', 1));
  decoder_lgraph = connectLayers(decoder_lgraph, 'dec_add2', 'dec_tconv_out');


  %% Create discriminator

  %% Assemble the model
  % Put all the networks into the model struct
  model.encoder = dlnetwork(encoder_lgraph);
  model.latent_sampler = dlnetwork(latent_sampler_lgraph);
  model.decoder = dlnetwork(decoder_lgraph);

  %% Set default training parameters
  training_params.enc_grad_avg = [];
  training_params.enc_grad_avg2 = [];
  training_params.dec_grad_avg = [];
  training_params.dec_grad_avg2 = [];

end
