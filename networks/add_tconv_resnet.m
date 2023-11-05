function [lgraph, input_name, output_name] = add_tconv_resnet(lgraph, block_name, input_size, kvargs)
% [lgraph, input_name, output_name] = add_tconv_resnet(lgraph, block_name, input_size)
% [lgraph, input_name, output_name] = add_tconv_resnet(__, Name, Value)
%
% Adds transposed convolutional resnet layers to a layer graph
%
% The input and output layers are left unconnected and must be connected after
% add_tconv_resnet is called.  The names of the input and output layers are returned
% through input_name and output_name, respectively.
%
% Arguments
% lgraph - Layer graph which will have the resnet block connected
% block_name - Name of resnet block (must be unique within its graph)
% input_size - Dimension of input
%
% Name/Value options
% FilterSize - Convolutional filter size
%
% Outputs
% lgraph - Layer graph with resnet layers added (input and output layers are
% left unconnected)
% input_name - Name of block input layer (to be connected by caller)
% output_name - Name of block output layer (to be connected by caller)
  arguments
    lgraph = [];
    block_name = [];
    input_size = [];

    kvargs.FilterSize = 3;
  end

  if isempty(lgraph)
    error('Missing lgraph argument');
  elseif isempty(block_name)
    error('Missing block_name argument');
  elseif isempty(input_size)
    error('Missing input_size argument');
  end

  filter_size = kvargs.FilterSize;

  % Initialize layer names
  input_name = sprintf('%s_input', block_name);

  projection_name = sprintf('%s_proj', block_name);
  
  conv_1_name = sprintf('%s_tconv1', block_name);
  conv_act_name = sprintf('%s_tanh_tconv1', block_name);
  conv_2_name = sprintf('%s_tconv2', block_name);

  addition_name = sprintf('%s_sum', block_name);
  
  output_name = sprintf('%s_output', block_name);

  % Add input layer
  lgraph = addLayers(lgraph, tanhLayer('Name', input_name));

  % Add projection layer and connect the input
  lgraph = addLayers(lgraph, ...
    transposedConv1dLayer(1, input_size, 'Name', projection_name, ...
      'Cropping', 'same', ...
      'Stride', 1));
  lgraph = connectLayers(lgraph, input_name, projection_name);

  % Add convolution layers and connect the input
  lgraph = addLayers(lgraph, [...
    transposedConv1dLayer(filter_size, input_size, 'Name', conv_2_name, ...
      'Cropping', 'same', ...
      'Stride', 1)
    tanhLayer('Name', conv_act_name)
    transposedConv1dLayer(filter_size, input_size, 'Name', conv_1_name, ...
      'Cropping', 'same', ...
      'Stride', 1)
  ]);
  lgraph = connectLayers(lgraph, input_name, conv_2_name);

  % Add summation layer and connect projection and resnet layers
  lgraph = addLayers(lgraph, additionLayer(2, 'Name', addition_name));
  lgraph = connectLayers(lgraph, projection_name, sprintf('%s/in1', addition_name));
  lgraph = connectLayers(lgraph, conv_1_name, sprintf('%s/in2', addition_name));

  % Add output layer
  lgraph = addLayers(lgraph, ...
    transposedConv1dLayer(1, input_size, 'Name', output_name, ...
      'Cropping', 'same', ...
      'Stride', 1));
  lgraph = connectLayers(lgraph, addition_name, output_name);
end
