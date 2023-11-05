classdef gaussianSamplingLayer < nnet.layer.Layer

  methods
    function layer = gaussianSamplingLayer(args)
      % layer = gaussianSamplingLayer()
      % layer = gaussianSamplingLayer(args)
      %
      % Create a layer which samples a value from a Gaussian distribution
      %  whos parameters are means and log-variances provided through the
      %  input layer
      %
      % Optional arguments:
      %  Name - Name of the layer

      arguments
        args.Name = "gaussian_sampling";
      end

      layer.Name = args.Name;
      layer.Type = "Sampling";
      layer.Description = "Gaussian sampler";
      %layer.OutputNames = ["sample"];
    end

    function [sample] = predict(~, input)
      % [sample] = predict(~, input)

      % Assumes input format is CBT
      C = size(input, 1);
      B = size(input, 2);
      T = size(input, 3);

      means = input(1:C/2, :, :);
      logvars = input((C/2)+1:end, :, :);

      epsilons = randn(C/2, B, T, ...
        like = input);
      sigmas = exp(0.5 * logvars);

      % Sample value
      sample = dlarray(epsilons .* sigmas + means);
    end

  end

end