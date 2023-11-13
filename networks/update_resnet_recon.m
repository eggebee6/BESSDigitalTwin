function [model, training_params] = update_resnet_recon(model, losses, grads, training_params)
%% Update network
try
  % Debug stuff
  %if check_learnables(model.encoder.Learnables)
  %  error('Bad encoder weights before update');
  %end

  % Update encoder
  [model.encoder, training_params.enc_grad_recon_avg, training_params.enc_grad_recon_avg2] = adamupdate(...
    model.encoder, grads.encoder_recon, ...
    training_params.enc_grad_recon_avg, training_params.enc_grad_recon_avg2, ...
    training_params.iteration, training_params.learn_rate);

  % Debug stuff
  %if check_learnables(model.encoder.Learnables)
  %  error('Bad encoder weights after update');
  %end

  % Debug stuff
  %if check_learnables(model.decoder.Learnables)
  %  error('Bad decoder weights before update');
  %end

  % Update decoder
  [model.decoder, training_params.dec_grad_avg, training_params.dec_grad_avg2] = adamupdate(...
    model.decoder, grads.decoder, ...
    training_params.dec_grad_avg, training_params.dec_grad_avg2, ...
    training_params.iteration, training_params.learn_rate);

  % Debug stuff
  %if check_learnables(model.decoder.Learnables)
  %  error('Bad decoder weights after update');
  %end

catch ex
  save('update_recon_debug.mat', ...
    'model', 'losses', 'grads', 'training_params');
  rethrow(ex);  
end

%% Helper functions
% Debug stuff
function [bad] = check_learnables(learnables)
  bad = false;

  for i = 1:height(learnables)
    if any(~isfinite(learnables{i, 3}{:}), 'all')
      bad = true;
      break;
    end
  end
  
end

end
