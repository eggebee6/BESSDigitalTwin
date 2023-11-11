function [model, training_params] = update_resnet_action(model, losses, grads, training_params)
%% Update network
try
  % Debug stuff
  %if check_learnables(model.encoder.Learnables)
  %  error('Bad encoder weights before update');
  %end

  % Update encoder
  [model.encoder, training_params.enc_grad_act_avg, training_params.enc_grad_act_avg2] = adamupdate(...
    model.encoder, grads.encoder_action, ...
    training_params.enc_grad_act_avg, training_params.enc_grad_act_avg2, ...
    training_params.iteration, training_params.learn_rate);

  % Debug stuff
  %if check_learnables(model.encoder.Learnables)
  %  error('Bad encoder weights after update');
  %end

  % Debug stuff
  %if check_learnables(model.action_recommender.Learnables)
  %  error('Bad action recommender weights before update');
  %end

  % Update action recommender
  [model.action_recommender, training_params.act_grad_avg, training_params.act_grad_avg2] = adamupdate(...
    model.action_recommender, grads.action_recommender, ...
    training_params.act_grad_avg, training_params.act_grad_avg2, ...
    training_params.iteration, training_params.learn_rate);

  % Debug stuff
  %if check_learnables(model.action_recommender.Learnables)
  %  error('Bad action recommender weights after update');
  %end

catch ex
  save('update_action_debug.mat', ...
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
