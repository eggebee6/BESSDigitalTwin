function [recon_fig, action_fig, kl_fig] = loss_plots(csv_filename)
  csvdata = readmatrix(csv_filename);
  start_skip = 1;

  iteration = csvdata(start_skip:end, 3);
  recon_loss = csvdata(start_skip:end, 5);
  action_loss = csvdata(start_skip:end, 6);
  kl_loss = csvdata(start_skip:end, 7);

  recon_fig = figure;
  plot(iteration, recon_loss);
  title('Reconstruction loss');
  xlim('tight');
  ylim([0 1500]);
  xlabel('Iteration');
  ylabel('Loss (MSE)');
  pretty_plot(recon_fig);

  action_fig = figure;
  plot(iteration, action_loss);
  title('Action recommendation loss');
  xlim('tight');
  xlabel('Iteration');
  ylabel('Loss (cross-entropy)');
  pretty_plot(action_fig);

  kl_fig = figure;
  plot(iteration, kl_loss);
  title('KL divergence loss');
  xlim('tight');
  xlabel('Iteration');
  ylabel('Loss (KL divergence)');
  pretty_plot(kl_fig);

  function pretty_plot(fig)
    fig.CurrentAxes.FontSize = 12;
    fig.CurrentAxes.FontWeight = 'bold';
    fig.CurrentAxes.GridAlpha = 0.5;
    %fig.CurrentAxes.FontName = 'Latin Modern';
    %fig.CurrentAxes.XMinorGrid = 'on';
    %fig.CurrentAxes.YMinorGrid = 'on';
    fig.CurrentAxes.XMinorTick = 'on';
    fig.CurrentAxes.YMinorTick = 'on';
    fig.CurrentAxes.LineWidth = 2;
  end

end
