function [W, LOG_ERRORS, ERRORS_01] = sswg(x,y, x_test, y_test)
  #Regresja logistyczna
  #[x, y] = readdata(train_file,' ');
  #[x_test, y_test] = readdata(test_file,' ');  
  
  ages = 100;
  tic();
  [W, LOG_ERRORS, ERRORS_01] = sswg_train(x, y, ages);
  toc()
  csvwrite('res/W_SGD.txt', W);
  
  x_test = [ones(size(x_test,1),1),x_test];
  Y_pred_test = x_test * W;
  
  #blad 0/1
  ERR = Y_pred_test .* y_test;
  mean_01_error_test = mean(ERR < 0);
  mean_01_error_train = ERRORS_01(size(ERRORS_01,2));
  
  mean_01_error_train
  mean_01_error_test

  #blad logistyczny
  mean_log_error_test = mean(log(1 + e.^((-1) .* y_test .* Y_pred_test)));
  mean_log_error_train = LOG_ERRORS(size(LOG_ERRORS,2));

  mean_log_error_train
  mean_log_error_test
  
  fig = figure();
  #set(fig, "visible", "off");
  plot(1:ages,LOG_ERRORS, "-r;SDG;", 1:ages, 0.2639 * ones(ages, 1), "--b;NR;");
  print(fig, "res/SDG_log_error.png");
  plot(1:ages,ERRORS_01, "-r;SDG;", 1:ages, 0.11632 * ones(ages, 1), "--b;NR;");
  print(fig, "res/SDG_01_error.png");
  hist(W, 100);
  print(fig, "res/SDG_W_hist.png");
end