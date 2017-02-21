function [W] = reg_log_alg(X, Y, X_test, Y_test)  
  #[X, Y] = readdata(trainFile, " ");  
  #[X_test, Y_test] = readdata(testFile, " "); 
  tic();
  [X, W] = train_reg_log(X, Y);     #Optymalizacja wektora wag
  toc()
  csvwrite('res/W_NR.txt', W);
  
  Y_pred = X*W;
  
  X_test = [ones(size(X_test,1),1),X_test];
  Y_pred_test = X_test * W;
  
  #blad 0/1
  ERR = Y_pred .* Y;
  mean_01_error_train = mean(ERR < 0);
  
  ERR = Y_pred_test .* Y_test;
  mean_01_error_test = mean(ERR < 0);
  
  mean_01_error_train
  mean_01_error_test
  
  #blad logistyczny
  mean_log_error_train = mean(log(1 + e.^((-1) .* Y .* Y_pred)));  
  mean_log_error_test = mean(log(1 + e.^((-1) .* Y_test .* Y_pred_test)));
  
  mean_log_error_train
  mean_log_error_test
  
  fig = figure();
  #set(fig, "visible", "off");
  hist(W, 100);
  print(fig, "res/NR_W_hist.png");
 
end



