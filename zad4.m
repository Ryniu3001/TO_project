function [] = zad4(X, Y, X_test, Y_test)  
  lambda = 0;
  lambdaMax = 100;
  err_01_train = [];
  err_01_test = [];
  err_log_train = [];
  err_log_test = [];
  while (lambda <= lambdaMax)

    [X, W] = train_reg_log(X, Y, true, lambda);     #Optymalizacja wektora wag
    
    Y_pred = X*W;
    
    X_test = [ones(size(X_test,1),1),X_test];
    Y_pred_test = X_test * W;
    
    #blad 0/1
    ERR = Y_pred .* Y;
    mean_01_error_train = mean(ERR < 0);
    ERR = Y_pred_test .* Y_test;
    mean_01_error_test = mean(ERR < 0);
    #blad logistyczny
    mean_log_error_train = mean(log(1 + e.^((-1) .* Y .* Y_pred)));  
    mean_log_error_test = mean(log(1 + e.^((-1) .* Y_test .* Y_pred_test)));
    
    err_01_train = [err_01_train, mean_01_error_train];
    err_01_test = [err_01_test, mean_01_error_test];
    err_log_train = [err_log_train, mean_log_error_train];
    err_log_test = [err_log_test, mean_log_error_test];
    lambda += 1;
  endwhile
  
  fig = figure();
  #set(fig, "visible", "off");
  plot(0:lambdaMax,err_01_train, "-r;01train;", 0:lambdaMax,err_01_test, "-b;01test;");
  print(fig, "err_01_lambda.png");
  
  plot(0:lambdaMax,err_log_train, "-r;logTrain;", 0:lambdaMax,err_log_test, "-b;logTest;");
  print(fig, "err_log_train.png");  
 
end