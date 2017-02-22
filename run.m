function [] = run(X, Y, X_test, Y_test)
   #[X, Y] = readdata("file5_train.txt", " ");  
   #[X_test, Y_test] = readdata("file5_test.txt", " "); 
   
   WNR = reg_log_alg(X, Y, X_test, Y_test);  # regresja logistyczna N-R

   [WSGD, LOG_ERRORS, ERRORS_01] = sswg(X, Y, X_test, Y_test);  # SDG 
   
   distance = sqrt(sum((WNR .- WSGD).^2))  
  
end