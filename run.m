function [] = run()
   [X, Y] = readdata("file3_train.txt", " ");  
   [X_test, Y_test] = readdata("file3_test.txt", " "); 
   
   WNR = reg_log_alg(X, Y, X_test, Y_test);  # regresja logistyczna N-R

   [WSGD, LOG_ERRORS, ERRORS_01] = sswg(X, Y, X_test, Y_test);  # SDG 
   
   distance = sqrt(sum((WNR .- WSGD).^2))  #odleglosc euklidesowa
   
   # zad4(X, Y, X_test, Y_test);  #zad 4 
  
end