function [W, LOG_ERRORS, ERRORS_01] = sswg_train(X, Y, ages)
  
  
  n = size(X, 1);      #wiersze
  X = [ones(n,1), X];
  
  rp = randperm(n);
  
  Xp = X(rp, :);
  Yp = Y(rp);
  LOG_ERRORS = [];
  ERRORS_01 = [];
  m = size(Xp, 2);    #kolumny
  W = zeros(m, 1);
  W_prev = W;           #wektor W jedna epoke wstecz
  c = 0.25;            #krok
  for k=1:ages
    for i = 1:n
      step = c/sqrt(i + (k-1) * n);
      grad = step * (Yp(i) / (1 + exp(Yp(i)*Xp(i,:)*W))) * Xp(i,:);
      W = W + grad';
    endfor
    Y_pred = Xp * W;                                  # predycja y
    ERR = mean(log(1 + e.^((-1) .* Yp .* Y_pred)))    #sredni blad logistyczny
    LOG_ERRORS = [LOG_ERRORS, ERR];
    
    ERR01 = mean((Y_pred .* Yp) < 0);
    ERRORS_01 = [ERRORS_01, ERR01];
  endfor  
end