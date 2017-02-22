function [X, W] = train_reg_log(X, Y, l = 10^(-5))  
  n = size(X, 1);                   #wiesze (obserwacje)
  X = [ones(n, 1), X];              #dodanie x0
  m = size(X, 2);                   #kolumny (cechy)
  
  W = zeros(m, 1);                  #poczatkowy wektor W        
  
  it = 0;
  do
    B = diag(1./(1+e.^(Y.*(X*W))));
    H = X'*B*(eye(n)-B)*X;
    if (cond(H) == Inf)
      H = l*eye(m) + H;
      grad = l*W - X'*B*Y;
    else
      grad = (-1).*(X'*B*Y);
    endif
    
    W = W - H^(-1)*grad; 

    #B = diag(1./(1+e.^(Y.*(X*W))));   #dla nastepnej iteracji
    #grad = l*W - X'*B*Y;              #dla nastepnej iteracji
    it += 1;
  until (norm(grad) <= 10^(-8))
    it
end