function [X, W] = train_reg_log(X, Y, forceLambda, l = 10^(-5))  
  #Wyznaczenie wektora wag dla metody N-R
  
  n = size(X, 1);                   #wiesze (obserwacje)
  X = [ones(n, 1), X];              #dodanie x0
  m = size(X, 2);                   #kolumny (cechy)
  
  W = zeros(m, 1);                  #poczatkowy wektor W
  it = 0;
  do
    B = diag(1./(1+e.^(Y.*(X*W))));
    H = X'*B*(eye(n)-B)*X;
    if (forceLambda == true || cond(H) == Inf)
      H = l*eye(m) + H;
      grad = l*W - X'*B*Y;
    else
      grad = (-1).*(X'*B*Y);
    endif
    
    W = W - H^(-1)*grad; 
    it += 1;
  until (norm(grad) <= 10^(-8))
    it
end