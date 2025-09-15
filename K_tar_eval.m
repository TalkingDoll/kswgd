function K = K_tar_eval(x_tar, x_eval, p_tar, sq_tar, D, epsilon)

    sq_x_eval = sum(x_eval.^2,2);
   
    
    

    K_cross = exp(-(sq_x_eval' + sq_tar - 2 * x_tar * x_eval') ./ (2*epsilon));
%     K_cross = exp(-(x_tar - x_eval').^2/ (2*epsilon));
    p_eval = sum(K_cross);
   
    
    M = K_cross ./ sqrt(p_eval) ./ sqrt(p_tar);
    
    
    K1 = M ./ D;
    
    K2 = M ./sum(M);
    
    K = 1/2 * (K2 + K1);
   
    
    
    

end