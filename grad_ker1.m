function g = grad_ker1(x_eval, x_tar, p_tar, sq_tar, D2, epsilon)
    dim = size(x_eval,2);
    

   
    sq_x_eval = sum(x_eval.^2,2);
   
    
    
    p_tar = p_tar';
    K_cross = exp(-(sq_x_eval + sq_tar' - 2 * x_eval * x_tar') ./ (2*epsilon));
    p_eval = sum(K_cross,2);
    
    
    M = K_cross ./ sqrt(p_eval) ./ sqrt(p_tar);

    D = sum(M,2);
    g = zeros(size(M,1), size(M,2) ,dim);
    for i = 1:dim
        d_K_cross = (-(x_eval(:,i) - x_tar(:,i)')./epsilon) .* K_cross;
        d_sqrt_p_eval = 1 ./(2 * sqrt(p_eval)) .* sum(d_K_cross,2);
    
        dM = (d_K_cross .* sqrt(p_eval)  - K_cross .* d_sqrt_p_eval)./sqrt(p_tar) ./p_eval;
        dD = sum(dM,2);
    
        g1 = (dM .* D - dD .* M) ./ (D.^2) / epsilon;
    
    
        g2 = dM ./ D2' ./epsilon;
    
        g(:,:,i) = 1/2 * (g1 + g2);
    end

end