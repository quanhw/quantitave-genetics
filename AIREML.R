##AIREML
##参考宁超师兄公众号数量遗传学与Python
##我见青山多妩媚，料青山见我应如是。

FUN_AIReml <- function(y, X, Z, kin, theta=c(1.0, 1.0), max_iter=30, cc=1.0e-8) {
  source(paste0(code_path, 'matrix_vector.R'), encoding='UTF-8')
  use_vector_matrix=FUN_Xmatrix(phe_path,pedfinal_path, ID=1, first_fix_col=2, last_fix_col=3, 
                                first_ran_col=1, last_ran_col=1, first_phe_col=4, last_phe_col=4)
  X=use_vector_matrix[[1]]
  Z1=use_vector_matrix[[4]]
  Z2=use_vector_matrix[[5]]
  R=use_vector_matrix[[6]]
  y=use_vector_matrix[[7]]
  ZI=use_vector_matrix[[8]]
  
  if(delta <= cc){
    source(paste0(code_path, 'MME.R'), encoding='UTF-8')
    FUN_MME(X, Z1, Z2, y, R, ZI )
    
    
    theta = c(1.0, 1.0)
    
    V = Z %*% A %*% Zt * theta[1] + diag(nrow(X)) * theta[2]
    Vinv = solve(V)
    VVinv = solve(Xt %*% Vinv %*% X)
    P  = Vinv - Vinv %*% X %*% VVinv %*% Xt %*% Vinv
    Pt = t(P)
    Vs = Z %*% A %*% Zt
    AI11 = Pt %*% Vs %*% P %*% Vs %*% P %*% y
    AI12 = t(y) %*% P %*% Vs %*% P %*% P %*% y
    AI22 = t(y) %*% P %*% P %*% P %*% y
    AI = cbind(rbind(AI11,AI12), rbind(AI12,AI22))
    AI = 0.5 * AI
    rm(AI11, AI12, AI22)
    
    #theta11 = -0.5tr(P %*% Z %*% A %*% Zt) +0.5* t(y) %*% P %*% Z %*% A %*% Zt %*% P %*% y
    theta11 = -0.5(q/starta
                   -lava::tr(solve(A) %*% C**aa)/(starta**2)
                   -t(e)*Z*μ/(starte*starta))
    #theta11 = -0.5tr(P) +0.5* t(y) %*% P %*% P %*% y
    theta22 = -0.5((N-qr(X)$rank)/starte
                   -(q-(lava::tr(solve(A) %*% C**aa)/starta**2))/starte
                   -t(e)*e/(starte**2))
    delta = AI %*% c(theta11, theta22)
    theta = theta + delta
  }
  else{
    return(MME结果、thta)
  }
  
}
  