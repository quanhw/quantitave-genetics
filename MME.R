##the data we used is 10-21
##注意，本模型用的是单性状重复力模型，即只看的是总产仔数

p=0.67439102E-06
a=1.0666860
e=6.2842911  
kk=e/a
kkk=e/p
Ainv = as.matrix(read.table("C:/Users/Quan/Desktop/Qblup/Ainv.txt"))
addr=matrix(1,439,1)
X=cbind(addr,X)
##corssprod(x,y))==t(x)%*%x
##tcorssprod(x,y)==x%*%t(x)

FUN_MME <- function(X, Z1, Z2, y, R, ZI ) {
  Xt = t(X)
  Z1t = t(Z1)
  Z2t = t(Z2)
  
  XtX = Xt %*% X
  XtZ1 = Xt %*% Z1
  XtZ2 = Xt %*% Z2
  left1=cbind(XtX,  XtZ1,  XtZ2)
  rm(XtX,  XtZ1,  XtZ2)
  
  Z1tX = Z1t %*% X
  Z1tZ1 = Z1t %*% Z1 +Ainv * kk
  Z1tZ2 = Z1t %*% Z2
  left2=cbind(Z1tX, Z1tZ1, Z1tZ2)
  rm(Z1tX, Z1tZ1, Z1tZ2)
  
  Z2tX = Z2t %*% X
  Z2tZ1 = Z2t %*% Z1
  Z2tZ2 = Z2t %*% Z2 + diag(ZI) * kkk
  left3=cbind(Z2tX, Z2tZ1, Z2tZ2)
  rm(Z2tX, Z2tZ1, Z2tZ2)
 
  left=rbind(left1, left2, left3)
  rm(left1, left2, left3)
  right=rbind(Xt %*% y, Z1t %*% y, Z2t %*% y)
  
  left1=left + diag(nrow(left))*0.00000001
  
  solution=solve(left1,right)
  solution=solve(left,right)
  
  solution[1]
  
  so=data.frame(solution[24:5031])
  
  sum(so)
  
  
  
  
  ##竟然还要算残差，真难！
  ##加油！
  FUN_Residual <- function(y, X, beita, Z1, u1, Z2, u2){
    residual <- y- X %*% beita - Z1 %*% u1 - Z2 %*% u2
    return(residual)
  }
 
  residual <- FUN_Residual(y, 
               X,   solution[1:ncol(X)], 
               Z1,  solution[ncol(X)+1:ncol(X)+ncol(Z1)], 
               Z2,  solution[ncol(X)+ncol(Z1)+1:length(solution)])
  return(list(sol=solution, re=residual))
}

