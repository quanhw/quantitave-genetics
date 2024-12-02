##终于有个没有参考前人自己写的了
##为满足MME，我们需要将表型id顺序和系谱顺序相同，这样才能保证y=Xβ+Zu+e是相对应的

FUN_Xmatrix <- function(phe_path, pedfinal_path,
                        ID, first_fix_col, last_fix_col, 
                        first_ran_col, last_ran_col ,first_phe_col, last_phe_col) {
  
  dataX <- read.table(phe_path)
  ped <- read.table(pedfinal_path)
  
  ##这里注意重复力模型会有重复的育种值向量
  index <- match(dataX$V1, ped$V1)
  
  ##构建固定效应映射矩阵
  X_rol=nrow(dataX)
  X_fix_matirix=matrix(0,nrow=X_rol,ncol=0)
  for (i in first_fix_col:last_fix_col){
      X_beita=dataX[,i]
      X_beita=unique(sort(X_beita))
      X_col=length(X_beita)
      X_fix=matrix(0,nrow = X_rol, ncol = X_col)
      p=1
      for( j in dataX[,i]){
        X_fix[p,which(X_beita==j)]=1
        p=p+1
      }
      X_fix_matirix=cbind(X_fix_matirix,X_fix)
  }
  
  ##固定效应向量长度
  X_beita=ncol(X_fix_matirix)
  
  ##构建随机效应向量长度
  ##u_U=X_rol
  
  
  ##构建育种值随机效应映射矩阵
  u_Zmatrix1=matrix(0,nrow = X_rol ,ncol = nrow(ped))
  for(i in 1:X_rol){
    u_Zmatrix1[i,index[i]]=1
  }
  
  ##构建litter随机效应映射矩阵
  indexx <- match(dataX$V1,unique(dataX$V1))
  u_Zmatrix2=matrix(0,nrow = X_rol ,ncol = length(unique(dataX$V1)))
  for(i in 1:X_rol){
    u_Zmatrix2[i,indexx[i]]=1
  }
  
  ##构建残差效应向量长度
  ##e_e=X_rol
  
  ##构建残差效应映射矩阵
  e_Rmatrix=diag(X_rol)

  ##构建表型向量
  y_phe=as.matrix(dataX[,c(first_phe_col:last_phe_col)])
  
  return(list(X=X_fix_matirix, beita=X_beita, u=X_rol,  
              Z1=u_Zmatrix1, Z2=u_Zmatrix2, R=e_Rmatrix ,y=y_phe
              ,ZI=length(unique(dataX$V1))))
}


#ID=1
#first_fix_col=2
#last_fix_col=3
#first_ran_col=1
#last_ran_col=1
#first_phe_col=4
#last_phe_col=4





