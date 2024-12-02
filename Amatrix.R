##第一列id,第二列sire，第三列dam
##系谱需要从1开始排列，即系谱需要重新编码，每一行的id列等于其行号
##系谱重新编码详间Ready_for_pedigree
##不过是站在巨人的肩膀，感谢先辈们的研究！

Amatrix <- function(ped_path = NULL){
  pedigree=read.table(ped_path)
  
  ##sire列
  s = pedigree$V2
  ##dam列
  d = pedigree$V3
  
  A=matrix(NA, nrow=nrow(pedigree), ncol=nrow(pedigree))
  
  A[1,1] <- 1
  for( i in 2:nrow(pedigree)){
    
    ## Both are unknown
    if( s[i] == 0 && d[i] == 0 ){
      A[i,i] <- 1
      for( j in 1:(i-1))
        A[j,i] <- A[i,j] <- 0
    }
    
    ## Sire is unknown
    if( s[i] == 0 && d[i] != 0 ){
      A[i,i] <- 1
      for( j in 1:(i-1))
        A[j,i] <- A[i,j] <- 0.5*(A[j,d[i]])
    }
    
    ## Dam is unknown
    if( d[i] == 0 && s[i] != 0 ){
      A[i,i] <- 1
      for( j in 1:(i-1))
        A[j,i] <- A[i,j] <- 0.5*(A[j,s[i]])
    }
    
    ## Both are known
    if( d[i] != 0 && s[i] != 0 ){
      A[i,i] <- 1+0.5*(A[d[i],s[i]])
      for( j in 1:(i-1))
        A[j,i] <- A[i,j] <- 0.5*(A[j,s[i]]+A[j,d[i]])
    }
  }
  return(A)
}

#ped_path='C:/Users/Quan/Desktop/dmu_pedigree.txt'
#Ama=Amatrix(ped_path)


#根据系谱构建显性效应矩阵
#D <- matrix(NA,ncol=n,nrow=n)
#for(i in 1:n){
  #for(j in 1:n){
    #u1 <- ifelse(length(A[s[i],s[j]])>0,A[s[i],s[j]],0)
    #u2 <- ifelse(length(A[d[i],d[j]])>0,A[d[i],d[j]],0)
    #u3 <- ifelse(length(A[s[i],d[j]])>0,A[s[i],d[j]],0)
    #u4 <- ifelse(length(A[s[j],d[i]])>0,A[s[j],d[i]],0)
    #D[i,j] <- D[j,i] <- 0.25*(u1*u2+u3*u4)
  #}
#}
#diag(D)<-1

