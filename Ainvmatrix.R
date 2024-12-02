Ainvmatrix <- function(ped_path = NULL){
  pedigree=read.table(ped_path)
  ##sire列
  s = pedigree$V2
  ##dam列
  d = pedigree$V3
  
  Ainv=matrix(0, nrow=nrow(pedigree), ncol=nrow(pedigree))
  
  ##读取A阵对角线
  f=diag(Amatrix(ped_path))-1
  
  Ainv[1,1] <- 1
  
  for( i in 2:nrow(pedigree)){
    ## Both are unknown
    
    if( s[i] == 0 && d[i] == 0 ){
      Ainv[i,i] <- Ainv[i,i]+1
    }
    
    ## Sire is unknown
    if( s[i] == 0 && d[i] != 0 ){
      dd=1/(0.75-0.25*f[d[i]])
      Ainv[i,i] <- Ainv[i,i]+dd
      Ainv[i,d[i]] <- Ainv[d[i],i] <- Ainv[d[i],i]-0.5*dd
      Ainv[d[i],d[i]] <- Ainv[d[i],d[i]]+0.25*dd
    }

    ## Dam is unknown
    if( d[i] == 0 && s[i] != 0 ){
      dd=1/(0.75-0.25*f[s[i]])
      Ainv[i,i] <- Ainv[i,i]+dd
      Ainv[i,s[i]] <- Ainv[s[i],i] <- Ainv[s[i],i]-0.5*dd
      Ainv[s[i],s[i]] <- Ainv[s[i],s[i]]+0.25*dd
    }

    ## Both are known
    if( d[i] != 0 && s[i] != 0 ){
      dd=1/(0.5-0.25*(f[s[i]]+f[d[i]]))
      Ainv[i,i] <- Ainv[i,i]+dd
      Ainv[i,s[i]] <- Ainv[s[i],i] <- Ainv[s[i],i]-0.5*dd
      Ainv[d[i],i] <- Ainv[i,d[i]]  <- Ainv[i,d[i]]-0.5*dd
      Ainv[s[i],s[i]] <- Ainv[s[i],s[i]]+0.25*dd
      Ainv[d[i],d[i]] <- Ainv[d[i],d[i]]+0.25*dd
      Ainv[s[i],d[i]] <-Ainv[d[i],s[i]] <-Ainv[s[i],d[i]]+0.25*dd
    }
  }
  return(Ainv)
}


