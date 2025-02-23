#####################################################################
# 									
# Package: AGHmatrix 							
# 									
# File: Gmatrix.R
# Contains: Gmatrix slater_par check_Gmatrix_data			
# 									
# Written by Rodrigo Rampazo Amadeu 			
# Contributors: Marcio Resende Jr, Leticia AC Lara, Ivone Oliveira, Luis Felipe V Ferrao
# 									
# First version: Feb-2014 					
# Last update: 05-Aug-2021 						
# License: GPL-3	
# 									
#####################################################################

#' Construction of Relationship Matrix G
#'
#' Given a matrix (individual x markers), a method, a missing value, and a maf threshold, return a additive or non-additive relationship matrix. For diploids, the methods "Yang" and "VanRaden" for additive relationship matrices, and "Su" and "Vitezica" for non-additive relationship matrices are implemented. For autopolyploids, the method "VanRaden" for additive relationship, method "Slater" for full-autopolyploid model including non-additive effects, and pseudo-diploid parametrization are implemented. Weights are implemented for "VanRaden" method as described in Liu (2020). 
#' 
#' @param SNPmatrix matrix (n x m), where n is is individual names and m is marker names (coded inside the matrix as 0, 1, 2, ..., ploidy, and, missingValue). 
#' @param method "Yang" or "VanRaden" for marker-based additive relationship matrix. "Su" or "Vitezica" for marker-based dominance relationship matrix. "Slater" for full-autopolyploid model including non-additive effects. "Endelman" for autotetraploid dominant (digentic) relationship matrix. "MarkersMatrix" for a matrix with the amount of shared markers between individuals (3). Default is "VanRaden", for autopolyploids will be computed a scaled product (similar to Covarrubias-Pazaran, 2006).
#' @param missingValue missing value in data. Default=-9.
#' @param thresh.missing threshold on missing data, SNPs below of this frequency value will be maintained, if equal to 1, no threshold and imputation is considered. Default = 0.50.
#' @param maf minimum allele frequency accepted to each marker. Default=0.
#' @param verify.posdef verify if the resulting matrix is positive-definite. Default=FALSE.
#' @param ploidy data ploidy (an even number between 2 and 20). Default=2.
#' @param pseudo.diploid if TRUE, uses pseudodiploid parametrization of Slater (2016).
#' @param ratio if TRUE, molecular data are considered ratios and its computed the scaled product of the matrix (as in "VanRaden" method).
#' @param impute.method "mean" to impute the missing data by the mean per marker, "mode" to impute the missing data by the mode per marker, "global.mean" to impute the missing data by the mean across all markers, "global.mode" to impute the missing data my the mode across all marker. Default = "mean".
#' @param integer if FALSE, not check for integer numbers. Default=TRUE.
#' @param ratio.check if TRUE, run Mcheck with ratio data.
#' @param weights vector with weights for each marker. Only works if method="VanRaden". Default is a vector of 1's (equal weight).
#' @param ploidy.correction It sets the denominator (correction) of the crossprod. Used only when ploidy > 2 for "VanRaden" and ratio models. If TRUE, it uses the sum of "Ploidy" times "Frequency" times "(1-Frequency)" of each marker as method 1 in VanRaden 2008 and Endelman (2018). When ratio=TRUE, it uses "1/Ploidy" times "Frequency" times "(1-Frequency)". If FALSE, it uses the sum of the sampling variance of each marker. Default = FALSE. 
#' @param rmv.mono if monomorphic markers should be removed. Default=FALSE.
#' @param thresh.htzy threshold heterozigosity, remove SNPs below this threshold. Default=0.
#' @param ASV if TRUE, transform matrix into average semivariance (ASV) equivalent (K = K / (trace(K) / (nrow(K)-1))). Details formula 2 of Fieldmann et al. (2022). Default = FALSE.
#' @return Matrix with the marker-bases relationships between the individuals
#'
#' @examples
#' \dontrun{
#' ## Diploid Example
#' data(snp.pine)
#' #Verifying if data is coded as 0,1,2 and missing value.
#' str(snp.pine)
#' #Build G matrices
#' Gmatrix.Yang <- Gmatrix(snp.pine, method="Yang", missingValue=-9, maf=0.05)
#' Gmatrix.VanRaden <- Gmatrix(snp.pine, method="VanRaden", missingValue=-9, maf=0.05)
#' Gmatrix.Su <- Gmatrix(snp.pine, method="Su", missingValue=-9, maf=0.05)
#' Gmatrix.Vitezica <- Gmatrix(snp.pine, method="Vitezica", missingValue=-9, maf=0.05)
#' 
#' ## Autetraploid example
#' data(snp.sol)
#' #Build G matrices
#' Gmatrix.VanRaden <- Gmatrix(snp.sol, method="VanRaden", ploidy=4)
#' Gmatrix.Endelman <- Gmatrix(snp.sol, method="Endelman", ploidy=4) 
#' Gmatrix.Slater <- Gmatrix(snp.sol, method="Slater", ploidy=4)
#' Gmatrix.Pseudodiploid <- Gmatrix(snp.sol, method="VanRaden", ploidy=4, pseudo.diploid=TRUE) 
#' 
#' #Build G matrix with weights
#' Gmatrix.weighted <- Gmatrix(snp.sol, method="VanRaden", weights = runif(3895,0.001,0.1), ploidy=4)
#' }
#' 
#' @author Rodrigo R Amadeu \email{rramadeu@@gmail.com}, Marcio Resende Jr, Letícia AC Lara, Ivone Oliveira, and Felipe V Ferrao
#' 
#' @references \emph{Covarrubias-Pazaran, G. 2016. Genome assisted prediction of quantitative traits using the R package sommer. PLoS ONE 11(6):1-15.}
#' @references \emph{Endelman, JB, et al., 2018. Genetic variance partitioning and genome-wide prediction with allele dosage information in autotetraploid potato. Genetics, 209(1) pp. 77-87.}
#' @references \emph{Feldmann MJ, et al. 2022. Average semivariance directly yields accurate estimates of the genomic variance in complex trait analyses. G3 (Bethesda), 12(6).}
#' @references \emph{Liu, A, et al. 2020. Weighted single-step genomic best linear unbiased prediction integrating variants selected from sequencing data by association and bioinformatics analyses. Genet Sel Evol 52, 48.}
#' @references \emph{Slater, AT, et al. 2016. Improving genetic gain with genomic selection in autotetraploid potato. The Plant Genome 9(3), pp.1-15.}
#' @references \emph{Su, G, et al. 2012. Estimating additive and non-additive genetic variances and predicting genetic merits using genome-wide dense single nucleotide polymorphism markers. PloS one, 7(9), p.e45293.}
#' @references \emph{VanRaden, PM, 2008. Efficient methods to compute genomic predictions. Journal of dairy science, 91(11), pp.4414-4423.}
#' @references \emph{Vitezica, ZG, et al. 2013. On the additive and dominant variance and covariance of individuals within the genomic selection scope. Genetics, 195(4), pp.1223-1230.}
#' @references \emph{Yang, J, et al. 2010. Common SNPs explain a large proportion of the heritability for human height. Nature genetics, 42(7), pp.565-569.}
#' 
#' @export

Gmatrix <- function (SNPmatrix = NULL, method = "VanRaden", 
                     missingValue = -9, maf = 0, thresh.missing = .50,
                     verify.posdef = FALSE, ploidy=2,
                     pseudo.diploid = FALSE, integer=TRUE,
                     ratio = FALSE, impute.method = "mean", rmv.mono=FALSE, thresh.htzy=0,
                     ratio.check = TRUE, weights = NULL, ploidy.correction = FALSE, ASV=FALSE){
  Time = proc.time()
  markers = colnames(SNPmatrix)
  
  if(!is.null(weights))
    if(length(weights)!=ncol(SNPmatrix))
      stop(deparse("weight should be a numeric vector of the same number of markers in the SNPmatrix"))

  if(ratio){ #This allows to enter in the scaled crossprod condition
    method="VanRaden"
  }
  
  if (!is.na(missingValue)) {
    m <- match(SNPmatrix, missingValue, 0)
    SNPmatrix[m > 0] <- NA
  }
  
  check_Gmatrix_data(SN0
  NumberMarkers <- ncol(SNPmatrix)
  nindTotal <- colSums(!is.na(SNPmatrix))
  nindAbs <- max(nindTotal)
  cat("Initial data: \n")
  cat("\tNumber of Individuals:", max(nindTotal), "\n")
  cat("\tNumber of Markers:", NumberMarkers, "\n")
  
  if(ratio==FALSE){
    SNPmatrix <- Mcheck(SNPmatrix,
                        ploidy = ploidy, 
                        thresh.maf = maf, 
                        rmv.mono = rmv.mono,
                        thresh.htzy = thresh.htzy,
                        thresh.missing = thresh.missing,
                        impute.method = impute.method)
  }
  
  ## Testing ratio check function: not final!
  if(ratio && ratio.check){
    SNPmatrix <- Mcheck(SNPmatrix,
                        ploidy = ploidy, 
                        thresh.maf = maf, 
                        rmv.mono = rmv.mono,
                        thresh.missing = thresh.missing,
                        impute.method = impute.method)
  }
  
  if(method=="Slater"){
    P <- colSums(SNPmatrix,na.rm = TRUE)/nrow(SNPmatrix)
    SNPmatrix[,which(P>ploidy/2)] <- ploidy-SNPmatrix[,which(P>(ploidy/2))]
    SNPmatrix <- slater_par(SNPmatrix,ploidy=ploidy)
    NumberMarkers <- ncol(SNPmatrix)
    Frequency <- colSums(SNPmatrix,na.rm=TRUE)/nrow(SNPmatrix)
    FreqP <- matrix(rep(Frequency, each = nrow(SNPmatrix)), 
                    ncol = ncol(SNPmatrix))
  }
  
  if(ploidy==2){
    alelleFreq <- function(x, y) {
      (2 * length(which(x == y)) + length(which(x == 1)))/(2 * 
                                                             length(which(!is.na(x))))
    }
    Frequency <- cbind(apply(SNPmatrix, 2, function(x) alelleFreq(x,0))
                       , apply(SNPmatrix, 2, function(x) alelleFreq(x, 2)))
    
#   if (any(Frequency[, 1] <= maf) & maf != 0) {
#      cat("\t", length(which(Frequency[, 1] <= maf)), "markers dropped due to maf cutoff of", maf, "\n")
#      SNPmatrix <- SNPmatrix[,-which(Frequency[, 1] <= maf)]
#      cat("\t", ncol(SNPmatrix), "markers kept \n")
#      Frequency <- as.matrix(Frequency[-which(Frequency[,1] <= 
#                                                maf), ])
#      NumberMarkers <- ncol(SNPmatrix)
#    }
    FreqP <- matrix(rep(Frequency[, 2], each = nrow(SNPmatrix)), 
                    ncol = ncol(SNPmatrix))
  }
  
  if(ploidy>2 && pseudo.diploid){## Uses Pseudodiploid model
    P <- colSums(SNPmatrix,na.rm = TRUE)/nrow(SNPmatrix)
    SNPmatrix[,which(P>ploidy/2)] <- ploidy-SNPmatrix[,which(P>(ploidy/2))]
    Frequency <- colSums(SNPmatrix,na.rm=TRUE)/(ploidy*nrow(SNPmatrix))
    Frequency <- cbind(1-Frequency,Frequency)
    FreqP <- matrix(rep(Frequency[, 2], each = nrow(SNPmatrix)), 
                    ncol = ncol(SNPmatrix))
    SNPmatrix[SNPmatrix %in% c(1:(ploidy-1))] <- 1
    SNPmatrix[SNPmatrix==ploidy] <- 2
  }
  
  if (method == "MarkersMatrix") {
    Gmatrix <- !is.na(SNPmatrix)
    Gmatrix <- tcrossprod(Gmatrix, Gmatrix)
    return(Gmatrix)
  }
  
  ## VanRaden ##
  if (method == "VanRaden") {
    if(is.null(weights)){
      if(ploidy==2 & ratio==FALSE){
        TwoPQ <- 2 * t(Frequency[, 1]) %*% Frequency[, 2]
        SNPmatrix <- SNPmatrix- 2 * FreqP
        SNPmatrix[is.na(SNPmatrix)] <- 0
        Gmatrix <- (tcrossprod(SNPmatrix, SNPmatrix))/as.numeric(TwoPQ)
      }else{
        if(ploidy.correction){
          if(ratio==FALSE){
            Frequency <- apply(X=SNPmatrix,FUN=mean,MARGIN=2,na.rm=TRUE)/ploidy
            K <- sum(ploidy * Frequency * (1-Frequency))
          }else{
            Frequency <- apply(X=SNPmatrix,FUN=mean,MARGIN=2,na.rm=TRUE)
            K <- sum(1/ploidy * Frequency * (1-Frequency))
          }
        }
        
        SNPmatrix<-scale(SNPmatrix,center=TRUE,scale=FALSE)
        if(!ploidy.correction){
          K <- sum(apply(X=SNPmatrix,FUN=var,MARGIN=2,na.rm=TRUE))
        }
        SNPmatrix[which(is.na(SNPmatrix))] <- 0
        Gmatrix<-tcrossprod(SNPmatrix)/K
      }
    }else{
      weights = weights[match(colnames(SNPmatrix),markers)]
      if(ploidy==2 & ratio==FALSE){
        TwoPQ <- 2 * t(Frequency[, 1]) %*% Frequency[, 2]
        SNPmatrix <- SNPmatrix- 2 * FreqP
        SNPmatrix[is.na(SNPmatrix)] <- 0
        Gmatrix <- tcrossprod(tcrossprod(SNPmatrix, diag(weights)), SNPmatrix)/as.numeric(TwoPQ)
      }else{
        if(ploidy.correction){
          if(ratio==FALSE){
            Frequency <- apply(X=SNPmatrix,FUN=mean,MARGIN=2,na.rm=TRUE)/ploidy
            K <- sum(ploidy * Frequency * (1-Frequency))
          }else{
            Frequency <- apply(X=SNPmatrix,FUN=mean,MARGIN=2,na.rm=TRUE)
            K <- sum(Frequency * (1-Frequency))
          }
        }
        SNPmatrix<-scale(SNPmatrix,center=TRUE,scale=FALSE)
        if(!ploidy.correction){
          K <- sum(apply(X=SNPmatrix,FUN=var,MARGIN=2,na.rm=TRUE))
        }
        SNPmatrix[which(is.na(SNPmatrix))] <- 0
        Gmatrix<-tcrossprod(tcrossprod(SNPmatrix, diag(weights)), SNPmatrix)/K
      }
    }
  }
  
  if (method == "Yang") {
    FreqPQ <- matrix(rep(2 * Frequency[, 1] * Frequency[, 
                                                        2], each = nrow(SNPmatrix)), ncol = ncol(SNPmatrix))
    G.all <- (SNPmatrix^2 - (1 + 2 * FreqP) * SNPmatrix + 
                2 * (FreqP^2))/FreqPQ
    G.ii <- as.matrix(colSums(t(G.all), na.rm = T))
    SNPmatrix <- (SNPmatrix - (2 * FreqP))/sqrt(FreqPQ)
    G.ii.hat <- 1 + (G.ii)/NumberMarkers
    SNPmatrix[is.na(SNPmatrix)] <- 0
    Gmatrix <- (tcrossprod(SNPmatrix, SNPmatrix))/NumberMarkers
    diag(Gmatrix) <- G.ii.hat
  }
  
  if (method == "Su"){
    TwoPQ <- 2*(FreqP)*(1-FreqP)
    SNPmatrix[SNPmatrix==2 | SNPmatrix==0] <- 0
    SNPmatrix <- SNPmatrix - TwoPQ
    SNPmatrix[is.na(SNPmatrix)] <- 0
    Gmatrix <- tcrossprod(SNPmatrix,SNPmatrix)/
      sum(TwoPQ[1,]*(1-TwoPQ[1,]))        
  }
  
  if (method == "Vitezica"){
    TwoPQ <- 2*(FreqP[1,])*(1-FreqP[1,])
    SNPmatrix[is.na(SNPmatrix)] <- 0
    SNPmatrix <- (SNPmatrix==0)*-2*(FreqP^2) +
      (SNPmatrix==1)*2*(FreqP)*(1-FreqP) +
      (SNPmatrix==2)*-2*((1-FreqP)^2)
    Gmatrix <- tcrossprod(SNPmatrix,SNPmatrix)/sum(TwoPQ^2)
  }
  
  if (method == "Slater"){
    drop.alleles <- which(Frequency==0)
    if(length(drop.alleles)>0){
      Frequency <- Frequency[-drop.alleles]
      SNPmatrix <- SNPmatrix[,-drop.alleles]
      FreqP <- FreqP[,-drop.alleles]
    }
    FreqPQ <- matrix(rep(Frequency * (1-Frequency),
                         each = nrow(SNPmatrix)),
                     ncol = ncol(SNPmatrix))
    SNPmatrix[which(is.na(SNPmatrix))] <- 0
    G.ii <- (SNPmatrix^2 - (2 * FreqP) * SNPmatrix + FreqP^2)/FreqPQ
    G.ii <- as.matrix(colSums(t(G.ii), na.rm = T))
    G.ii <- 1 + (G.ii)/NumberMarkers
    SNPmatrix <- (SNPmatrix - (FreqP))/sqrt(FreqPQ)
    SNPmatrix[is.na(SNPmatrix)] <- 0
    Gmatrix <- (tcrossprod(SNPmatrix, SNPmatrix))/NumberMarkers
    diag(Gmatrix) <- G.ii
  }
  
  if( method == "Endelman" ){
    if( ploidy != 4 ){
      cat( stop( "'Endelman' method is just implemented for ploidy=4" ))
    }
    Frequency <- colSums(SNPmatrix)/(nrow(SNPmatrix)*ploidy)
    Frequency <- cbind(Frequency,1-Frequency)
    SixPQ <- 6 * t((Frequency[, 1]^2)) %*% (Frequency[, 2]^2)
    SNPmatrix <- 6 * t((Frequency[, 1]^2)%*%t(rep(1,nrow(SNPmatrix)))) - 
      3*t((Frequency[, 1])%*%t(rep(1,nrow(SNPmatrix))))*SNPmatrix + 0.5 * SNPmatrix*(SNPmatrix-1)
    Gmatrix <- (tcrossprod(SNPmatrix, SNPmatrix))/as.numeric(SixPQ)
  }
  
  if (verify.posdef) {
    e.values <- eigen(Gmatrix, symmetric = TRUE)$values
    indicator <- length(which(e.values <= 0))
    if (indicator > 0) 
      cat("\t Matrix is NOT positive definite. It has ", indicator, 
          " eigenvalues <= 0 \n \n")
  }
  
  if(ASV){
    Gmatrix = get_ASV(Gmatrix)
  }
  
  Time = as.matrix(proc.time() - Time)
  cat("Completed! Time =", Time[3], " seconds \n")
  gc()
  return(Gmatrix)
}

## Internal Functions ##
get_ASV = function(x){
  return( x / ( sum(diag(x)) / (nrow(x) - 1)) )
}

# Coding SNPmatrix as Slater (2016) Full autotetraploid model including non-additive effects (Presence/Absence per Genotype per Marker)
slater_par <- function(X,ploidy){
  prime.index <- c(3,5,7,11,13,17,19,23,29,31,37,
                   41,43,47,53,59,61,67,71,73,79)
  
  NumberMarkers <- ncol(X)
  nindTotal <- nrow(X)
  X <- X+1
  
  ## Breaking intervals to use less RAM
  temp <- seq(1,NumberMarkers,10000)
  temp <- cbind(temp,temp+9999)
  temp[length(temp)] <- NumberMarkers
  prime.index <- prime.index[1:(ploidy+1)]
  
  ## Uses Diagonal (which is Sparse mode, uses less memmory)
  for(i in 1:nrow(temp)){
    X.temp <- X[,c(temp[i,1]:temp[i,2])]
    NumberMarkers <- ncol(X.temp)
    X.temp <- X.temp %*% t(kronecker(diag(NumberMarkers),prime.index))
    X.temp[which(as.vector(X.temp) %in%
                   c(prime.index*c(1:(ploidy+1))))] <- 1
    X.temp[X.temp!=1] <- 0
    if(i==1){
      X_out <- X.temp
    }else{
      X_out <- cbind(X_out,X.temp)
    }   
  }
  gc()
  return(X_out)
}

# Internal function to check input Gmatrix arguments
check_Gmatrix_data <- function(SNPmatrix,ploidy,method, ratio=FALSE, integer=TRUE){
  if (is.null(SNPmatrix)) {
    stop(deparse("Please define the variable SNPdata"))
  }
  if (all(method != c("Yang", "VanRaden", "Slater", "Su", "Vitezica", "MarkersMatrix","Endelman"))) {
    stop("Method to build Gmatrix has to be either `Yang` or `VanRaden` for marker-based additive relationship matrix, or `Su` or `Vitezica` or `Endelman` for marker-based dominance relationship matrx, or `MarkersMatrix` for matrix with amount of shared-marks by individuals pairs")
  }
  
#  if( method=="Yang" && ploidy>2)
#    stop("Change method to 'VanRaden' for ploidies higher than 2 for marker-based additive relationship matrix")
  
  if( method=="Su" && ploidy>2)
    stop("Change method to 'Slater' for ploidies higher than 2 for marker-based non-additive relationship matrix")

  if( method=="Vitezica" && ploidy>2)
    stop("Change method to 'Slater' for ploidies higher than 2 for marker-based non-additive relationship matrix")
  
  if(!is.matrix(SNPmatrix)){
    cat("SNPmatrix class is:",class(SNPmatrix),"\n")
    stop("SNPmatrix class must be matrix. Please verify it.")
  }
  
  if(!ratio){
  if( ploidy > 20 | (ploidy %% 2) != 0)
    stop(deparse("Only even ploidy from 2 to 20"))
  
  t <- max(SNPmatrix,na.rm = TRUE)
  if( t > ploidy )
    stop(deparse("Check your data, it has values above ploidy number"))
  
  t <- min(SNPmatrix,na.rm=TRUE)
  if( t < 0 )
    stop(deparse("Check your data, it has values under 0"))
  
  if(integer)
    if(prod(SNPmatrix == round(SNPmatrix),na.rm = TRUE)==0)
      stop(deparse("Check your data, it has not integer values"))
    }

  if(ratio){
    t <- max(SNPmatrix,na.rm = TRUE)
    if( t > 1)
      stop(deparse("Check your data, it has values above 1. It is expected a ratio values [0;1]."))
    
    t <- min(SNPmatrix,na.rm=TRUE)
    if( t < 0 )
      stop(deparse("Check your data, it has values under 0. It is expected a ratio values [0;1]."))
  }
}