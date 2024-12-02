#系谱是三列系谱嗷，且分别是id, sire, dam
##Let us do something boring but interesting

library(tidyverse)
ped_path='C:/Users/Quan/Desktop/BLUP/DBTL/20240923/DBTL系谱数据.XLSX'

Ready_for_pedigree <- function(ped_path = NULL
){
  ##读取系谱文件，并把系谱提取成3列形式
  ped = readxl::read_xlsx(ped_path)
  columns_need <- list(
    c("个体号", "父亲",   "母亲"),
    c("父亲",   "父父",   "父母"),
    c("父父",   "父父父", "父父母"),
    c("父母",   "父母父", "父母母"),
    c("母亲",   "母父",   "母母"),
    c("母父",   "母父父", "母父母"),
    c("母母",   "母母父", "母母母")
  )
  
  ped <- columns_need %>%
    lapply(function(cols) {
      ped %>%
        select(all_of(cols)) %>%
        set_names(c('id', 'sire', 'dam'))
    }) %>%
    bind_rows()
  
  ped <- ped %>% 
    filter(!is.na(id)) %>% 
    distinct() %>%
    mutate(breed = substr(id, 1, 2))
  
  ##只要大白品种
  ped=subset(ped,ped$breed=='YY')
  ped=ped[,c(1:3)]
  
  ##重排序
  id.code <- 
    data.frame(id = sort(unique(c(ped$id, 
                                  ped$sire, 
                                  ped$dam)))) %>% 
    mutate(num = row_number())
  
  ped[is.na(ped)] <- '0'
  id.code=rbind(id.code,c(0,0))
  
  ped <- ped %>% 
    inner_join(id.code %>% rename(id_num   = num), join_by(id   == id)) %>% 
    inner_join(id.code %>% rename(sire_num = num), join_by(sire == id)) %>% 
    inner_join(id.code %>% rename(dam_num  = num), join_by(dam  == id)) %>% 
    select(id_num ,sire_num ,dam_num)
  
  ##将一些不在个体记录中的父母找出来，并把他们加到原来的系谱中
  id.code1=id.code[! id.code$num %in% ped$id_num,]
  id.code1$sire_num <-id.code1$dam_num <-0
  id.code1=id.code1[,c(2:4)]
  names(id.code1)=c('id_num','sire_num', 'dam_num')
  ped=rbind(ped, id.code1)
  rm(id.code,id.code1)
  ped=ped[ped$id_num !=0,]
  names(ped)=c('ID','SIRE','DAM')

  OldcountGen <- function(ped)
  {
    findParents <- function(i)
    {
      id <- ped$ID[i]
      s <- match(ped$SIRE[i],ped$ID)
      if(!is.na(s))
      {
        if(is.na(gen[s]))findParents(s)
        genS <- gen[s]+1
      }
      else genS <- 0
      
      d <- match(ped$DAM[i],ped$ID)
      if(!is.na(d))
      {
        if(is.na(gen[d]))findParents(d)
        genD <- gen[d]+1
      }
      else genD <- 0
      
      gen[i] <<- max(genD,genS)
    }
    gen <- rep(NA,nrow(ped))
    for(i in 1:nrow(ped))findParents(i)
    return(gen)
  }
  ped$generate=OldcountGen(ped)
  ped <- ped[order(ped$generate), ]
  return(ped)
}




