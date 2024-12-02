code_path='C:/Users/Quan/Desktop/Qblup/'
phe_path='C:/Users/Quan/Desktop/dmu_sire.txt'
ped_path='C:/Users/Quan/Desktop/dmu_pedigree.txt'
pedfinal_path='C:/Users/Quan/Desktop/dmu_pedigree1.txt'

##准备各个矩阵
source(paste0(code_path, 'matrix_vector.R'), encoding='UTF-8')
use_vector_matrix=FUN_Xmatrix(phe_path,pedfinal_path, ID=1, first_fix_col=2, last_fix_col=3, 
                      first_ran_col=1, last_ran_col=1, first_phe_col=4, last_phe_col=4)
X=use_vector_matrix[[1]]
Z1=use_vector_matrix[[4]]
Z2=use_vector_matrix[[5]]
R=use_vector_matrix[[6]]
y=use_vector_matrix[[7]]
ZI=use_vector_matrix[[8]]


##计算A阵以及A逆
source(paste0(code_path, 'Amatrix.R'), encoding='UTF-8')
#A <- Amatrix(ped_path)
source(paste0(code_path, 'Ainvmatrix.R'), encoding='UTF-8')
Ainv <- Ainvmatrix(ped_path)

##进行MME
source(paste0(code_path, 'MME.R'), encoding='UTF-8')


