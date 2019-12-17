clear
clc
M = csvRead("data.csv")
P=M(1:500,3:32)'; //Exemplos de entrada
T=M(1:500,2)';   // Valores desejados de saída (target)
Q=M(501:569,3:32)'; // Exemplos usados para teste
TQ=M(501:569,2)'; // Valores desejados na saída para os exemplos de teste
N=[30 12 1];
//Tipos de funções de ativação:
//ann_logsig_activ
//ann_purelin_activ
//ann_tansig_activ
//ann_hardlim_activ
//Define a função de ativação da camada intermediária e da saída
//Se tiverem 2 intermediárias, tem que definir uma função para cada uma
af = ['ann_logsig_activ','ann_logsig_activ'];  
lr = 0.08; //taxa de aprendizado
itermax = 1000; // número máximo de iterações
mse_min = 1e-5; //mínimo erro quadrático médio desejado
gd_min =  1e-8; // variação mínima do gradiente
//W = ann_FFBP_gd(P,T,N,af,lr,itermax,mse_min,gd_min)
//[y] = ann_FFBP_run(Q,W)
W = ann_FFBP_lm(P,T,N);
y = ann_FFBP_run(Q,W)
sleep(1000)
scf(1)
plot (y,'bo')
plot (TQ,'r*')
[lin col]=size(Q);
erro=(1/(col))*sum((y-TQ)^2)
