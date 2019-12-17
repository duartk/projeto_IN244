clear
clc
M = csvRead('winequality-white_normalizado.csv')
P=M(2:700,2:12)'; //DADOS DE TREINAMENTO
T=M(2:700,13)'; //SAIDAS DESEJADAS
Q=M(701:1000,2:12)'; // Exemplos usados para teste
TQ=M(701:1000,13)'; // Valores desejados na saída para os exemplos de teste
N=[11 8 4 1];
//Tipos de funções de ativação:
//ann_logsig_activ
//ann_purelin_activ
//ann_tansig_activ
//ann_hardlim_activ
//Define a função de ativação da camada intermediária e da saída
//Se tiverem 2 intermediárias, tem que definir uma função para cada uma
af = ['ann_tansig_activ','ann_tansig_activ','ann_purelin_activ'];  
lr = 0.03; //taxa de aprendizado
itermax = 1000; // número máximo de iterações
mse_min = 1e-5; //mínimo erro quadrático médio desejado
gd_min =  1e-5; // variação mínima do gradiente
W = ann_FFBP_gd(P,T,N,af,lr,itermax,mse_min,gd_min)
[y] = ann_FFBP_run(Q,W, af)
disp(y);
xpause(30000)
scf(1)
plot (y)
plot (TQ,'r-')
