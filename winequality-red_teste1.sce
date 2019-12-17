clear all
clc
M = csvRead('winequality-red_normalizado.csv')
P=M(2:1120,2:12)'; //Exemplos de entrada
T=M(2:1120,13)';   // Valores desejados de saída (target)
Q=M(1121:1600,2:12)'; // Exemplos usados para teste
TQ=M(1121:1600,13)'; // Valores desejados na saída para os exemplos de teste
N=[11 9 6 1];
//Tipos de funções de ativação:
//ann_logsig_activ
//ann_purelin_activ
//ann_tansig_activ
//ann_hardlim_activ
//Define a função de ativação da camada intermediária e da saída
//Se tiverem 2 intermediárias, tem que definir uma função para cada uma
af = ['ann_tansig_activ','ann_tansig_activ','ann_tansig_activ'];  
W = ann_ffbp_init(N)
//lr = 0.08; //taxa de aprendizado
//itermax = 1000; // número máximo de iterações
//mse_min = 1e-5; //mínimo erro quadrático médio desejado
//gd_min =  1e-5; // variação mínima do gradiente
//W = ann_FFBP_gd(P,T,N,af,lr,itermax,mse_min,gd_min)
//[y] = ann_FFBP_run(Q,W)
W = ann_FFBP_lm(P,T,N,af);
s = ann_FFBP_run(Q,W,af)
y = ann_hardlim_activ(s);
sleep(1000)
scf(1)
plot (y,'b')
plot (TQ,'r')
[lin col]=size(Q);
erro=(1/(col))*sum((y-TQ)^2)
