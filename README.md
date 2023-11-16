# mba-famesp-machine-learning
Projeto de Machine Learning / MBA Famesp

# Projeto de Previsão de Turnover em Operação de Atendimento.
- Após o projeto de Analytics onde resultou em um PowerBi para acompanhamento do Turnover e dos analistas, foi solicitado um odelo que ajude a prever quais analistas estão mais propensos a pedir desligamento para que assim medidas preventivas possam ser tomadas pela empresa em questão.

# Passos seguidos no projeto:

## Análise Exploratória de Dados (EDA):

- Explorou-se a base de dados, identificando características e padrões iniciais.
- Realizou-se uma análise estatística e visual para compreender a distribuição das variáveis.

## Pré-processamento de Dados:

- Tratamento de valores ausentes e outliers.
- Transformação e codificação de variáveis categóricas.

## Análise de Correlação e Seleção de Variáveis:

- Utilizou-se correlação para entender a relação entre variáveis.
- Seleção de variáveis com base em critérios estatísticos e práticos.

## Construção e Avaliação do Modelo:

- Treinamento de diferentes modelos de classificação.
- Utilização de técnicas de validação cruzada (k-fold) para avaliação.
- Ajuste de hiperparâmetros por meio de Grid Search.
- Avaliação do desempenho dos modelos utilizando métricas relevantes (accuracy, precision, recall, f1-score).
- Escolha do modelo final (Random Forest).

## Interpretação do Modelo:

- Análise da importância das variáveis no modelo final.
- Avaliação de como o modelo toma decisões.

## Previsão em Novos Dados:

- Aplicação do modelo treinado para prever novos dados (Data_previsões).

## Interpretação do Modelo

### Importância das Features:

O modelo Random Forest atribuiu importâncias diferentes às variáveis, indicando quais características mais influenciam nas previsões. Abaixo estão as importâncias das features calculadas pelo modelo:

## Interpretação do Modelo

### Importância das Features:

O modelo Random Forest atribuiu importâncias diferentes às variáveis, indicando quais características mais influenciam nas previsões. Abaixo estão as importâncias das features calculadas pelo modelo:

Visualização da Árvore de Decisão:
A imagem abaixo representa uma árvore de decisão do modelo, mostrando como o algoritmo toma decisões com base nas variáveis do conjunto de dados.

Árvore de Decisão

![Árvore de Decisão](https://github.com/Sam-Batisti/mba-famesp-machine-learning/blob/main/Arvore_decis%C3%A3o.png)


```python
feature_importances = best_rf_model.feature_importances_
print("Importância das Features:")
for feature, importance in zip(X_train.columns, feature_importances):
    print(f"{feature}: {importance}")

Hrs._Mensais: 0.0017
C_TEMPO_LOGADO_MEDIA_sum: 0.0254
C_AHT_sum: 0.0442
C_HORAS_PRODUTIVAS_mean: 0.0355
C_HORAS_PRODUTIVAS_sum: 0.0536
C_TEMPO_DESLOGADO_sum: 0.0529
C_TEMPO_LOGADO_MEDIA_mean: 0.0720
C_ABS_mean: 0.1286
C_ADR_mean: 0.0690
C_HORAS_EM_PAUSA_mean: 0.1355
C_PAUSAS_NR17_mean: 0.0489
Desc.Funcao_Analista_Atendimento: 0.0332
Desc.Funcao_Analista_Jr: 0.0018
Desc.Funcao_Analista_Midias_Sociais: 0.0119
Desc.Funcao_Analista_Pleno: 0.0014
Desc.Funcao_Jovem_Aprendiz: 0.0038
Celula_Customer Care - Voz: 0.0112
Celula_Mídias sociais: 0.0101
Celula_Reclame aqui: 0.0015
Bairro_ARISTON: 0.0
Bairro_BRAS: 0.0008
Bairro_CANTINHO DO CEU: 0.0023
Bairro_CASA VERDE ALTA: 0.0
Bairro_CHACARA SAO JOS: 0.0080
Bairro_CHACARAS VERDES: 0.0008
Bairro_CIDADE TIRADENT: 0.0037
Bairro_COLONIA: 0.0
Bairro_CONJUNTO HABITA: 0.0228
Bairro_FRANCISCO MATAR: 0.0
Bairro_FREGUESIA DO O: 0.0
Bairro_HELENA MARIA: 0.0
Bairro_IMIRIM: 0.0
Bairro_IPES: 0.0003
Bairro_IPIRANGA: 0.0
Bairro_ITAQUERA: 0.0
Bairro_JARDIM ANA ESTE: 0.0020
Bairro_JARDIM APUANA: 0.0
Bairro_JARDIM CELIA: 0.0
Bairro_JARDIM COIMBRA: 0.0238
Bairro_JARDIM DAMASCEN: 0.0000
Bairro_JARDIM DAS IMBU: 0.0000
Bairro_JARDIM DO CAMPO: 0.0000
Bairro_JARDIM GRACINDA: 0.0000
Bairro_JARDIM HELENA: 0.0000
Bairro_JARDIM IGUATEMI: 0.0000
Bairro_JARDIM JARAGUA: 0.0002
Bairro_JARDIM JULIO: 0.0015
Bairro_JARDIM LAURA: 0.0
Bairro_JARDIM LIBANO: 0.0
Bairro_JARDIM LOURDES: 0.0
Bairro_JARDIM MANGALOT: 0.0
Bairro_JARDIM MARABA: 0.0
Bairro_JARDIM MARISTEL: 0.0110
Bairro_JARDIM MELO: 0.0011
Bairro_JARDIM PALMIRA: 0.0069
Bairro_JARDIM PAULISTA: 0.0
Bairro_JARDIM PROGRESS: 0.0
Bairro_JARDIM ROSANA: 0.0
Bairro_JARDIM SANTA MA: 0.0000
Bairro_JARDIM SANTA RO: 0.0
Bairro_JARDIM SAO BERN: 0.0
Bairro_JARDIM SAO JORG: 0.0
Bairro_JARDIM SAO LUIS: 0.0
Bairro_JARDIM SYDNEY: 0.0017
Bairro_JARDIM TRES MAR: 0.0000
Bairro_JARDIM VERONIA: 0.0143
Bairro_JARDIM VILA FOR: 0.0
Bairro_JARDM FONTALIS: 0.0
Bairro_JD CAPAO REDON: 0.0
Bairro_JD CAROMBE: 0.0004
Bairro_JD IMPERADOR: 0.0
Bairro_JD IPORA: 0.0
Bairro_JD MITSUTANI: 0.0
Bairro_JD ROBERTO: 0.0
Bairro_JD SANTA FE: 0.0013
Bairro_JD SAO BENTO NV: 0.0088
Bairro_JDM VASSOURAS I: 0.0
Bairro_LAGO AZUL: 0.0130
Bairro_NOVO OSASCO: 0.0
Bairro_PARI: 0.0
Bairro_PARQUE BRASIL G: 0.0020
Bairro_PARQUE CISPER: 0.0
Bairro_PARQUE COCAIA: 0.0000
Bairro_PARQUE DOS LAGO: 0.0
Bairro_PARQUE RESIDENC: 0.0
Bairro_PARQUE SAVOY CI: 0.0
Bairro_PINHEIROS: 0.0002
Bairro_PIRITUBA: 0.0079
Bairro_PQ EDU CHAVES: 0.0008
Bairro_PQ MIKAIL: 0.0114
Bairro_PQ PAULISTA: 0.0
Bairro_PQ SANT ANTONIO: 0.0017
Bairro_SACOMA: 0.0001
Bairro_SOMMA: 0.0156
Bairro_VARZEA DA BARRA: 0.0
Bairro_VELOSO: 0.0001
Bairro_VILA ANTONIETA: 0.0001
Bairro_VILA ARICANDUVA: 0.0000
Bairro_VILA AUREA: 0.0002
Bairro_VILA BRASILEIRA: 0.0
Bairro_VILA CARMEM: 0.0
Bairro_VILA CARRAO: 0.0022
Bairro_VILA CRETY: 0.0
Bairro_VILA DAS BELEZA: 0.0
Bairro_VILA ESMERALDA: 0.0
Bairro_VILA GUILHERMIN: 0.0
Bairro_VILA ITAIM: 0.0
Bairro_VILA JACUI: 0.0001
Bairro_VILA JULIA: 0.0084
Bairro_VILA MARIA: 0.0
Bairro_VILA MENCK: 0.0001
Bairro_VILA NATAL: 0.0
Bairro_VILA NOSSA SRA: 0.0
Bairro_VILA NOVA: 0.0000
Bairro_VILA NV PINHEIR: 0.0
Bairro_VILA PEREIRA BA: 0.0057
Bairro_VILA ROSEIRA LL: 0.0021
Bairro_VILA SANTA MARI: 0.0179
Bairro_VILA SAO RAFAEL: 0.0
Bairro_VILA SONIA: 0.0
Bairro_VILA SUZANA: 0.0015
Bairro_VILA TEREZINHA: 0.0005
Bairro_VILA URUPES: 0.0054
Bairro_VILA VIRGINIA: 0.0010
Bairro_VL ZULMIRA MARI: 0.0
Municipio_BARUERI: 0.0000
Municipio_CAIEIRAS: 0.0000
Municipio_CAJAMAR: 0.0002
Municipio_CARAPICUIBA: 0.0003
Municipio_FRACISCO MORATO: 0.0000
Municipio_FRANCISCO MORATO: 0.0
Municipio_FRANCO DA ROCHA: 0.0092
Municipio_GUARULHOS: 0.0039
Municipio_ITAQUAQUECETUBA: 0.0000
Municipio_JANDIRA: 0.0000
Municipio_MAIRIPORA: 0.0
Municipio_MOGI DAS CRUZES: 0.0
Municipio_OSASCO: 0.0001
Municipio_POA: 0.0052
Municipio_RIBEIRAO PIRES: 0.0082
Municipio_SANTO ANDRE: 0.0006
Municipio_SAO PAULO: 0.0150
Municipio_SUZANO: 0.0046
Municipio_TABOAO DA SERRA: 0.0005
