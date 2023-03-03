# Projeto Ciência de Dados - Previsão de Vendas

- Nosso desafio é conseguir prever as vendas que vamos ter em determinado período com base nos gastos em anúncios nas 3 grandes redes que a empresa Hashtag investe: TV, Jornal e Rádio
- TV, Jornal e Rádio estão em milhares de reais
- Vendas estão em milhões



### Passo a Passo de um Projeto de Ciência de Dados

- Passo 1: Entendimento do Desafio
- Passo 2: Entendimento da Área/Empresa
- Passo 3: Extração/Obtenção de Dados
- Passo 4: Ajuste de Dados (Tratamento/Limpeza)
- Passo 5: Análise Exploratória
- Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
- Passo 7: Interpretação de Resultados

#### Importar a Base de dados


```python
!pip install matplotlib
!pip install seaborn
!pip install scikit-learn
```

    Requirement already satisfied: matplotlib in c:\users\python\anaconda3\lib\site-packages (3.3.4)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\python\anaconda3\lib\site-packages (from matplotlib) (1.3.1)
    Requirement already satisfied: cycler>=0.10 in c:\users\python\anaconda3\lib\site-packages (from matplotlib) (0.10.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\python\anaconda3\lib\site-packages (from matplotlib) (2.4.7)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\python\anaconda3\lib\site-packages (from matplotlib) (2.8.1)
    Requirement already satisfied: numpy>=1.15 in c:\users\python\anaconda3\lib\site-packages (from matplotlib) (1.20.1)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\python\anaconda3\lib\site-packages (from matplotlib) (8.2.0)
    Requirement already satisfied: six in c:\users\python\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib) (1.15.0)
    Requirement already satisfied: seaborn in c:\users\python\anaconda3\lib\site-packages (0.11.1)
    Requirement already satisfied: numpy>=1.15 in c:\users\python\anaconda3\lib\site-packages (from seaborn) (1.20.1)
    Requirement already satisfied: scipy>=1.0 in c:\users\python\anaconda3\lib\site-packages (from seaborn) (1.6.2)
    Requirement already satisfied: matplotlib>=2.2 in c:\users\python\anaconda3\lib\site-packages (from seaborn) (3.3.4)
    Requirement already satisfied: pandas>=0.23 in c:\users\python\anaconda3\lib\site-packages (from seaborn) (1.2.4)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in c:\users\python\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (2.4.7)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\python\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (8.2.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\python\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (1.3.1)
    Requirement already satisfied: python-dateutil>=2.1 in c:\users\python\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (2.8.1)
    Requirement already satisfied: cycler>=0.10 in c:\users\python\anaconda3\lib\site-packages (from matplotlib>=2.2->seaborn) (0.10.0)
    Requirement already satisfied: six in c:\users\python\anaconda3\lib\site-packages (from cycler>=0.10->matplotlib>=2.2->seaborn) (1.15.0)
    Requirement already satisfied: pytz>=2017.3 in c:\users\python\anaconda3\lib\site-packages (from pandas>=0.23->seaborn) (2021.1)
    Requirement already satisfied: scikit-learn in c:\users\python\anaconda3\lib\site-packages (0.24.1)
    Requirement already satisfied: numpy>=1.13.3 in c:\users\python\anaconda3\lib\site-packages (from scikit-learn) (1.20.1)
    Requirement already satisfied: joblib>=0.11 in c:\users\python\anaconda3\lib\site-packages (from scikit-learn) (1.0.1)
    Requirement already satisfied: scipy>=0.19.1 in c:\users\python\anaconda3\lib\site-packages (from scikit-learn) (1.6.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in c:\users\python\anaconda3\lib\site-packages (from scikit-learn) (2.1.0)
    


```python
import pandas as pd

tabela = pd.read_csv("advertising.csv")
display(tabela)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Jornal</th>
      <th>Vendas</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>230.1</td>
      <td>37.8</td>
      <td>69.2</td>
      <td>22.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.5</td>
      <td>39.3</td>
      <td>45.1</td>
      <td>10.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17.2</td>
      <td>45.9</td>
      <td>69.3</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>151.5</td>
      <td>41.3</td>
      <td>58.5</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>180.8</td>
      <td>10.8</td>
      <td>58.4</td>
      <td>17.9</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>38.2</td>
      <td>3.7</td>
      <td>13.8</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>196</th>
      <td>94.2</td>
      <td>4.9</td>
      <td>8.1</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>197</th>
      <td>177.0</td>
      <td>9.3</td>
      <td>6.4</td>
      <td>14.8</td>
    </tr>
    <tr>
      <th>198</th>
      <td>283.6</td>
      <td>42.0</td>
      <td>66.2</td>
      <td>25.5</td>
    </tr>
    <tr>
      <th>199</th>
      <td>232.1</td>
      <td>8.6</td>
      <td>8.7</td>
      <td>18.4</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 4 columns</p>
</div>


#### Análise Exploratória
- Vamos tentar visualizar como as informações de cada item estão distribuídas
- Vamos ver a correlação entre cada um dos itens


```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(tabela.corr(), annot=True, cmap="Wistia")
plt.show()

# outra forma de ver a mesma análise
# sns.pairplot(tabela)
# plt.show()
```


    
![png](output_6_0.png)
    


#### Com isso, podemos partir para a preparação dos dados para treinarmos o Modelo de Machine Learning

- Separando em dados de treino e dados de teste


```python
from sklearn.model_selection import train_test_split

y = tabela["Vendas"]
x = tabela.drop("Vendas", axis=1)

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3, random_state=1)
```

#### Temos um problema de regressão - Vamos escolher os modelos que vamos usar:

- Regressão Linear
- RandomForest (Árvore de Decisão)


```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# cria as inteligencias aritificiais
modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treina as inteligencias artificias
modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)
```




    RandomForestRegressor()



#### Teste da AI e Avaliação do Melhor Modelo

- Vamos usar o R² -> diz o % que o nosso modelo consegue explicar o que acontece


```python
from sklearn import metrics

# criar as previsoes
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# comparar os modelos
print(metrics.r2_score(y_teste, previsao_regressaolinear))
print(metrics.r2_score(y_teste, previsao_arvoredecisao))  
```

    0.9071151423684273
    0.9634775407906989
    

#### Visualização Gráfica das Previsões


```python
tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsoes ArvoreDecisao"] = previsao_arvoredecisao
tabela_auxiliar["Previsoes Regressao Linear"] = previsao_regressaolinear

plt.figure(figsize=(15,6))
sns.lineplot(data=tabela_auxiliar)
plt.show()
```


    
![png](output_14_0.png)
    


#### Como fazer uma nova previsao?


```python
# Como fazer uma nova previsao
# importar a nova_tabela com o pandas (a nova tabela tem que ter os dados de TV, Radio e Jornal)
# previsao = modelo_randomforest.predict(nova_tabela)
# print(previsao)
nova_tabela = pd.read_csv("novos.csv")
display(nova_tabela)
previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>TV</th>
      <th>Radio</th>
      <th>Jornal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23.1</td>
      <td>3.8</td>
      <td>69.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44.5</td>
      <td>0.0</td>
      <td>5.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>170.2</td>
      <td>45.9</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    [ 7.509  8.531 19.814]
    

#### Qual a importância de cada variável para as vendas?


```python
sns.barplot(x=x_treino.columns, y=modelo_arvoredecisao.feature_importances_)
plt.show()

# Caso queira comparar Radio com Jornal
# print(df[["Radio", "Jornal"]].sum())
```


    
![png](output_18_0.png)
    



```python

```
