# Funções para pré-processamento de dados
def inteiros(x):
  inteiros = np.vectorize(int)
  return inteiros(x)

def class_binaria(feature, classe, dataset):

  feature_nova = []

  for letra in feature:

    if letra == classe:
      letra = 1
    else:
      letra = 0
    feature_nova.append(letra)

  feature_nova = np.asarray(feature_nova)
  dataset = np.hstack((dataset,feature_nova.reshape(-1,1)))
  return dataset

def valor_minimo(x):
  return np.min(x)
def valor_maximo(x):
  return np.max(x)

def convert(x):
    # Dimensões da matriz a ser convertida
    linhas = len(x)
    colunas = len(x[0])

    # Array vázio
    array_do_numpy = [[0] * colunas for _ in range(linhas)]

    # Loop que faz a cópia e converção para um array Numpy
    for i in range(linhas):
      for j in range(colunas):
        array_do_numpy[i][j] = x[i][j]
    array_do_numpy = np.asarray(array_do_numpy)
    array_do_numpy = array_do_numpy.astype('float64')
    return array_do_numpy

def normaliza(dataset, novo_maximo = 1, novo_minimo = 0):
  dataset_novo = (dataset - valor_minimo(dataset)) * (novo_maximo - novo_minimo) / (valor_maximo(dataset) - valor_minimo(dataset)) + novo_minimo
  return dataset_novo

def corelacao(dataset, feature_analisada, feature_inicial, feature_final):
  while feature_inicial <= feature_final:
    analise = np.corrcoef(dataset[:,feature_analisada], dataset[:,feature_inicial])
    print(f'correlação da feature {feature_analisada} com {feature_inicial}: {analise[0,1]}')
    feature_inicial += 1
  return None

def media(dataset, n_de_features, ):
  features = 0
  t = 0
  while t < n_de_features:
    feature = int(input('Digite a coluna da feature: '))
    features += dataset[:,feature]
    t += 1
  media_das_features = features/n_de_features
  return media_das_features
