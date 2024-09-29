import numpy as np
import matplotlib.pyplot as plt
from deap import creator, base, tools, algorithms
import random
import math
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from rescuer import Rescuer
import clustering
import joblib
from shared_instances import SharedInstances
import csv
import Astar
import os


cromossomoGlobal = [()]  # (x, y, gravidade)
posicao_resgatador = (0, 0)
mapa = None  # Inicializando o mapa global

def calcular_distancia(x1, x2):
    return ((x1[0] - x2[0])**2 + (x1[1] - x2[1])**2) ** 0.5

def carregarDados():
    clusters = {}
    for i in range(1, 5):
        file_name = f"clusters_seqs\cluster{i}.txt"
        df = pd.read_csv(file_name, header=None)
        indices_desejados = [0, 1, 2, 3, 4, 5]  # Todas as colunas
        df_selecionado = df.iloc[:, indices_desejados]
        df_selecionado.columns = ["id", "x", "y", "qPA", "Pulso", "Frequência Respiratória"]
        vetor_cluster = df_selecionado.values.tolist()
        clusters[f'cluster_{i}'] = vetor_cluster

    vetor_cluster1 = clusters['cluster_1']
    vetor_cluster2 = clusters['cluster_2']
    vetor_cluster3 = clusters['cluster_3']
    vetor_cluster4 = clusters['cluster_4']

    return vetor_cluster1, vetor_cluster2, vetor_cluster3, vetor_cluster4

def CalcularGrauDePrioridade(vetor):
    decision_tree = joblib.load('modelo_arvore_decisao.pkl')
    vetor_np = np.array(vetor)
    coordenadas_prioridade = vetor_np[:, [3, 4, 5]]
    colunas = ["qPA", "Pulso", "frequência Respiratória"]
    vetor_df = pd.DataFrame(coordenadas_prioridade, columns=colunas)
    classe_prevista = decision_tree.predict(vetor_df)
    probabilidades_previstas = decision_tree.predict_proba(vetor_df)
    vetor_com_previsao = np.hstack((vetor_np, classe_prevista.reshape(-1, 1)))
    

    return vetor_com_previsao

def avaliar_cromossomo(cromossomo):
    dist_total = 0
    penalidade = 0
    posicao_atual = posicao_resgatador

    for i, vitima in enumerate(cromossomo):
        coordenada_vitima = (int(vitima[1]), int(vitima[2]))
        gravidade = vitima[3]
        
        if not mapa.in_map(coordenada_vitima) or calcular_distancia(posicao_atual, coordenada_vitima) > 1:
            penalidade += 1000
        else:
            dist_total += calcular_distancia(posicao_atual, coordenada_vitima)
            posicao_atual = coordenada_vitima

        penalidade += i * gravidade

    dist_total += calcular_distancia(posicao_atual, posicao_resgatador)
    
    return dist_total + penalidade,

def mapear_para_indices(individuo, vitimas):
    return [vitimas.index(vitima) for vitima in individuo]

def mapear_para_tuplas(individuo_indices, vitimas):
    return [vitimas[i] for i in individuo_indices]

def crossover_pmx(ind1, ind2):
    ind1_indices = mapear_para_indices(ind1, cromossomoGlobal)
    ind2_indices = mapear_para_indices(ind2, cromossomoGlobal)
    tools.cxPartialyMatched(ind1_indices, ind2_indices)
    ind1[:] = mapear_para_tuplas(ind1_indices, cromossomoGlobal)
    ind2[:] = mapear_para_tuplas(ind2_indices, cromossomoGlobal)
    return ind1, ind2

def salvaEstatisticas(individuo):
    return individuo.fitness.values

def preencheSequencia():
    global vetor1, vetor2, vetor3, vetor4

    vetor1, vetor2, vetor3, vetor4 = carregarDados()
    
    V1Calculado = CalcularGrauDePrioridade(vetor1)
    V2Calculado = CalcularGrauDePrioridade(vetor2)
    V3Calculado = CalcularGrauDePrioridade(vetor3)
    V4Calculado = CalcularGrauDePrioridade(vetor4)

    colunas_desejadas1 = [tuple(p) for p in np.array(V1Calculado)[:, [0, 1, 2, 6]]]
    colunas_desejadas2 = [tuple(p) for p in np.array(V2Calculado)[:, [0, 1, 2, 6]]]
    colunas_desejadas3 = [tuple(p) for p in np.array(V3Calculado)[:, [0, 1, 2, 6]]]
    colunas_desejadas4 = [tuple(p) for p in np.array(V4Calculado)[:, [0, 1, 2, 6]]]
    #print(colunas_desejadas1)
    #input(f"{self.NAME}: type [ENTER] to proc--eed")
    return colunas_desejadas1, colunas_desejadas2, colunas_desejadas3, colunas_desejadas4

def retorna_sequencia(clusterVitimas):
    if "FitnessMin" not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)

    if len(clusterVitimas) == 1:
        return clusterVitimas
    if len(clusterVitimas) == 0:
        print("Não há vítimas para salvar")
        return 0

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: random.sample(clusterVitimas, len(clusterVitimas)))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", avaliar_cromossomo)
    toolbox.register("mate", crossover_pmx)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)

    populacao = toolbox.population(n=30)
    estatistica = tools.Statistics(salvaEstatisticas)
    estatistica.register("mean", np.mean)
    estatistica.register("min", np.min)
    estatistica.register("max", np.max)

    num_geracoes = 40
    algorithms.eaSimple(populacao, toolbox, cxpb=0.8, mutpb=0.1, ngen=num_geracoes, stats=estatistica, halloffame=None, verbose=True)
    melhor_individuo = tools.selBest(populacao, 1)[0]

    return melhor_individuo

def Iniciar(map):
    global mapa, cromossomoGlobal
    mapa = map

    cromossomo1, cromossomo2, cromossomo3, cromossomo4 = preencheSequencia()

    cromossomoGlobal = cromossomo1
    seq1 = retorna_sequencia(cromossomoGlobal)

    print(f"Melhor Sequência: {seq1}")
    print(f"Fitness da melhor Sequência: {avaliar_cromossomo(seq1)}")

    print("\n=========================== OUTRA SEQUENCIA AGORA =================================\n")
    with open(r"clusters_seqs\seq1.txt", "w") as file:
      for tupla in seq1:
        # Supondo que 'vetorCluster1' é um vetor que você está comparando com a sequência
        if tupla in seq1:
            # Criar a string no formato desejado
            strCluster = f"{int(tupla[0])}, {int(tupla[1])}, {int(tupla[2])}, 0, {int(tupla[3])}"
            # Escrever a string no arquivo
            file.write(strCluster + "\n")
    cromossomoGlobal = cromossomo2
    seq2 = retorna_sequencia(cromossomoGlobal)
    print(f"Melhor Sequência: {seq2}")
    print(f"Fitness da melhor Sequência: {avaliar_cromossomo(seq2)}")

    seq_dict = read_seq_file('clusters_seqs\seq1.txt')
    update_cluster_file('clusters_seqs\cluster1.txt', seq_dict, 'clusters_seqs\clusterN1.txt')
    replace_file_content('clusters_seqs\cluster1.txt', 'clusters_seqs\clusterN1.txt')

    print("\n=========================== OUTRA SEQUENCIA AGORA =================================\n")
    with open(r"clusters_seqs\seq2.txt", "w") as file:
      for tupla in seq2:
        # Supondo que 'vetorCluster1' é um vetor que você está comparando com a sequência
        if tupla in seq2:
            # Criar a string no formato desejado
            strCluster = f"{int(tupla[0])}, {int(tupla[1])}, {int(tupla[2])}, 0, {int(tupla[3])}"
            # Escrever a string no arquivo
            file.write(strCluster + "\n")

    cromossomoGlobal = cromossomo3
    seq3 = retorna_sequencia(cromossomoGlobal)
    print(f"Melhor Sequência: {seq3}")
    print(f"Fitness da melhor Sequência: {avaliar_cromossomo(seq3)}")

    seq_dict = read_seq_file('clusters_seqs\seq2.txt')
    update_cluster_file('clusters_seqs\cluster2.txt', seq_dict, 'clusters_seqs\clustern2.txt')
    replace_file_content('clusters_seqs\cluster2.txt', 'clusters_seqs\clustern2.txt')

    print("\n=========================== OUTRA SEQUENCIA AGORA =================================\n")
    with open(r"clusters_seqs\seq3.txt", "w") as file:
      for tupla in seq3:
        # Supondo que 'vetorCluster1' é um vetor que você está comparando com a sequência
        if tupla in seq3:
            # Criar a string no formato desejado
            strCluster = f"{int(tupla[0])}, {int(tupla[1])}, {int(tupla[2])}, 0, {int(tupla[3])}"
            # Escrever a string no arquivo
            file.write(strCluster + "\n")
    cromossomoGlobal = cromossomo4
    seq4 = retorna_sequencia(cromossomoGlobal)
    print(f"Melhor Sequência: {seq4}")
    print(f"Fitness da melhor Sequência: {avaliar_cromossomo(seq4)}")

    seq_dict = read_seq_file('clusters_seqs\seq3.txt')
    update_cluster_file('clusters_seqs\cluster3.txt', seq_dict, 'clusters_seqs\clustern3.txt')
    replace_file_content('clusters_seqs\cluster3.txt', 'clusters_seqs\clustern3.txt')

    print("\n=========================== OUTRA SEQUENCIA AGORA =================================\n")
    with open(r"clusters_seqs\seq4.txt", "w") as file:
      for tupla in seq4:
        # Supondo que 'vetorCluster1' é um vetor que você está comparando com a sequência
        if tupla in seq4:
            # Criar a string no formato desejado
            strCluster = f"{int(tupla[0])}, {int(tupla[1])}, {int(tupla[2])}, 0, {int(tupla[3])}"
            # Escrever a string no arquivo
            file.write(strCluster + "\n")
    seq_dict = read_seq_file('clusters_seqs\seq4.txt')
    update_cluster_file('clusters_seqs\cluster4.txt', seq_dict, 'clusters_seqs\clustern4.txt')
    replace_file_content('clusters_seqs\cluster4.txt', 'clusters_seqs\clustern4.txt')

    seq1 = [tuple(p) for p in np.array(seq1)[:, [1, 2, 3]]]
    seq2 = [tuple(p) for p in np.array(seq2)[:, [1, 2, 3]]]
    seq3 = [tuple(p) for p in np.array(seq3)[:, [1, 2, 3]]]
    seq4 = [tuple(p) for p in np.array(seq4)[:, [1, 2, 3]]]

    caminho1 = Astar.a_star_sequencia(mapa, seq1)
    caminho2 = Astar.a_star_sequencia(mapa, seq2)
    caminho3 = Astar.a_star_sequencia(mapa, seq3)
    caminho4 = Astar.a_star_sequencia(mapa, seq4)
    print('--------------------------------')
    print(caminho1)
    print('--------------------------------')

    direção1 = direcoes_para_percorrer(caminho1)
    direção2 = direcoes_para_percorrer(caminho2)
    direção3 = direcoes_para_percorrer(caminho3)
    direção4 = direcoes_para_percorrer(caminho4)

    print(direção1)

    SharedInstances.resc1.go_save_victims(mapa, vetor1, direção1)
    SharedInstances.resc2.go_save_victims(mapa, vetor2, direção2)
    SharedInstances.resc3.go_save_victims(mapa, vetor3, direção3)
    SharedInstances.resc4.go_save_victims(mapa, vetor4, direção4)


def direcoes_para_percorrer(pontos):
    """Converte uma lista de coordenadas em mudanças (dx, dy) a serem seguidas para percorrer os pontos."""
    
    # Extrair as colunas (x, y) das coordenadas
    colunas_1_e_2 = [[linha[0], linha[1]] for linha in pontos]

    # Extrair a coluna restante (z) das coordenadas
    colunas_restantes = [linha[2] for linha in pontos]

    # Calcular as mudanças (dx, dy)
    mudancas = []
    for i in range(len(colunas_1_e_2) - 1):
        x1, y1 = colunas_1_e_2[i]
        x2, y2 = colunas_1_e_2[i + 1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        mudancas.append((dx, dy))
    
    # Juntar mudanças (dx, dy) com a coluna restante (z)
    resultado = []
    for i in range(len(mudancas)):
        if i < len(colunas_restantes) - 1:
            resultado.append(list(mudancas[i]) + [colunas_restantes[i + 1]])
    
    print("Mudanças:")
    print(mudancas)
    print("Colunas Restantes:")
    print(colunas_restantes)
    print("Resultado:")
    print(resultado)

    resultado_convertido = [(int(x), int(y), int(z)) for x, y, z in resultado]

    return resultado_convertido

def read_seq_file(seq_file):
    seq_dict = {}
    with open(seq_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # Converter strings para números e remover espaços em branco
            key = tuple(map(lambda x: int(x.strip()), row[:3]))
            values = tuple(map(lambda x: int(x.strip()), row[3:]))
            seq_dict[key] = values
    return seq_dict

# Função para atualizar o arquivo cluster1.txt com base no seq_dict
def update_cluster_file(cluster_file, seq_dict, updated_cluster_file):
    with open(cluster_file, 'r') as infile, open(updated_cluster_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for row in reader:
            key = tuple(map(lambda x: int(float(x.strip())), row[:3]))
            if key in seq_dict:
                updated_row = list(key) + list(seq_dict[key])
                writer.writerow(updated_row)
            else:
                print(f"Chave {key} não encontrada em seq1.txt. Linha original mantida.")
                writer.writerow(row)

def replace_file_content(original_file, updated_file):
    # Ler o conteúdo do arquivo atualizado
    with open(updated_file, 'r') as src:
        updated_content = src.read()

    # Sobrescrever o conteúdo do arquivo original
    with open(original_file, 'w') as dest:
        dest.write(updated_content)

    # Deletar o arquivo atualizado
    os.remove(updated_file)