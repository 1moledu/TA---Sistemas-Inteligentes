# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import socorro
import os
import random
import csv
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from rescuer import Rescuer
import clustering

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0
    
    def size (self):
        return len(self.items)

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, direcao, delibera):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = Stack()  # a stack to store the movements
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.time_to_comeback = math.ceil(self.TLIM * 0.20)  # set the time to come back to the base
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals

        #Chama o socorro
        self.Delibera = delibera

        #saber o momento de unir os mapas
        self.retornou = False
        #usado para sair de "buracos" (por enquanto)
        self.pilhaRetorno = Stack()
        #usada para alimentar a pilha de retorno
        self.flagDeRetorno = 1
        #um vetor com o caminho percorrido, igual a pilha walkstack
        self.vetorDoCaminho = []

        self.pilhaTeste = Stack()
        #define a direçao que o explorador tende a seguir
        self.direcao = direcao
        #usada para que nao se repita a criaçao do vetor de retorno
        self.flagCalculo = 0
        #vetor usado para o comeback
        self.vetorDeRetorno = []
        #sinaliza que ja passou o mapa todo
        self.flagTodo = 1
        #
        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_next_position(self):
        """ Randomically, gets the next position that can be explored (no wall and inside the grid)
            There must be at least one CLEAR position in the neighborhood, otherwise it loops forever.
        """
        A =  int (0), int (-1),  #  u: Up
        B =  int (1), int (-1),  # ur: Upper right diagonal
        C =  int (1), int (0),   #  r: Right
        D =  int (1), int (1),   # dr: Down right diagonal
        E =  int (0), int (1),   #  d: Down
        F =  int (-1), int (1),  # dl: Down left left diagonal
        G =  int (-1), int (0),  #  l: Left
        H =  (int (-1), int (-1))  # ul: Up left diagonal
        
        self.vet = [A, B, C, D, E, F, G, H]

        possicaovetor = self.direcao
        contador = 0
        # Check the neighborhood walls and grid limits
        obstacles = self.check_walls_and_lim()

        # Loop until a CLEAR position is found
        while True:
            # Get a random direction
            
            direction = possicaovetor

            possivelposicaox = (self.vet[possicaovetor])[0] + self.x
            possivelposicaoy = (self.vet[possicaovetor])[1] + self.y
            possivelposicao = (possivelposicaox, possivelposicaoy)

            # Check if the corresponding position in walls_and_lim is CLEAR
            
            if obstacles[direction] == VS.CLEAR and not self.map.in_map(possivelposicao):
                self.flagDeRetorno = 1
                return Explorer.AC_INCR[direction] 
            
            #se não tenho mais como seguir faco ele voltar para a ultima posicao com vizinhos livres
            elif obstacles[direction] == VS.CLEAR and self.map.in_map(possivelposicao) and contador > 7:
                self.flagDeRetorno = 0
                return Explorer.AC_INCR[self.recuarCaminho()]  
            else:
                if possicaovetor < 7:
                    possicaovetor += 1
                else:
                    possicaovetor = 0
            
            contador+=1
        
        
    def explore(self):
        # get an random increment for x and y       
        dx, dy = self.get_next_position()

        # Moves the body to another position
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()

        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.push((dx, dy))
            
            ################################################################
            #só atualiza quando sai do "buraco"
            if self.flagDeRetorno :
                self.pilhaRetorno.push((dx, dy))

            ################################################################    
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy        


            self.vetorDoCaminho.append((self.x, self.y))



            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

        return

    def come_back(self):
        dx,dy = self.vetorDeRetorno.pop(0)
        print(dx, dy)
        dx =  self.x - dx
        dy =  self.y - dy

        dx = dx * -1
        dy = dy * -1

        result = self.walk(dx, dy)
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            return
        
        if result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            print(f"{self.NAME}: coming back at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        if self.get_rtime() > self.time_to_comeback and self.flagTodo:
            self.explore()
            return True
        else:
            # time to come back to the base
            #if self.walk_stack.is_empty():
            if  self.x == 0 and self.y == 0:
                # time to wake up the rescuer
                # pass the walls and the victims (here, they're empty)
                print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
                
              
                #mostra que o explorador voltou
                self.retornou = True

                mergedmap = Map
                chegoutodos = bool

                #se todos retornaram, o metodo retorna true
                chegoutodos, mergedmap = clustering.mergeMapas()
                

                if chegoutodos:

                                       
                    clustering.mapaTXT(mergedmap)
                    vetorVitm = clustering.vetorVitimas(mergedmap)
                    clusters, centroids = clustering.kmeans(vetorVitm, 4)

                    
                    print("Clusters:")
                    
                    print(clusters)

                    vetorCluster1 = []
                    vetorCluster2 = []
                    vetorCluster3 = []
                    vetorCluster4 = []
                    for i in range(len(vetorVitm)):
                        if clusters[i] == 0:
                            vetorCluster1.append(vetorVitm[i])
                        elif clusters[i] == 1:
                            vetorCluster2.append(vetorVitm[i])
                        elif clusters[i] == 2:
                            vetorCluster3.append(vetorVitm[i])
                        elif clusters[i] == 3:
                            vetorCluster4.append(vetorVitm[i])
                        
                    #print(vetorCluster1)

                    
                    #print(vetorCluster2)
                    #print(vetorCluster3)
                    #print(vetorCluster4)

                    #vs[0]<seq, pSist, pDiast, qPA, pulse, respiratory freq>
                    
                    #vs[3], vs[4], vs[5]
                    #print(clustering.dictVitimas())
                    with open(r"clusters_seqs\cluster1.txt", 'w') as arquivo, \
                        open(r"clusters_seqs\cluster2.txt", 'w') as arquivo2, \
                        open(r"clusters_seqs\cluster3.txt", 'w') as arquivo3, \
                        open(r"clusters_seqs\cluster4.txt", 'w') as arquivo4:
                        for id,(tupla,vetor) in clustering.dictVitimas():
                            for i in range(len(vetorCluster1)):
                                if tupla == vetorCluster1[i]:
                                    #print(strCluster)
                                    strCluster=f"{id},{tupla[0]},{tupla[1]},{vetor[3]},{vetor[4]},{vetor[5]}"
                                    
                                    arquivo.write(strCluster+"\n")    
                            for i in range(len(vetorCluster2)):
                                if tupla == vetorCluster2[i]:
                                    strCluster=f"{id},{tupla[0]},{tupla[1]},{vetor[3]},{vetor[4]},{vetor[5]}"
                                    
                                    arquivo2.write(strCluster+"\n")
                            for i in range(len(vetorCluster3)):
                                if tupla == vetorCluster3[i]:
                                    strCluster=f"{id},{tupla[0]},{tupla[1]},{vetor[3]},{vetor[4]},{vetor[5]}"
                                    
                                    arquivo3.write(strCluster+"\n")
                            for i in range(len(vetorCluster4)):
                                if tupla == vetorCluster4[i]:
                                    strCluster=f"{id},{tupla[0]},{tupla[1]},{vetor[3]},{vetor[4]},{vetor[5]}"
                                    
                                    arquivo4.write(strCluster+"\n")
                    #print("Centroides:")
                    #print(centroids)
                    #clustering.dictVitimas()

                    input(f"{self.NAME}: type [ENTER] to proc--eed")

                    #calcula a gravidade e cria o caminho
                    #print("delibera?")
                
                    #print("delibera")
                    #print(self.victims)
                    socorro.Iniciar(mergedmap)
                    
                    #self.resc.go_save_victims(self.map, self.victims)
                else:
                    print("chegou antes")
                    
                    

                    

                return False
            else:
                if self.flagCalculo == 0:
                    ##cria a pilha de retorno rapido
                    self.retornoRapido()
                    
                self.flagCalculo = 1
                self.come_back()
                return True      
                
    #vai retirando os items dessa pilha até achar alguem com vizinhos
    def recuarCaminho(self):
        
        if self.pilhaRetorno.size() > 0:
            dx, dy = self.pilhaRetorno.pop()
            variax = dx * -1
            variay = dy * -1

            for i in range(8):
                if self.vet[i] == (variax, variay):
                    return i
        else:
            self.flagTodo = 0


    #A ideia é criar uma pilha que vai conter os pontos vizitados mas vai buscar o menor caminho entre eles
    #o vetor caminho é uma copia da pilha já criada do professor, ela serve de base para encontrar os pontos 
    #que já foram visitados 
    def retornoRapido(self):
        cont = 0
        dx1 = self.x
        dy1 = self.y
        i = 0
        self.vetorDeRetorno = []
        tamanhodabusca = len(self.vetorDoCaminho)

        while i < tamanhodabusca:

            #vetor do caminho salva possicao anterior que foi dado
            (dx2, dy2) = self.vetorDoCaminho[i]

            if ((dx1 - dx2) <= 1) and ((dy1 - dy2) <= 1) and ((dx1 - dx2) >= -1) and ((dy1 - dy2) >= -1):
                
                self.vetorDeRetorno.append(self.vetorDoCaminho[i])

                dx1,dy1 = self.vetorDoCaminho[i]

                tamanhodabusca = i
                i=0
                cont +=1

            else:
                i+=1
        self.vetorDeRetorno.append((0,0))
 
