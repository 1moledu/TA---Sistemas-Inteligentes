import heapq

def heuristica(coord_atual, coord_destino):
    """Calcula a distância de Manhattan como heurística."""
    return abs(coord_atual[0] - coord_destino[0]) + abs(coord_atual[1] - coord_destino[1])

def calcular_distancia(coord1, coord2):
    """Calcula a distância Euclidiana entre dois pontos."""
    return ((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) ** 0.5

def encontrar_caminho_a_star(map_obj, inicio, fim):
    """Encontra o caminho entre dois pontos usando A*."""
    open_set = []
    heapq.heappush(open_set, (0, inicio))
    
    g_score = {inicio: 0}
    f_score = {inicio: heuristica(inicio, fim)}
    came_from = {}
    
    while open_set:
        _, atual = heapq.heappop(open_set)
        
        if atual == fim:
            return reconstruir_caminho(came_from, atual)
        
        # Explora os vizinhos válidos
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]:
            vizinho = (atual[0] + dx, atual[1] + dy)
            
            if map_obj.in_map(vizinho):  # Apenas considere vizinhos dentro do mapa
                tent_g_score = g_score[atual] + calcular_distancia(atual, vizinho)
                
                if vizinho not in g_score or tent_g_score < g_score[vizinho]:
                    came_from[vizinho] = atual
                    g_score[vizinho] = tent_g_score
                    f_score[vizinho] = tent_g_score + heuristica(vizinho, fim)
                    heapq.heappush(open_set, (f_score[vizinho], vizinho))
    
    return []  # Retorna caminho vazio se não encontrar

def reconstruir_caminho(came_from, atual):
    """Reconstrói o caminho a partir do dicionário came_from."""
    caminho = [atual]
    while atual in came_from:
        atual = came_from[atual]
        caminho.append(atual)
    caminho.reverse()
    return caminho

def a_star_sequencia(map_obj, sequencia):
    """Encontra o melhor caminho passando por uma sequência de pontos na ordem fornecida."""

    # Extrair apenas coordenadas (x, y) e manter o valor para uso posterior
    coordenadas = [(x, y) for (x, y, _) in sequencia]
    valores = {(x, y): valor for (x, y, valor) in sequencia}

    # Adicionar o ponto inicial e final
    coordenadas = [(0, 0)] + coordenadas + [(0, 0)]
    
    caminho_final = []
    posicao_atual = coordenadas[0]
    
    for i in range(1, len(coordenadas)):
        destino = coordenadas[i]
        caminho = encontrar_caminho_a_star(map_obj, posicao_atual, destino)
        if not caminho:
            raise ValueError(f"Não foi possível encontrar um caminho entre {posicao_atual} e {destino}.")
        
        # Adicionar o caminho ao resultado, incluindo valor 1 para pontos da sequência original
        for ponto in caminho[:-1]:  # Adiciona todos os pontos, exceto o último para evitar duplicação
            caminho_final.append((*ponto, 1 if ponto in valores else 0))
        
        posicao_atual = destino
    
    # Adicionar o último ponto da sequência com o valor correspondente
    caminho_final.append((0, 0, 0))
    
    return caminho_final
