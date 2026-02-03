import pygame
import random
import sys
import time
import copy
import csv
from variaveis import *
from rl_agent import load_weights, choose_action
import game_logic


pygame.init()

try:
    AI_WEIGHTS = load_weights("weights.json")
except Exception:
    AI_WEIGHTS = [0.0, 0.0, 0.0, 0.0]
    
# Estado da UI/IA
LAST_MOVE_PC = None
LAST_MOVE_PLAYER = None
LAST_AI_EPSILON = 0.05
LAST_AI_FEATURES = None  # features do RL
TRAINING_SUMMARY = None  # info do stats.csv (último checkpoint)

# tentar ler o último checkpoint do treino (stats.csv)
try:
    with open("stats.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            TRAINING_SUMMARY = rows[-1]
except Exception:
    TRAINING_SUMMARY = None


def main(): # Inicializa a tela do jogo, carrega imagens de fundo, e inicia o loop principal do jogo.
    global MAIN_CLOCK, EXIBIR_JANELA, FONTE, BIGFONTE, BGIMAGEM

    EXIBIR_JANELA = pygame.display.set_mode((WIN_LARGURA, WIN_ALTURA))
    pygame.display.set_caption('Reversi')
    FONTE = pygame.font.Font('freesansbold.ttf', 20)
    BIGFONTE = pygame.font.Font('freesansbold.ttf', 30)
    MAIN_CLOCK = pygame.time.Clock()

    # Configure a imagem de fundo. 
    Imag_quadro = pygame.image.load('arquivos/Fundo_tab_3.jpg').convert_alpha()
    # scale() para ajustar a imagem. 
    Imag_quadro = pygame.transform.scale(Imag_quadro, (LARGURA_QUAD * TAMANHO_ESPACO, ALTURA_QUAD * TAMANHO_ESPACO))
    Imag_quadroRect = Imag_quadro.get_rect()
    Imag_quadroRect.topleft = (XMARGEM, YMARGEM)
    BGIMAGEM = pygame.image.load('arquivos/Fundo 1.jpg')
    BGIMAGEM = pygame.transform.scale(BGIMAGEM, (WIN_LARGURA, WIN_ALTURA))
    BGIMAGEM.blit(Imag_quadro, Imag_quadroRect)
      
    menu_inicial()
    # Loop que mantenha a janela aberta
    while True:
        if executarGame() == False:
            break


pygame.mixer.init() # Inicialize o Pygame e o mixer
SOM_JOGO = pygame.mixer.Sound(SOM_JOGO) # Carregue o arquivo de áudio
pygame.time.Clock().tick(FPS)
SOM_JOGO.play() # Reproduza o áudio
pygame.mixer.music.set_volume(0.7)
# Mantenha o programa em execução para permitir a reprodução


def menu_inicial():
    # Dimensões do painel
    panel_w, panel_h = 700, 420
    panel_x = WIN_LARGURA // 2 - panel_w // 2
    panel_y = WIN_ALTURA // 2 - panel_h // 2

    # Textos
    titleSurf = BIGFONTE.render("Reversi (Othello) + RL Agent", True, CORTEXTO)
    titleRect = titleSurf.get_rect()
    titleRect.center = (WIN_LARGURA // 2, panel_y + 70)

    info1 = FONTE.render(
        "Modelo: Aprendizagem por Reforço (linear) | pesos em weights.json",
        True, CORTEXTO
    )
    info2 = FONTE.render(
        "Dica: execute train.py para treinar e melhorar o agente",
        True, CORTEXTO
    )

    # Botões
    playSurf = BIGFONTE.render("Jogar", True, CORTEXTO, TEXTOBGCOR1)
    playRect = playSurf.get_rect()
    playRect.center = (WIN_LARGURA // 2, panel_y + 240)

    exitSurf = BIGFONTE.render("Sair", True, CORTEXTO, TEXTOBGCOR1)
    exitRect = exitSurf.get_rect()
    exitRect.center = (WIN_LARGURA // 2, panel_y + 310)

    while True:
        EXIBIR_JANELA.blit(BGIMAGEM, BGIMAGEM.get_rect())

        # Painel escuro para legibilidade
        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 150))
        EXIBIR_JANELA.blit(panel, (panel_x, panel_y))

        # Desenhar textos
        EXIBIR_JANELA.blit(titleSurf, titleRect)
        EXIBIR_JANELA.blit(
            info1,
            (WIN_LARGURA // 2 - info1.get_width() // 2, panel_y + 120)
        )
        EXIBIR_JANELA.blit(
            info2,
            (WIN_LARGURA // 2 - info2.get_width() // 2, panel_y + 145)
        )

        # Botões
        EXIBIR_JANELA.blit(playSurf, playRect)
        EXIBIR_JANELA.blit(exitSurf, exitRect)

        # Eventos
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYUP and event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONUP:
                mx, my = event.pos
                if playRect.collidepoint((mx, my)):
                    return "play"
                if exitRect.collidepoint((mx, my)):
                    pygame.quit()
                    sys.exit()

        pygame.display.update()
        MAIN_CLOCK.tick(FPS)


def executarGame(): # 
    # Joga um Unico jogo cada vez que esta função é chamada. 

    #Renicie o tabuleiro e o jogo. 
    placaPrincipal = getNovoQuadro()
    redefinirPlaca(placaPrincipal)
    mostrarDIcas = False
    turno = random.choice(['Pc', 'Jogador'])

    # Desenhe o tabuleiro inicial e pergunte ao jogador qual cor ele deseja.
    designTabu(placaPrincipal)
    peca_jogador, peca_Pc = solicPecaJogador()
    
    # Faça os objetos Surface e Rest para os botões "Novo jogo", "Sair" e "Dicas"
    # Botões do topo
    novoJogoSurf = FONTE.render('Novo Jogo', True, CORTEXTO, TEXTOBGCOR2)
    novoJogoRect = novoJogoSurf.get_rect()
    novoJogoRect.topleft = (20, 10)

    dicasSurf = FONTE.render('Dicas', True, CORTEXTO, TEXTOBGCOR2)
    dicasRect = dicasSurf.get_rect()
    dicasRect.topleft = (160, 10)

    sairSurf = FONTE.render('Sair', True, CORTEXTO, TEXTOBGCOR2)
    sairRect = sairSurf.get_rect()
    sairRect.topright = (WIN_LARGURA - 20, 10)



    while True: # Loop principal do jogo
        # Fica repetindo os turnos do jogador e do computador. 
        if turno == 'Jogador': # Vez do Jogador
            if getMoveValidos(placaPrincipal, peca_jogador) == []:
                # se for a vez do jogador, mas ele não conseguese mover e ntm termina o jogo. 
                break
            moverxy = None
            while moverxy == None:
                # Contunue fazendo loop até que o jogador usar clique em um espaço válido. 
                if mostrarDIcas:
                    espacoValido = getTabuMoveValidos(placaPrincipal, peca_jogador)
                else:
                    espacoValido = placaPrincipal
                verificarSaida()
                for event in pygame.event.get(): # Um loop de manipulação eventos
                    if event.type == pygame.MOUSEBUTTONUP:
                        # Eventos de clique do mouse
                        mouseX, mouseY = event.pos
                        if sairRect.collidepoint((mouseX, mouseY)):
                            pygame.quit()
                            sys.exit()
                        
                        if novoJogoRect.collidepoint((mouseX, mouseY)):
                            return True
                        elif dicasRect.collidepoint((mouseX, mouseY)):
                            mostrarDIcas = not mostrarDIcas
                        # moverxy é difinido como uma coordenada XY de tuplas dois itens ou valor nenhum
                        moverxy = getEspacoClicado(mouseX, mouseY)
                        if moverxy != None and not isMoveValido(placaPrincipal, peca_jogador, moverxy[0], moverxy[1]):
                            moverxy = None
                              
                # Desenhe o tabuleiro do jogo. 
                designTabu(espacoValido)
                desenharInfo(espacoValido, peca_jogador, peca_Pc, turno)

                # Desenhe os botões "Novo Jogo" e "Dicas". 
                EXIBIR_JANELA.blit(novoJogoSurf, novoJogoRect)
                EXIBIR_JANELA.blit(dicasSurf, dicasRect)
                EXIBIR_JANELA.blit(sairSurf, sairRect)
                
                MAIN_CLOCK.tick(FPS)
                pygame.display.update()

            # Faça o movimento e termina o turno. 
            fazerMover(placaPrincipal, peca_jogador, moverxy[0], moverxy[1], True)
            LAST_MOVE_PLAYER = (moverxy[0], moverxy[1])
            if getMoveValidos(placaPrincipal, peca_Pc) != []:
                    # Defina apenas para o turno do Pc se ele puder fazer um move. 
                    turno = 'PC'
        else:
            #Vez do computador
            if getMoveValidos(placaPrincipal, peca_Pc) == []:
                # Se foi definido para ser a vez do Pc, mas não conseguir se mover e então encerram o jogo.
                break

            # Desenhe o tabuleiro.
            designTabu(placaPrincipal)
            desenharInfo(placaPrincipal, peca_jogador, peca_Pc, turno)

            # Desenhe os botões "Novo Jogo" e "Dicas".
            EXIBIR_JANELA.blit(novoJogoSurf, novoJogoRect)
            EXIBIR_JANELA.blit(dicasSurf, dicasRect)

            # O Pc está pousado em leitura.
            pausaUntil = time.time() + random.randint(5, 15) * 0.1
            while time.time() < pausaUntil:
                pygame.display.update()

            # Faça o movimento e termine o turno.
            x, y = getMovePc(placaPrincipal, peca_Pc)
            LAST_MOVE_PC = (x, y)
            if x is None or y is None:
                break
            fazerMover(placaPrincipal, peca_Pc, x, y, True)
            # O jogador fazer um movimento se o puder fazer.
            turno = 'Jogador'

    # Exibe a pontuação final.
    designTabu(placaPrincipal)
    ponto = getPontoTabu(placaPrincipal)

    # Determine o texto da mensagem a ser exibida.
    if ponto[peca_jogador] > ponto[peca_Pc]:
        texto = 'Você venceu o computador por %s pontos! Parabéns!' % (ponto[peca_jogador] - ponto[peca_Pc])
    
    elif ponto[peca_jogador] < ponto[peca_Pc]:
        texto = 'Você perdeu. O computador venceu você por %s pontos.' %  (ponto[peca_Pc] - ponto[peca_jogador])
    
    else:
        texto = 'O jogo empatou!'

    textoSurf = FONTE.render(texto, True, CORTEXTO, TEXTOBGCOR1)
    textoRect = textoSurf.get_rect()
    textoRect.center = (int(WIN_LARGURA / 2), int(WIN_ALTURA / 2))
    EXIBIR_JANELA.blit(textoSurf, textoRect)

    # Exiba a mensagem "Jogar de novo?" texto com botões Sim e Não.
    texto2Surf = BIGFONTE.render('Jogar de novo?', True, CORTEXTO, TEXTOBGCOR1)
    texto2Rect = texto2Surf.get_rect()
    texto2Rect.center = (int(WIN_LARGURA / 2), int(WIN_ALTURA / 2) + 50)

    # Faça o botão "Sim".
    simSurf = BIGFONTE.render('Sim', True, CORTEXTO, TEXTOBGCOR1)
    simRect = simSurf.get_rect()
    simRect.center = (int(WIN_LARGURA / 2) - 60, int(WIN_ALTURA / 2) + 90)
    
    # Faça o botão "Não"
    semSurf = BIGFONTE.render('Não', True, CORTEXTO, TEXTOBGCOR1)
    semRect = semSurf.get_rect()
    semRect.center = (int(WIN_LARGURA / 2) + 60, int(WIN_ALTURA / 2) + 90)

    while True:
        # Processe eventos até que o usuário clique em Sim ou Não
        verificarSaida()
        for event in pygame.event.get(): # loop de manipulação de eventos
            if event.type == pygame.MOUSEBUTTONUP:
                mouseX, mouseY = event.pos
                if simRect.collidepoint((mouseX, mouseY)):
                    return True
                elif semRect.collidepoint((mouseX, mouseY)):
                    return False
        EXIBIR_JANELA.blit(textoSurf, textoRect)
        EXIBIR_JANELA.blit(texto2Surf, texto2Rect)
        EXIBIR_JANELA.blit(simSurf, simRect)
        EXIBIR_JANELA.blit(semSurf, semRect)
        pygame.display.update()
        MAIN_CLOCK.tick(FPS)


def coordPixelQuadro(x, y):
    return XMARGEM + x * TAMANHO_ESPACO + int(TAMANHO_ESPACO / 2), YMARGEM + y * TAMANHO_ESPACO + int(TAMANHO_ESPACO / 2)


def moverPecasTabu(virarPecas, corPecas, adicionarBloco):
    # Desenhe a peça adicional que acabou de ser colocada. 
    # (Caso contrário, teríamos que redesenhar completamente o quadro e as informações do quadro). 
    if corPecas == PECA_BRANCA:
        adicionarCorPecas = WHITE
    else:
        adicionarCorPecas = BLACK
    adicionarPecaX, adicionarPecaY = coordPixelQuadro(adicionarBloco[0], adicionarBloco[1])
    pygame.draw.circle(EXIBIR_JANELA, adicionarCorPecas, (adicionarPecaX, adicionarPecaY), int(TAMANHO_ESPACO / 2) - 4)
    pygame.display.update()

    for ValoresRGB in range(0, 255, int(ANIMACAO_SPEED * 2.55)):
         if ValoresRGB > 255:
            ValoresRGB = 255
         elif ValoresRGB < 0:
            ValoresRGB = 0

         if corPecas == PECA_BRANCA:
            cor = tuple([ValoresRGB] * 3) # ValoresRGB ​​vai de 0 a 255
         elif corPecas == PECA_PRETA:
            cor = tuple([255 - ValoresRGB] * 3) # ValoresRGB ​​vai de 255 a 0

         for x, y in virarPecas:
            centroX, centroY = coordPixelQuadro(x, y)
            pygame.draw.circle(EXIBIR_JANELA, cor, (centroX, centroY), int(TAMANHO_ESPACO / 2) - 4)
         pygame.display.update()
         MAIN_CLOCK.tick(FPS)
         verificarSaida()


def designTabu(quadro):
    # Desenhe o fundo do tabuleiro.
    EXIBIR_JANELA.blit(BGIMAGEM, BGIMAGEM.get_rect())

    # Desenhe linhas de grade do quadro.
    for x in range(LARGURA_QUAD + 1):
        # Desenhe as linhas horizontais.
        iniciarX = (x * TAMANHO_ESPACO) + XMARGEM
        iniciarY = YMARGEM
        fimX = (x * TAMANHO_ESPACO) + XMARGEM
        fimY = YMARGEM + (ALTURA_QUAD * TAMANHO_ESPACO)
        pygame.draw.line(EXIBIR_JANELA, CORLINHAMATRIZ, (iniciarX, iniciarY), (fimX, fimY))
    for y in range(ALTURA_QUAD + 1):
        # Desenhe as linhas verticais.
        iniciarX = XMARGEM
        iniciarY = (y * TAMANHO_ESPACO) + YMARGEM
        fimX = XMARGEM + (LARGURA_QUAD * TAMANHO_ESPACO)
        fimY = (y * TAMANHO_ESPACO) + YMARGEM
        pygame.draw.line(EXIBIR_JANELA, CORLINHAMATRIZ, (iniciarX, iniciarY), (fimX, fimY))

    # Desenhe os ladrilhos pretos e brancos ou os pontos de dicas.
    for x in range(LARGURA_QUAD):
        for y in range(ALTURA_QUAD):
            centroX, centroY = coordPixelQuadro(x, y)
            if quadro[x][y] == PECA_BRANCA or quadro[x][y] == PECA_PRETA:
                if quadro[x][y] == PECA_BRANCA:
                    corPecas = WHITE
                else:
                    corPecas = BLACK
                pygame.draw.circle(EXIBIR_JANELA, corPecas, (centroX, centroY), int(TAMANHO_ESPACO / 2) - 4)
            if quadro[x][y] == TELHA_DICAS:
                pygame.draw.rect(EXIBIR_JANELA, CORDADICA, (centroX - 4, centroY - 4, 8, 8))

    # Highlight da ultima jogada
    H_PLAYER = (0, 120, 255)
    H_PC = (255, 200, 0)
    
    def _draw_highlight(move, color):
        if move is None:
            return
        x, y = move
        # retângulo da célula
        left = XMARGEM + x * TAMANHO_ESPACO
        top = YMARGEM + y * TAMANHO_ESPACO
        pygame.draw.rect(EXIBIR_JANELA, color, (left+2, top+2, TAMANHO_ESPACO-4, TAMANHO_ESPACO-4), 3)

    _draw_highlight(LAST_MOVE_PLAYER, H_PLAYER)
    _draw_highlight(LAST_MOVE_PC, H_PC)
    
    
def getEspacoClicado(mouseX, mouseY):
    # Retorna uma tuplas de dois inteiros das coordenadas do espaço do tabuleiro onde o mouse foi clicado. (Ou retorna None sem nenhum espaço.)
    for x in range(LARGURA_QUAD):
        for y in range(ALTURA_QUAD):
           if mouseX > x * TAMANHO_ESPACO + XMARGEM and \
            mouseX < (x + 1) * TAMANHO_ESPACO + XMARGEM and \
            mouseY > y * TAMANHO_ESPACO + YMARGEM and \
            mouseY < (y + 1) * TAMANHO_ESPACO + YMARGEM:
                return (x, y)
    return None


def desenharInfo(quadro, peca_jogador, peca_Pc, turno):
    # Pontuacao e turno
    ponto = getPontoTabu(quadro)
    texto_base = "Jogador: %s  /  PC: %s   Vez do: %s" % (str(ponto[peca_jogador]), str(ponto[peca_Pc]), turno.title())
    pontourf = FONTE.render(texto_base, True, CORTEXTO)
    pontoRect = pontourf.get_rect()
    pontoRect.bottomleft = (40, WIN_ALTURA -8)
    EXIBIR_JANELA.blit(pontourf, pontoRect)

    # Painel IA
    y0 = 45
    x0 = 20
    
    # Fundo do painel (semi-transparente) para legibilidade
    panel_w = WIN_LARGURA - 40
    panel_h = 70
    panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 140))  # preto com alpha
    EXIBIR_JANELA.blit(panel, (x0 - 10, y0 - 5))

    model_line = f"IA: RL (linear) | epsilon={LAST_AI_EPSILON:.2f} | weights={ [round(w,2) for w in AI_WEIGHTS] }"
    surf = FONTE.render(model_line, True, CORTEXTO)
    EXIBIR_JANELA.blit(surf, (x0, y0))
    
    # Última jogada do PC
    if LAST_MOVE_PC is not None:
        surf2 = FONTE.render(f"Última jogada PC: {LAST_MOVE_PC}", True, CORTEXTO)
        EXIBIR_JANELA.blit(surf2, (x0, y0 + 22))

    # Info do treino (se existir stats.csv)
    if TRAINING_SUMMARY is not None:
        try:
            tg = TRAINING_SUMMARY.get("trained_games", "?")
            wr_rand = float(TRAINING_SUMMARY.get("rand_win_rate", "0"))
            wr_greedy = float(TRAINING_SUMMARY.get("greedy_win_rate", "0"))
            surf3 = FONTE.render(
                f"Treino: {tg} jogos | win_rate vs Random={wr_rand:.2f} | vs Greedy={wr_greedy:.2f}",
                True, CORTEXTO
            )
            EXIBIR_JANELA.blit(surf3, (x0, y0 + 44))
        except Exception:
            pass

    
    
def redefinirPlaca(quadro):
    # Anula o tabuleiro em que foi passado e configura as peças iniciais.
    for x in range(LARGURA_QUAD):
        for y in range(ALTURA_QUAD):
            quadro[x][y] = ESPACO_VAZIO

    # Adicione peças iniciais ao centro
    quadro[3][3] = PECA_BRANCA
    quadro[3][4] = PECA_PRETA
    quadro[4][3] = PECA_PRETA
    quadro[4][4] = PECA_BRANCA


def getNovoQuadro():
    # Cria uma estrutura de dados de tabuleiro totalmente nova e vazia.
    quadro = []
    for i in range(LARGURA_QUAD):
        quadro.append([ESPACO_VAZIO] * ALTURA_QUAD)

    return quadro


def isMoveValido(quadro, telha, Xinicial, Yinicial):
    # Retorna False se a jogada do jogador for inválida. Se for válido move, retorna uma lista de espaços das peças capturadas.
    if quadro[Xinicial][Yinicial] != ESPACO_VAZIO or not quadroAtivo(Xinicial, Yinicial):
        return False

    quadro[Xinicial][Yinicial] = telha # define temporariamente o ladrilho no tabuleiro.
    if telha == PECA_BRANCA:
        outraTelha = PECA_PRETA
    else:
        outraTelha = PECA_BRANCA

    virarPecas = []
    # verifique cada uma das 8 direções:
    for direcaoX, direcaoY in [[0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
        x, y = Xinicial, Yinicial
        x += direcaoX
        y += direcaoY
        if quadroAtivo(x, y) and quadro[x][y] == outraTelha:
            # A peça pertence ao outro jogador próximo à nossa peça.
            x += direcaoX
            y += direcaoY
            if not quadroAtivo(x, y):
                continue
            while quadro[x][y] == outraTelha:
                x += direcaoX
                y += direcaoY
                if not quadroAtivo(x, y):
                    break # break out do loop while, continuar no loop for
            if not quadroAtivo(x, y):
                continue
            if quadro[x][y] == telha:
                #Há peças para virar. Vá no sentido inverso até chegarmos ao espaço original, anotando todos as peças ao longo do caminho.
                while True:
                    x -= direcaoX
                    y -= direcaoY
                    if x == Xinicial and y == Yinicial:
                        break
                    virarPecas.append([x, y])

    quadro[Xinicial][Yinicial] = ESPACO_VAZIO # Torna o espaço vazio
    if len(virarPecas) == 0: # Se nenhuma peça for virada, este movimento é inválido
        return False
    return virarPecas


def quadroAtivo(x, y):
    # Retorna True se as coordenadas estiverem localizadas no quadro.
    return x >= 0 and x < LARGURA_QUAD and y >= 0 and y < ALTURA_QUAD


def getTabuMoveValidos(quadro, telha):
    # Retorna um novo tabuleiro com marcações de dicas.
    dupeQuadro = copy.deepcopy(quadro)

    for x, y in getMoveValidos(dupeQuadro, telha):
        dupeQuadro[x][y] = TELHA_DICAS
    return dupeQuadro


def getMoveValidos(quadro, telha):
    # Retorna uma lista de tuplass (x,y) de todos os movimentos válidos.
    MoveValidos = []

    for x in range(LARGURA_QUAD):
        for y in range(ALTURA_QUAD):
            if isMoveValido(quadro, telha, x, y) != False:
                MoveValidos.append((x, y))
    return MoveValidos


def getPontoTabu(quadro):
    # Determine a pontuação contando as peças.
    pontoBr = 0
    pontoPr = 0
    for x in range(LARGURA_QUAD):
        for y in range(ALTURA_QUAD):
            if quadro[x][y] == PECA_BRANCA:
                pontoBr += 1
            if quadro[x][y] == PECA_PRETA:
                pontoPr += 1
    return {PECA_BRANCA:pontoBr, PECA_PRETA:pontoPr}


def solicPecaJogador():
    #Desenha o texto e trata os eventos de clique do mouse para permitir o jogador escolhe a cor que deseja ser.
    #crie o texto. 
    textoSurf = FONTE.render('Você quer ser branco ou preto?', True, CORTEXTO, TEXTOBGCOR1)
    textoRect = textoSurf.get_rect()
    textoRect.center = (int(WIN_LARGURA / 2), int(WIN_ALTURA / 2))

    brSurf = BIGFONTE.render('Branco', True, CORTEXTO, TEXTOBGCOR1)
    BrRect = brSurf.get_rect()
    BrRect.center = (int(WIN_LARGURA / 2) - 60, int(WIN_ALTURA / 2) + 40)

    PrSurf = BIGFONTE.render('Preto', True, CORTEXTO, TEXTOBGCOR1)
    PrRect = PrSurf.get_rect()
    PrRect.center = (int(WIN_LARGURA / 2) + 60, int(WIN_ALTURA / 2) + 40)

    while True:
        # Vai fazer o loop até que o jogador clique em uma cor.
        verificarSaida()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP:
                mouseX, mouseY = event.pos
                if BrRect.collidepoint((mouseX, mouseY)):
                    return [PECA_BRANCA, PECA_PRETA]
                elif PrRect.collidepoint((mouseX, mouseY)):
                    return [PECA_PRETA, PECA_BRANCA]

        # Desenhar a tela
        EXIBIR_JANELA.blit(textoSurf, textoRect)
        EXIBIR_JANELA.blit(brSurf, BrRect)
        EXIBIR_JANELA.blit(PrSurf, PrRect)
        pygame.display.update()
        MAIN_CLOCK.tick(FPS)


def fazerMover(quadro, telha, Xinicial, Yinicial, realMove=False):
    # Coloque a peça no tabuleiro em x, y e vire as peças
    # Retorna False se um movimento for inválido, True se for válido.
    virarPecas = isMoveValido(quadro, telha, Xinicial, Yinicial)
    
    if virarPecas == False:
        return False
    
    quadro[Xinicial][Yinicial] = telha # Coloca a peça do jogador no tabuleiro na posição determinada.

    if realMove:
        moverPecasTabu(virarPecas, telha, (Xinicial, Yinicial))

    for x, y in virarPecas:
        quadro[x][y] = telha # Esta linha coloca a peça do jogador no tabuleiro nas posições invertidas.
    return True


def posicaoCanto(x, y):
    # Retorna True se a posição estiver em um dos quatro cantos.
    return (x == 0 and y == 0) or (x == LARGURA_QUAD-1 and y == 0) or \
        (x == 0 and y == ALTURA_QUAD-1) or (x == LARGURA_QUAD-1 and y == ALTURA_QUAD-1)


def converter_tabuleiro_para_rl(quadro, peca_Pc):
    rl_board = [[game_logic.EMPTY for _ in range(8)] for _ in range(8)]

    for x in range(LARGURA_QUAD):
        for y in range(ALTURA_QUAD):
            if quadro[x][y] == ESPACO_VAZIO:
                rl_board[x][y] = game_logic.EMPTY
            elif quadro[x][y] == peca_Pc:
                rl_board[x][y] = game_logic.BLACK
            else:
                rl_board[x][y] = game_logic.WHITE

    return rl_board


def getMovePc(quadro, peca_Pc):
    if not getMoveValidos(quadro, peca_Pc):
        return [None, None]

    rl_board = converter_tabuleiro_para_rl(quadro, peca_Pc)

    epsilon = 0.05

    move, _ = choose_action(
        rl_board,
        game_logic.BLACK,
        AI_WEIGHTS,
        epsilon
    )

    if move is None:
        return [None, None]

    x, y = move
    return [int(x), int(y)]
    

def verificarSaida():
    for event in pygame.event.get((pygame.QUIT, pygame.KEYUP)):
        if event.type == pygame.QUIT or (event.type == pygame.KEYUP and 
                                         event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()


if __name__ == '__main__':
    main()