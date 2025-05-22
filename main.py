import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import Slider
from collections import deque

def buildDetector(minArea = 25):
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 0
    params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = minArea
    params.filterByCircularity = False
    params.minCircularity = 0.1
    params.filterByConvexity = False
    params.minConvexity = 0.1
    params.filterByInertia = True
    params.minInertiaRatio = 0.05
    ver = (cv.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv.SimpleBlobDetector(params)
    else : 
        detector = cv.SimpleBlobDetector_create(params)

    return detector

def findHolds(img,detector = None):
    img = cv.medianBlur(img, 3)

    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    otsu, _ = cv.threshold(gray,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    edges = cv.Canny(img,otsu, otsu * 2, L2gradient = True)
    contours, _ = cv.findContours(edges,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
    hulls = map(cv.convexHull,contours)
    mask = np.zeros(gray.shape, np.uint8)
    cv.drawContours(mask, contours, -1, 255, 2)

    if detector == None:
        detector = buildDetector()

    keypoints = detector.detect(mask)
    return keypoints, hulls

def on_click(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        coordenadas.append((x, y))
        print(f"Coordenada registrada: ({x}, {y})")
        plt.close()  # fecha a janela após um clique

def on_click_regiao(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        coordenadas_regiao.append((x, y))
        print(f"Coordenada registrada: ({x}, {y})")
        if len(coordenadas_regiao) == 2:
            plt.close()

def on_click_agarras_iniciais(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        posicoes_agarras["Iniciais"].append((x, y))
        if len(posicoes_agarras["Iniciais"]) == 2:
            print(f"Agarras iniciais escolhidas: {posicoes_agarras['Iniciais']}")
            plt.close()

def on_click_agarras_finais(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        posicoes_agarras["Final"] = (x, y)
        print(f"Agarra final escolhida: {posicoes_agarras['Final']}")
        plt.close()

def atualizar(val):
    tol_h = int(slider_h.val)
    tol_s = int(slider_s.val)
    tol_v = int(slider_v.val)

    mascara_final = np.zeros(imagem_hsv.shape[:2], dtype=np.uint8)

    for (x, y) in coordenadas:
        h, s, v = imagem_hsv[y, x]
        h, s, v = int(h), int(s), int(v)

        min_h = np.clip(h - tol_h, 0, 179)
        max_h = np.clip(h + tol_h, 0, 179)
        min_s = np.clip(s - tol_s, 0, 255)
        max_s = np.clip(s + tol_s, 0, 255)
        min_v = np.clip(v - tol_v, 0, 255)
        max_v = np.clip(v + tol_v, 0, 255)

        min_cor = np.array([min_h, min_s, min_v], dtype=np.uint8)
        max_cor = np.array([max_h, max_s, max_v], dtype=np.uint8)

        mascara = cv.inRange(imagem_hsv, min_cor, max_cor)
        mascara_final = cv.bitwise_or(mascara_final, mascara)

    resultado = np.zeros_like(imagem_rgb)
    resultado[mascara_final == 255] = imagem_rgb[mascara_final == 255]
    img_plot.set_data(resultado)
    fig.canvas.draw_idle()

def gerar_mascara_binaria(imagem_hsv, coordenadas, tol_h, tol_s, tol_v):
    mascara_final = np.zeros(imagem_hsv.shape[:2], dtype=np.uint8)

    for (x, y) in coordenadas:
        h, s, v = imagem_hsv[y, x]
        h, s, v = int(h), int(s), int(v)

        min_h = np.clip(h - tol_h, 0, 179)
        max_h = np.clip(h + tol_h, 0, 179)
        min_s = np.clip(s - tol_s, 0, 255)
        max_s = np.clip(s + tol_s, 0, 255)
        min_v = np.clip(v - tol_v, 0, 255)
        max_v = np.clip(v + tol_v, 0, 255)

        min_cor = np.array([min_h, min_s, min_v], dtype=np.uint8)
        max_cor = np.array([max_h, max_s, max_v], dtype=np.uint8)

        mascara = cv.inRange(imagem_hsv, min_cor, max_cor)
        mascara_final = cv.bitwise_or(mascara_final, mascara)

    image_bin = cv.threshold(mascara_final, 127, 255, cv.THRESH_BINARY)[1]
    return image_bin

def bfs_segmentation(binary_image):
    visited = np.zeros_like(binary_image, dtype=bool)
    height, width = binary_image.shape
    components = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 255 and not visited[y, x]:
                queue = deque()
                queue.append((y, x))
                component = []
                visited[y, x] = True

                while queue:
                    cy, cx = queue.popleft()
                    component.append((cx, cy))
                    for dy, dx in directions:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if binary_image[ny, nx] == 255 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                queue.append((ny, nx))

                components.append(component)

    return components

def visualizar_componentes(binary_image, components):
    result = np.zeros((binary_image.shape[0], binary_image.shape[1], 3), dtype=np.uint8)
    cores = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
             (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for idx, comp in enumerate(components):
        cor = cores[idx % len(cores)]
        for x, y in comp:
            result[y, x] = cor

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Imagem Binária")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.title("Componentes Segmentados")
    plt.axis("off")
    plt.tight_layout()
    plt.show()






matplotlib.use('TkAgg')

coordenadas = []
coordenadas_regiao = []
posicoes_agarras = {"Iniciais": [], "Final": ()}
caminho_imagem = 'img/parede30.png'
imagem_bgr = cv.imread(caminho_imagem)

if imagem_bgr is None:
    print("Erro: imagem não encontrada.")
else:
    imagem_rgb = cv.cvtColor(imagem_bgr, cv.COLOR_BGR2RGB)
    imagem_hsv = cv.cvtColor(imagem_bgr, cv.COLOR_BGR2HSV)

    fig, ax = plt.subplots()
    ax.imshow(imagem_rgb)
    ax.set_title('Clique nos dois pontos da imagem para limitar a região da rota')
    cid = fig.canvas.mpl_connect('button_press_event', on_click_regiao)
    plt.show()

    print(f"Canto superior esquerdo: ({coordenadas_regiao[0][0]}, {coordenadas_regiao[0][1]})")
    print(f"Canto inferior direito: ({coordenadas_regiao[1][0]}, {coordenadas_regiao[1][1]})")

    # Limitar a região da imagem
    x1, y1 = coordenadas_regiao[0]
    x2, y2 = coordenadas_regiao[1]

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    imagem_rgb = imagem_rgb[y1:y2, x1:x2]
    imagem_hsv = imagem_hsv[y1:y2, x1:x2]

    fig, ax = plt.subplots()
    ax.imshow(imagem_rgb)
    ax.set_title('Clique num ponto da imagem para registrar a cor da rota')
    cid = fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

    print("\nCores HSV selecionadas:")
    for (x, y) in coordenadas:
        cor_hsv = imagem_hsv[y, x]
        print(f"Coordenada ({x}, {y}): Cor HSV: {cor_hsv}")

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.35)
    img_plot = ax.imshow(imagem_rgb)
    ax.axis('off')
    ax.set_title('Ajuste HSV - Regiões Detectadas')

    ax_h = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_s = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_v = plt.axes([0.25, 0.15, 0.65, 0.03])

    slider_h = Slider(ax_h, 'Tolerância H', 0, 50, valinit=4, valstep=1)
    slider_s = Slider(ax_s, 'Tolerância S', 0, 100, valinit=100, valstep=1)
    slider_v = Slider(ax_v, 'Tolerância V', 0, 100, valinit=100, valstep=1)

    slider_h.on_changed(atualizar)
    slider_s.on_changed(atualizar)
    slider_v.on_changed(atualizar)

    atualizar(None)
    plt.show()

    kernel = np.ones((5, 5), np.uint8)
    image_bin = gerar_mascara_binaria(imagem_hsv, coordenadas, int(slider_h.val), int(slider_s.val), int(slider_v.val))
    image_bin = cv.erode(image_bin, kernel, iterations=2) # só erosão
    components = bfs_segmentation(image_bin)
    visualizar_componentes(image_bin, components)

    fig, ax = plt.subplots()
    ax.imshow(imagem_rgb)
    ax.set_title('Clique em duas agarras da imagem para escolher as agarras iniciais')
    cid = fig.canvas.mpl_connect('button_press_event', on_click_agarras_iniciais)
    plt.show()

    print("\nAgarras iniciais selecionadas:")
    for (x, y) in posicoes_agarras["Iniciais"]:
        print(f"Coordenada: ({x}, {y})")
    print()

    agarras_selecionadas = {"Iniciais": [], "Final": ()}

    for i in range(len(posicoes_agarras["Iniciais"])):
        for agarra in components:
            if posicoes_agarras["Iniciais"][i] in agarra:
                idx = agarra.index(posicoes_agarras["Iniciais"][i])
                agarras_selecionadas["Iniciais"].append((idx, agarra))
                print(f"Agarra {i+1} encontrada na componente {components.index(agarra)+1}")
                break

    if len(agarras_selecionadas["Iniciais"]) != 2:
        print("Erro: número de agarras iniciais selecionadas não é igual a 2.")
        exit()

    # print("\nAgarras iniciais atualizadas:")
    # print(f"Agarras iniciais: {agarras_selecionadas['Iniciais']}")
    # for (x, y) in agarras_selecionadas["Iniciais"]:
        # print(f"Coordenada: ({x}, {y})")

    fig, ax = plt.subplots()
    ax.imshow(imagem_rgb)
    ax.set_title('Clique na agarra final da imagem para escolher a agarra final')
    cid = fig.canvas.mpl_connect('button_press_event', on_click_agarras_finais)
    plt.show()

    print("\nAgarra final selecionada:")
    print(f"Coordenada: {posicoes_agarras['Final']}")

    for agarra in components:
        if posicoes_agarras["Final"] in agarra:
            idx = agarra.index(posicoes_agarras["Final"])
            agarras_selecionadas["Final"] = (idx, agarra)
            print(f"Agarra final encontrada na componente {components.index(agarra)+1}")
            break

    if agarras_selecionadas["Final"] == ():
        print("Erro: agarra final não encontrada nas componentes.")
        exit()

    imagem_marcada = imagem_rgb.copy()

    for i in range(len(agarras_selecionadas["Iniciais"])):
        for (x, y) in agarras_selecionadas["Iniciais"][i][1]:
            imagem_marcada[y, x] = [255, 255, 255]

    for (x, y) in agarras_selecionadas["Final"][1]:
        imagem_marcada[y, x] = [0, 0, 0]

    fig, ax = plt.subplots()
    ax.imshow(imagem_marcada)
    ax.set_title('Agarras iniciais (branco) e agarra final (preto)')
    plt.show()