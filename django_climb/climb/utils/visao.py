import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def aplicar_mascara(imagem_bgr, coordenadas, tol_h, tol_s, tol_v):
    imagem_hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    mascara_final = np.zeros(imagem_hsv.shape[:2], dtype=np.uint8)

    for (x, y) in coordenadas:
        h, s, v = imagem_hsv[y, x]

        min_cor = np.array([max(h - tol_h, 0), max(s - tol_s, 0), max(v - tol_v, 0)], dtype=np.uint8)
        max_cor = np.array([min(h + tol_h, 179), min(s + tol_s, 255), min(v + tol_v, 255)], dtype=np.uint8)

        mascara = cv2.inRange(imagem_hsv, min_cor, max_cor)
        mascara_final = cv2.bitwise_or(mascara_final, mascara)

    resultado = np.zeros_like(imagem_rgb)
    resultado[mascara_final == 255] = imagem_rgb[mascara_final == 255]
    return resultado
