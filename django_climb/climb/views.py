from django.shortcuts import render, redirect
from django.http import JsonResponse
import numpy as np
import cv2
import base64
from .utils.visao import aplicar_mascara
from PIL import Image
import io
import json

def index(request):
    if request.method == "POST":
        imagem_file = request.FILES.get("imagem")
        coordenadas_json = request.POST.get("coordenadas")

        if not imagem_file or not coordenadas_json:
            return render(request, 'climb/index.html', {"erro": "Envie a imagem e selecione ao menos um ponto."})

        try:
            coordenadas = json.loads(coordenadas_json)
            coordenadas = [tuple(map(int, c)) for c in coordenadas]
        except Exception as e:
            return render(request, 'climb/index.html', {"erro": f"Coordenadas inválidas: {str(e)}"})

        try:
            img_pil = Image.open(imagem_file).convert("RGB")
            imagem_np = np.array(img_pil)
            imagem_bgr = cv2.cvtColor(imagem_np, cv2.COLOR_RGB2BGR)
        except Exception as e:
            return render(request, 'climb/index.html', {"erro": f"Erro ao processar imagem: {str(e)}"})

        # Tolerância padrão para primeira máscara
        tol_h, tol_s, tol_v = 10, 100, 100

        # Aplica máscara com tolerância padrão
        resultado = aplicar_mascara(imagem_bgr, coordenadas, tol_h, tol_s, tol_v)

        # Codifica imagem para base64
        _, buffer = cv2.imencode(".png", cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR))
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # Codifica imagem original para base64 para mostrar no ajustar.html
        _, buffer_orig = cv2.imencode(".png", cv2.cvtColor(imagem_bgr, cv2.COLOR_RGB2BGR))
        img_base64_orig = base64.b64encode(buffer_orig).decode("utf-8")

        # Passa os dados para página de ajuste
        context = {
            "imagem_processada": img_base64,
            "imagem_original": img_base64_orig,
            "coordenadas": json.dumps(coordenadas),
            "tol_h": tol_h,
            "tol_s": tol_s,
            "tol_v": tol_v,
        }

        return render(request, "climb/ajustar.html", context)

    return render(request, "climb/index.html")


def ajustar(request):
    if request.method == "POST" and request.headers.get('x-requested-with') == 'XMLHttpRequest':
        data = request.POST

        try:
            img_data = data.get("imagem")
            coordenadas_json = data.get("coordenadas")

            if not img_data or not coordenadas_json:
                return JsonResponse({"erro": "Imagem ou coordenadas não foram enviadas"})

            tol_h = int(data.get("tol_h", 10))
            tol_s = int(data.get("tol_s", 100))
            tol_v = int(data.get("tol_v", 100))

            # Decodifica imagem base64 para numpy
            if "," in img_data:
                header, base64data = img_data.split(",", 1)
            else:
                return JsonResponse({"erro": "Formato de imagem inválido (esperado base64 com header)."})

            img_bytes = base64.b64decode(base64data)
            img_np = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            imagem_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            coordenadas = json.loads(coordenadas_json)
            coordenadas = [tuple(map(int, c)) for c in coordenadas]

            resultado = aplicar_mascara(imagem_bgr, coordenadas, tol_h, tol_s, tol_v)

            _, buffer = cv2.imencode(".png", cv2.cvtColor(resultado, cv2.COLOR_RGB2BGR))
            img_base64 = base64.b64encode(buffer).decode("utf-8")

            return JsonResponse({"imagem": img_base64})

        except Exception as e:
            return JsonResponse({"erro": f"Erro ao processar: {str(e)}"})

    return JsonResponse({"erro": "Método inválido ou não ajax"})
