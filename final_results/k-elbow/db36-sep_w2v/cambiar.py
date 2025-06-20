from PIL import Image, ImageDraw, ImageFont
import pytesseract
import cv2

# Ruta manual a tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Cargar imagen
image = cv2.imread("db36-sep_w2v_11.png")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir a PIL para edición
img_pil = Image.fromarray(image_rgb)
draw = ImageDraw.Draw(img_pil)

# Fuente para números (más grande)
font_numbers = ImageFont.truetype("arial.ttf", 28)

# Detectar texto
data = pytesseract.image_to_data(image_rgb, output_type=pytesseract.Output.DICT)

for i in range(len(data['text'])):
    text = data['text'][i].strip()
    if text:
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

        # Verificar si es número (incluye decimales)
        if text.replace('.', '', 1).isdigit():
            # Agrandar números

            # Medir texto grande
            try:
                bbox = draw.textbbox((0, 0), text, font=font_numbers)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except AttributeError:
                text_w, text_h = draw.textsize(text, font=font_numbers)

            # Borrar original
            draw.rectangle([x, y, x + text_w, y + text_h], fill="white")
            # Escribir número más grande
            draw.text((x, y), text, font=font_numbers, fill="black")

        else:
            # Solo borrar textos que no son números
            draw.rectangle([x, y, x + w, y + h], fill="white")

# Guardar imagen modificada
img_pil.save("db36-sep_w2v_11_editada.png")
