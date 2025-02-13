from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import tempfile

# Inicializar o app Flask
app = Flask(__name__)

# Carregar o modelo salvo
MODEL_PATH = "dog_breed_classifier_EfficientNetB4.h5" 
model = load_model(MODEL_PATH)

# Definir parâmetros de imagem
target_size = (380, 380)  # Tamanho usado no treinamento
class_labels = {
    0: "Affenpinscher",
    1: "Galgo Afegão",
    2: "Cachorro Selvagem Africano",
    3: "Airedale",
    4: "Staffordshire Terrier Americano",
    5: "Appenzeller",
    6: "Terrier Australiano",
    7: "Basenji",
    8: "Basset",
    9: "Beagle",
    10: "Terrier Bedlington",
    11: "Cão da Montanha Bernês",
    12: "Black-and-Tan Coonhound",
    13: "Spaniel Blenheim",
    14: "Bloodhound",
    15: "Bluetick",
    16: "Collie de Fronteira",
    17: "Terrier de Fronteira",
    18: "Borzoi",
    19: "Boston Bull",
    20: "Boiadeiro de Flandres",
    21: "Boxer",
    22: "Griffon de Bruxelas",
    23: "Briard",
    24: "Spaniel Bretão",
    25: "Bullmastiff",
    26: "Cairn",
    27: "Cardigan",
    28: "Retriever da Baía de Chesapeake",
    29: "Chihuahua",
    30: "Chow-Chow",
    31: "Clumber",
    32: "Spaniel Cocker",
    33: "Collie",
    34: "Retriever de Pelo Encaracolado",
    35: "Dandie Dinmont",
    36: "Dhole",
    37: "Dingo",
    38: "Doberman",
    39: "Foxhound Inglês",
    40: "Setter Inglês",
    41: "Springer Spaniel Inglês",
    42: "Entlebucher",
    43: "Cão Esquimó",
    44: "Retriever de Pelo Liso",
    45: "Bulldog Francês",
    46: "Pastor Alemão",
    47: "Pointer Alemão de Pelo Curto",
    48: "Schnauzer Gigante",
    49: "Retriever Dourado",
    50: "Setter Gordon",
    51: "Dog Alemão",
    52: "Grande Pireneus",
    53: "Cão da Montanha Suíça",
    54: "Groenendael",
    55: "Podengo Ibicenco",
    56: "Setter Irlandês",
    57: "Terrier Irlandês",
    58: "Spaniel de Água Irlandês",
    59: "Lobo Irlandês",
    60: "Galgo Italiano",
    61: "Spaniel Japonês",
    62: "Keeshond",
    63: "Kelpie",
    64: "Terrier Azul de Kerry",
    65: "Komondor",
    66: "Kuvasz",
    67: "Retriever Labrador",
    68: "Terrier Lakeland",
    69: "Leonberger",
    70: "Lhasa Apso",
    71: "Malamute do Alasca",
    72: "Malinois",
    73: "Maltês",
    74: "Cachorro Sem Pelo Mexicano",
    75: "Pinscher Miniatura",
    76: "Poodle Miniatura",
    77: "Schnauzer Miniatura",
    78: "Terra Nova",
    79: "Terrier Norfolk",
    80: "Elkhound Norueguês",
    81: "Terrier Norwich",
    82: "Velho Cão Pastor Inglês",
    83: "Otterhound",
    84: "Papillon",
    85: "Pequinês",
    86: "Pembroke",
    87: "Spitz Alemão",
    88: "Pug",
    89: "Redbone",
    90: "Crestado da Rodésia",
    91: "Rottweiler",
    92: "São Bernardo",
    93: "Saluki",
    94: "Samoyeda",
    95: "Schipperke",
    96: "Terrier Escocês",
    97: "Cervo Escocês",
    98: "Terrier Sealyham",
    99: "Cão Pastor Shetland",
    100: "Shih Tzu",
    101: "Husky Siberiano",
    102: "Terrier Sedoso",
    103: "Terrier Wheaten de Pelo Macio",
    104: "Staffordshire Bull Terrier",
    105: "Poodle Padrão",
    106: "Schnauzer Padrão",
    107: "Spaniel Sussex",
    108: "Mastim Tibetano",
    109: "Terrier Tibetano",
    110: "Poodle Toy",
    111: "Terrier Toy",
    112: "Vizsla",
    113: "Coonhound Walker",
    114: "Weimaraner",
    115: "Spaniel Galês",
    116: "West Highland White Terrier",
    117: "Whippet",
    118: "Terrier Fox de Pelo Duro",
    119: "Yorkshire Terrier"
}

# Página inicial
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint para predição
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Nenhuma imagem foi enviada"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    try:
        # Criar um arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        temp_file.close()  # Fechar o arquivo para que possa ser removido

        # Carregar a imagem
        image = load_img(temp_file.name, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)

        # Fazer a predição
        predictions = model.predict(image)
        class_idx = np.argmax(predictions)
        class_prob = predictions[0][class_idx]

        # Remover o arquivo temporário
        os.remove(temp_file.name)

        return jsonify({
            "predicted_class": class_labels[class_idx],
            "probability": float(class_prob) * 100
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Rodar o app
    app.run(debug=True)
