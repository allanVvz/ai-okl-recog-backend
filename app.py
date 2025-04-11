from flask import Flask, request, jsonify
from PIL import Image
import io
import logging
from inference_model import predict
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Configuração do logging para o aplicativo
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/upload', methods=['POST'])
def upload_image():
    load_dotenv()  # Isto carrega as variáveis do .env para o os.environ

    logger.info("Requisição de upload recebida.")
    if 'image' not in request.files:
        logger.error("Nenhuma imagem fornecida na requisição.")
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    try:
        image = Image.open(io.BytesIO(image_file.read()))
        logger.info("Imagem carregada com sucesso.")
    except Exception as e:
        logger.exception("Erro ao processar a imagem: %s", e)
        return jsonify({"error": "Invalid image file"}), 400

    try:
        prediction = predict(image)
        logger.info("Inferência realizada com sucesso. Predição: %s", prediction)
    except Exception as e:
        logger.exception("Erro durante a inferência: %s", e)
        return jsonify({"error": "Inference failed"}), 500

    return jsonify({"prediction": prediction})

@app.route('/')
def home():
    return jsonify({"message": "API com modelo carregado funcionando! Envie imagens via POST para /upload"})

@app.route('/list-temp', methods=['GET'])
def list_temp_directory():
    temp_dir = '/tmp'
    try:
        files = os.listdir(temp_dir)
        return jsonify({"temp_directory": temp_dir, "files": files})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    logger.info("Iniciando a API REST...")
    app.run(debug=True, use_reloader=False)
