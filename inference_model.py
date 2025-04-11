import logging
from PIL import Image
import torch
from torchvision import transforms
import timm
from AWS3Dowloader import S3Downloader
import tempfile
import os



# Configuração do logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Definindo a transformação para a imagem
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def load_model(model_path=None):
    # Se nenhum caminho foi fornecido, usa o diretório temporário do sistema
    if model_path is None:
        temp_dir = tempfile.gettempdir()  # Ex: C:\Users\<usuário>\AppData\Local\Temp no Windows
        model_path = os.path.join(temp_dir, 'swin_oculos_model_state_dict.pth')

    logger.info("Iniciando o download do state_dict do modelo...")
    try:
        downloader = S3Downloader(bucket_name='vzvzz')
        # Supomos que o arquivo esteja na raiz do bucket ou ajuste o prefixo conforme necessário
        downloader.download_file('swin_oculos_model_state_dict.pth', model_path)
        logger.info("Arquivo baixado, carregando o state_dict...")
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model = timm.create_model('swin_small_patch4_window7_224', pretrained=False, num_classes=2)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info("Modelo carregado e configurado para avaliação.")
    except Exception as e:
        logger.exception("Erro ao carregar o modelo: %s", e)
        raise
    return model
# Carrega o modelo globalmente
model = load_model()

# Defina as classes de saída conforme sua aplicação
classes = ['Juliet', 'Radar']

def preprocess_image(image: Image.Image):
    """
    Pré-processa a imagem para inferência.
    """
    logger.debug("Iniciando pré-processamento da imagem...")
    try:
        image_tensor = preprocess_transform(image)
        # Adiciona a dimensão de batch
        image_tensor = image_tensor.unsqueeze(0)
        logger.debug("Pré-processamento concluído.")
        return image_tensor
    except Exception as e:
        logger.exception("Erro durante o pré-processamento: %s", e)
        raise

def predict(image: Image.Image) -> str:
    """
    Executa a inferência e retorna a classe prevista.
    """
    logger.debug("Iniciando inferência da imagem...")
    try:
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = torch.argmax(output, dim=1).item()
            predicted_class = classes[predicted_idx]
        logger.debug("Inferência concluída com sucesso. Predição: %s", predicted_class)
        return predicted_class
    except Exception as e:
        logger.exception("Erro durante a inferência: %s", e)
        raise
