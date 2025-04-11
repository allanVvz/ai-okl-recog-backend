# aws_downloader.py
import boto3
import botocore
import os
import logging
from dotenv import load_dotenv

# Configura o logger para o módulo
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class S3Downloader:
    def __init__(self, bucket_name, region='us-east-1'):
        self.bucket_name = bucket_name
        load_dotenv()
        # Lê as variáveis de ambiente utilizando os nomes corretos
        aws_access_key = os.environ.get("MY_AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("MY_AWS_SECRET_ACCESS_KEY")
        if not aws_access_key or not aws_secret_key:
            logger.error("Credenciais AWS não configuradas nas variáveis de ambiente.")
            raise ValueError("Variáveis de ambiente AWS_ACCESS_KEY_ID e AWS_SECRET_ACCESS_KEY são necessárias.")

        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=region
        )

    def download_file(self, object_key, download_path):
        """
        Faz o download de um arquivo do bucket S3 para o caminho especificado.
        Se o arquivo já existir, não realiza o download.
        
        :param object_key: A chave (nome) do objeto no bucket S3.
        :param download_path: Caminho completo onde salvar o arquivo baixado.
        """
        try:
            # Cria o diretório se necessário
            download_dir = os.path.dirname(download_path)
            if not os.path.exists(download_dir):
                os.makedirs(download_dir, exist_ok=True)
            
            if not os.path.exists(download_path):
                logger.info("Iniciando o download do arquivo S3: bucket=%s, objeto=%s, destino=%s",
                            self.bucket_name, object_key, download_path)
                # Realiza o download usando boto3
                self.s3.download_file(self.bucket_name, object_key, download_path)
                file_size = os.path.getsize(download_path)
                logger.info("Arquivo baixado com sucesso! Tamanho: %d bytes", file_size)
            else:
                logger.info("Arquivo já existe em %s, não é necessário baixar novamente.", download_path)
        except botocore.exceptions.ClientError as e:
            logger.exception("Erro ao baixar o arquivo do S3: %s", e)
            raise
        except Exception as ex:
            logger.exception("Erro inesperado ao baixar o arquivo: %s", ex)
            raise

