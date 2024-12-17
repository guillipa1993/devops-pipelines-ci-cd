from openai import OpenAI
import os

# Inicializar cliente OpenAI con la clave de API
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Listar modelos disponibles
try:
    models = client.models.list()
    print("Available models:")
    for model in models.data:
        print(model.id)
except Exception as e:
    print(f"Error while listing models: {e}")
