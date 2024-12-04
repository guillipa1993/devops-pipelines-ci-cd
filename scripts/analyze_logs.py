import os
import openai

# Inicializar la API de OpenAI con la clave
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_logs(log_dir):
    # Recorrer todos los archivos de logs
    for log_file in os.listdir(log_dir):
        with open(os.path.join(log_dir, log_file), 'r') as f:
            log_content = f.read()
            
            # Dividir el log en fragmentos para enviarlos a la API
            log_fragments = [log_content[i:i+2000] for i in range(0, len(log_content), 2000)]
            
            for fragment in log_fragments:
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=f"Analyze the following log fragment: \n{fragment}\n",
                    max_tokens=150
                )
                analysis = response['choices'][0]['text'].strip()
                
                # Guardar el análisis
                save_analysis(log_file, analysis)

def save_analysis(log_file, analysis):
    # Guardar el resultado del análisis en un archivo con el mismo nombre pero en otro directorio
    analysis_dir = "./analysis-results"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    analysis_file_path = os.path.join(analysis_dir, f"{log_file}_analysis.txt")
    with open(analysis_file_path, 'w') as f:
        f.write(analysis)

if __name__ == "__main__":
    log_directory = "./logs"
    analyze_logs(log_directory)
