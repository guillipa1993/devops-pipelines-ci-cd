import os
import openai
import argparse

# Inicializar la API de OpenAI con la clave
openai.api_key = os.getenv("OPENAI_API_KEY")

def validate_logs_directory(log_dir):
    """
    Valida si el directorio de logs existe y contiene archivos.
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")
    
    log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if os.path.isfile(os.path.join(log_dir, f))]
    if not log_files:
        raise FileNotFoundError(f"ERROR: No log files found in the directory '{log_dir}'.")
    
    return log_files

def analyze_logs(log_files):
    """
    Analiza los logs utilizando la API de OpenAI.
    """
    for log_file in log_files:
        with open(log_file, 'r') as f:
            log_content = f.read()
            
            # Dividir el log en fragmentos para enviarlos a la API
            log_fragments = [log_content[i:i+2000] for i in range(0, len(log_content), 2000)]
            
            for fragment in log_fragments:
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a log analysis assistant."},
                        {"role": "user", "content": f"Analyze the following log fragment: \n{fragment}\n"}
                    ],
                    max_tokens=150
                )
                analysis = response['choices'][0]['message']['content'].strip()
                
                # Guardar el análisis
                save_analysis(log_file, analysis)

def save_analysis(log_file, analysis):
    """
    Guarda el análisis en un archivo en el directorio `analysis-results`.
    """
    analysis_dir = "./analysis-results"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    analysis_file_path = os.path.join(analysis_dir, f"{os.path.basename(log_file)}_analysis.txt")
    with open(analysis_file_path, 'w') as f:
        f.write(analysis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze logs using OpenAI")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the logs directory")
    args = parser.parse_args()

    try:
        # Validar el directorio de logs
        log_files = validate_logs_directory(args.log_dir)
        print(f"Found {len(log_files)} log files in '{args.log_dir}'.")
        
        # Analizar los logs
        analyze_logs(log_files)
        print("Log analysis completed successfully.")
    except Exception as e:
        print(str(e))
        exit(1)
