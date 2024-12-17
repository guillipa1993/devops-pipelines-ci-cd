import os
import argparse
from openai import OpenAI
from openai import AuthenticationError, APIError, BadRequestError

# Verificar si la clave de API está configurada
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

# Inicializar la API de OpenAI con el cliente
client = OpenAI(api_key=api_key)

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
        print(f"Analyzing file: {log_file}")
        with open(log_file, 'r') as f:
            log_content = f.read()
            
            # Dividir el log en fragmentos para enviarlos a la API
            log_fragments = [log_content[i:i+4000] for i in range(0, len(log_content), 4000)]
            print(f"Total fragments for file '{log_file}': {len(log_fragments)}")
            
            for idx, fragment in enumerate(log_fragments, 1):
                print(f"Analyzing fragment {idx}/{len(log_fragments)} of file '{log_file}'")
                try:
                    # Usar el cliente correcto con el formato actualizado
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a log analysis assistant. Provide insights and recommendations based on the following log fragment."},
                            {"role": "user", "content": fragment}
                        ],
                        max_tokens=500,
                        temperature=0.5
                    )
                    analysis = response.choices[0].message.content.strip()
                    print(f"Fragment {idx} analysis complete.")
                    
                    # Guardar el análisis
                    save_analysis(log_file, analysis, idx)
                except AuthenticationError:
                    print("ERROR: Authentication failed. Check your API key.")
                    return
                except BadRequestError as bre:
                    print(f"Bad request error for fragment {idx}: {bre}")
                except APIError as ae:
                    print(f"API error while analyzing fragment {idx}: {ae}")
                except Exception as e:
                    print(f"Unexpected error while analyzing fragment {idx}: {e}")
                    continue

def save_analysis(log_file, analysis, fragment_idx):
    """
    Guarda el análisis en un archivo en el directorio `analysis-results`.
    """
    analysis_dir = "./analysis-results"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    analysis_file_path = os.path.join(
        analysis_dir, f"{os.path.basename(log_file)}_fragment_{fragment_idx}_analysis.txt"
    )
    with open(analysis_file_path, 'w') as f:
        f.write(analysis)
    print(f"Analysis for fragment {fragment_idx} saved to {analysis_file_path}")

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
        print(f"Critical error: {e}")
        exit(1)
