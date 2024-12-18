import os
import time
import argparse
import openai
from openai import OpenAI, RateLimitError, AuthenticationError, APIError, BadRequestError

# Verificar si la clave de API está configurada
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

# Inicializar la API de OpenAI
client = OpenAI(api_key=api_key)

def validate_logs_directory(log_dir):
    """
    Valida si el directorio de logs existe y contiene archivos .log excluyendo archivos JSON.
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")

    log_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, f)) and f.endswith(".log") and "python-vulnerabilities.log" not in f
    ]
    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid .log files found in the directory '{log_dir}'.")
    return log_files

def clean_log_content(content):
    """
    Elimina líneas vacías y contenido redundante del log.
    """
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]  # Elimina líneas vacías
    return "\n".join(cleaned_lines)

def analyze_logs(log_files):
    """
    Analiza los logs utilizando la API de OpenAI.
    """
    total_files = len(log_files)
    for file_idx, log_file in enumerate(log_files, start=1):
        print(f"\n[{file_idx}/{total_files}] Analyzing file: {log_file}", flush=True)
        with open(log_file, 'r') as f:
            log_content = f.read()
            cleaned_content = clean_log_content(log_content)

            # Dividir el contenido en fragmentos de hasta 30,000 tokens
            max_chunk_size = 30000  # Tokens por fragmento
            log_fragments = [cleaned_content[i:i + max_chunk_size] for i in range(0, len(cleaned_content), max_chunk_size)]
            total_fragments = len(log_fragments)

            print(f"File '{log_file}' divided into {total_fragments} fragments for analysis.", flush=True)
            
            for idx, fragment in enumerate(log_fragments, 1):
                print(f"   Analyzing fragment {idx}/{total_fragments} of file '{log_file}'...", flush=True)
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": "You are a log analysis assistant. Provide insights and recommendations based on the following log fragment."},
                            {"role": "user", "content": fragment}
                        ],
                        max_tokens=1000,
                        temperature=0.5
                    )
                    analysis = response.choices[0].message.content.strip()
                    save_analysis(log_file, analysis, idx)
                    print(f"   Fragment {idx}/{total_fragments} analysis complete.", flush=True)
                    
                    # Pausa exacta para cumplir 3 RPM
                    time.sleep(20)
                except RateLimitError:
                    print("Rate limit exceeded. Waiting 20 seconds...", flush=True)
                    time.sleep(20)  # Espera fija de 20 segundos en caso de RateLimitError
                except (AuthenticationError, BadRequestError, APIError) as e:
                    print(f"Error while analyzing fragment {idx}: {e}", flush=True)
                    break
                except Exception as e:
                    print(f"Unexpected error while analyzing fragment {idx}: {e}", flush=True)
                    break

def save_analysis(log_file, analysis, fragment_idx):
    """
    Guarda el análisis en un archivo en el directorio `analysis_results` dentro de GITHUB_WORKSPACE.
    """
    analysis_dir = os.path.join(os.getenv("GITHUB_WORKSPACE", "."), "analysis_results")

    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)  # Crea el directorio si no existe

    analysis_file_path = os.path.join(
        analysis_dir, f"{os.path.basename(log_file)}_fragment_{fragment_idx}_analysis.txt"
    )
    with open(analysis_file_path, 'w') as f:
        f.write(analysis)
    print(f"   Analysis for fragment {fragment_idx} saved to {analysis_file_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze log files using OpenAI")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the logs directory")
    args = parser.parse_args()

    try:
        # Validar el directorio de logs
        log_files = validate_logs_directory(args.log_dir)
        print(f"Found {len(log_files)} log files in '{args.log_dir}'.", flush=True)

        # Analizar los logs
        analyze_logs(log_files)

        print("Log analysis completed successfully.", flush=True)
    except Exception as e:
        print(f"Critical error: {e}", flush=True)
        exit(1)
