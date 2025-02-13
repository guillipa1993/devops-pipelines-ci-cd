import os
import time
import argparse
from openai import OpenAI

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
        if os.path.isfile(os.path.join(log_dir, f))
           and f.endswith(".log")
           and "python-vulnerabilities.log" not in f
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

def extract_relevant_lines(content, keyword="error", context_lines=10):
    """
    Extrae líneas que contienen un keyword específico y las líneas circundantes.
    """
    lines = content.splitlines()
    relevant_lines = []
    for idx, line in enumerate(lines):
        if keyword.lower() in line.lower():
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            relevant_lines.extend(lines[start:end])
    return "\n".join(relevant_lines)

def generate_prompt(log_type):
    """
    Genera el prompt según el tipo de log.
    """
    if log_type == "failure":
        return (
            "You are an expert log analysis assistant. The provided log type is 'failure'. "
            "Your primary goal is to identify, explain, and resolve issues found in the logs. "
            "Ensure the analysis includes: "
            "1. Root Cause Analysis (🔍🛠️): Pinpoint specific reasons for the failure, referencing patterns or anomalies. "
            "2. Actionable Fixes (💡🔧): Provide step-by-step recommendations to resolve each issue, with code where relevant. "
            "3. Preventive Measures (⚡🔒): Suggest changes to configurations, dependencies, or workflows to avoid repeats. "
            "4. Critical Issue Highlighting (💣🔥): Indicate urgent or blocking issues. "
            "5. Impact Analysis (🚫⚠️): Potential consequences if unaddressed."
        )
    else:
        return (
            "You are an expert log analysis assistant. The provided log type is 'success'. "
            "Your primary goal is to confirm the success of the process and provide insights to sustain or improve it. "
            "Ensure the analysis includes: "
            "1. Confirmation of Success (✅🌟): Clearly state completion without critical issues. "
            "2. Opportunities for Optimization (🚀💡): Suggest improvements or simplifications. "
            "3. Scalability Recommendations (📈🛡️): Show how to extend this success to larger scales. "
            "4. Sustainability Suggestions (🌿✨): Propose maintaining current success, e.g. best practices. "
            "5. Positive Feedback (🎉🙌): Acknowledge good practices or results achieved."
        )

def analyze_logs(log_files, output_dir, log_type):
    """
    Analiza los logs utilizando la API de OpenAI, generando un análisis por cada fragmento.
    """
    analysis_created = False
    total_files = len(log_files)

    for file_idx, log_file in enumerate(log_files, start=1):
        print(f"\n[{file_idx}/{total_files}] Analyzing file: {log_file}", flush=True)
        with open(log_file, 'r') as f:
            log_content = f.read()

            # ADDED: Si es log de error, extraemos las partes cercanas a 'error'.
            # Si no, limpiamos. Ajustable según preferencia.
            if log_type == "failure":
                relevant_content = extract_relevant_lines(log_content, keyword="error")
            else:
                relevant_content = clean_log_content(log_content)

            # Dividir el contenido en fragmentos de hasta ~30k caracteres
            max_chunk_size = 30000
            log_fragments = [relevant_content[i:i + max_chunk_size] for i in range(0, len(relevant_content), max_chunk_size)]
            total_fragments = len(log_fragments)

            print(f"File '{log_file}' divided into {total_fragments} fragments for analysis.", flush=True)

            for idx, fragment in enumerate(log_fragments, 1):
                print(f"   Analyzing fragment {idx}/{total_fragments} of file '{log_file}'...", flush=True)
                try:
                    role_content = generate_prompt(log_type)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": role_content},
                            {"role": "user", "content": fragment}
                        ],
                        max_tokens=2000,
                        temperature=0.5
                    )
                    analysis = response.choices[0].message.content.strip()
                    if analysis:
                        save_analysis(log_file, analysis, idx, output_dir, log_type)
                        analysis_created = True
                        print(f"   Fragment {idx}/{total_fragments} analysis complete.", flush=True)
                    else:
                        print(f"   WARNING: No analysis returned for fragment {idx}.", flush=True)

                    # Espera breve para no sobrecargar la API (ajusta si hace falta)
                    time.sleep(5)
                except Exception as e:
                    print(f"Unexpected error while analyzing fragment {idx}: {e}", flush=True)
                    break

    if not analysis_created:
        print("WARNING: No analysis files were created. Please check the logs for issues.", flush=True)

def save_analysis(log_file, analysis, fragment_idx, output_dir, log_type):
    """
    Guarda el análisis en un archivo en el directorio proporcionado.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    analysis_file_path = os.path.join(
        output_dir, f"{os.path.basename(log_file)}_{log_type}_fragment_{fragment_idx}_analysis.txt"
    )
    with open(analysis_file_path, 'w') as f:
        f.write(analysis)
    print(f"   Analysis for fragment {fragment_idx} saved to {analysis_file_path}", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze log files using OpenAI")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the logs directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the analysis results")
    parser.add_argument("--log-type", type=str, required=True, choices=["success", "failure"], help="Specify the type of logs to analyze: 'success' or 'failure'")
    args = parser.parse_args()

    try:
        log_files = validate_logs_directory(args.log_dir)
        print(f"Found {len(log_files)} log files in '{args.log_dir}'.", flush=True)

        analyze_logs(log_files, args.output_dir, args.log_type)

        print("Log analysis completed successfully.", flush=True)
    except Exception as e:
        print(f"Critical error: {e}", flush=True)
        exit(1)
