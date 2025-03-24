#!/usr/bin/env python3
import os
import sys
import re
import json
import time
import logging
import argparse
import subprocess
from typing import List, Optional

import openai  # Se importa 'openai' en vez de 'openai.error'
from openai import OpenAI

# ===================================
# CONFIGURACIÓN DE LOGGING
# ===================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================================
# CONFIGURACIÓN OPENAI
# ===================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ===================================
# CONSTANTES Y PARÁMETROS
# ===================================
OPENAI_MODEL = "gpt-4o"       # Se asume que usarás 'gpt-4o', igual que en tu ejemplo
MAX_CHUNK_SIZE = 30000        # Tamaño de fragmento al dividir logs
MAX_TOKENS_OPENAI = 2000      # Máx tokens para la respuesta de OpenAI
TEMPERATURE_OPENAI = 0.5      # Ajusta la temperatura de las respuestas
MAX_RETRIES = 5               # Reintentos en caso de error 429
BASE_DELAY = 2.0              # Espera base (segundos) para backoff
EXCLUDE_FILE = "python-vulnerabilities.log"  # Ejemplo de log excluido

# ===================================
# FUNCIONES AUXILIARES
# ===================================
def validate_logs_directory(log_dir: str) -> List[str]:
    """
    Valida si el directorio de logs existe y contiene archivos .log
    (excluye uno en concreto y descarta lo que no termine en .log).
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")

    log_files = []
    for f in os.listdir(log_dir):
        file_path = os.path.join(log_dir, f)
        if os.path.isfile(file_path) and f.endswith(".log") and EXCLUDE_FILE not in f:
            log_files.append(file_path)

    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid .log files found in the directory '{log_dir}'.")
    return log_files

def clean_log_content(content: str) -> str:
    """
    Elimina líneas vacías (o espacios) del log.
    """
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]  # Elimina líneas vacías
    return "\n".join(cleaned_lines)

def extract_relevant_lines(content: str, keyword: str = "error", context_lines: int = 10) -> str:
    """
    Extrae las líneas que contienen un 'keyword' (ej. 'error')
    y un cierto número de líneas antes y después.
    """
    lines = content.splitlines()
    relevant_lines = []
    for idx, line in enumerate(lines):
        if keyword.lower() in line.lower():
            start = max(0, idx - context_lines)
            end = min(len(lines), idx + context_lines + 1)
            relevant_lines.extend(lines[start:end])
    return "\n".join(relevant_lines)

def generate_prompt(log_type: str) -> str:
    """
    Genera el 'prompt' base según el tipo de log.
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

def chunk_string(text: str, chunk_size: int = MAX_CHUNK_SIZE) -> List[str]:
    """
    Divide 'text' en fragmentos de un tamaño máximo.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def call_openai_with_retry(prompt_role: str, prompt_content: str) -> Optional[str]:
    """
    Llama a la API de OpenAI (chat.completions) con reintentos en caso de error 429.
    Devuelve el texto de la respuesta o None si falla.
    """
    messages = [
        {"role": "system", "content": prompt_role},
        {"role": "user", "content": prompt_content}
    ]

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                temperature=TEMPERATURE_OPENAI,
                max_tokens=MAX_TOKENS_OPENAI
            )
            return response.choices[0].message.content.strip()

        except openai.error.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI RateLimitError (429). Reintentando en %.1f s (Intento %d/%d).",
                    wait_time, attempt+1, MAX_RETRIES
                )
                time.sleep(wait_time)
            else:
                logger.error("Agotados reintentos tras 429.")
                return None

        except openai.error.APIError as e:
            if getattr(e, "http_status", None) == 429 and attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI APIError 429. Reintentando en %.1f s (Intento %d/%d).",
                    wait_time, attempt+1, MAX_RETRIES
                )
                time.sleep(wait_time)
            else:
                logger.error("APIError no recuperable: %s", e)
                return None

        except Exception as ex:
            logger.error("Error no controlado al llamar a OpenAI: %s", ex)
            return None

    return None

def save_analysis(log_file: str, analysis: str, fragment_idx: int, output_dir: str, log_type: str):
    """
    Guarda el análisis en un archivo dentro de 'output_dir'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    base_name = os.path.basename(log_file)
    analysis_filename = f"{base_name}_{log_type}_fragment_{fragment_idx}_analysis.txt"
    analysis_file_path = os.path.join(output_dir, analysis_filename)

    with open(analysis_file_path, "w", encoding="utf-8") as f:
        f.write(analysis)

    logger.info("   Analysis for fragment %d saved to %s", fragment_idx, analysis_file_path)

# =============================================
# FUNCIÓN PRINCIPAL DE ANÁLISIS
# =============================================
def analyze_logs(log_files: List[str], output_dir: str, log_type: str):
    """
    Analiza cada archivo de log en 'log_files' con la API de OpenAI.
    Guarda un archivo de análisis por cada fragmento que se produce.
    """
    analysis_created = False
    total_files = len(log_files)

    for file_idx, log_file in enumerate(log_files, start=1):
        logger.info("[%d/%d] Analyzing file: %s", file_idx, total_files, log_file)
        with open(log_file, "r", encoding="utf-8") as f:
            log_content = f.read()

        # Según log_type, extraemos contenido relevante o lo limpiamos
        if log_type == "failure":
            relevant_content = extract_relevant_lines(log_content, keyword="error")
        else:
            relevant_content = clean_log_content(log_content)

        # Fragmentar contenido
        log_fragments = chunk_string(relevant_content, MAX_CHUNK_SIZE)
        total_fragments = len(log_fragments)

        logger.info("File '%s' divided into %d fragments for analysis.", log_file, total_fragments)

        # Generar prompt base (rol "system")
        role_content = generate_prompt(log_type)

        for frag_idx, fragment in enumerate(log_fragments, start=1):
            logger.info("   Analyzing fragment %d/%d of file '%s'...", frag_idx, total_fragments, log_file)

            result = call_openai_with_retry(
                prompt_role=role_content,
                prompt_content=fragment
            )
            if result is None:
                logger.warning("   WARNING: No result returned for fragment %d.", frag_idx)
                continue  # Pasar al siguiente fragmento

            result = result.strip()
            if result:
                save_analysis(log_file, result, frag_idx, output_dir, log_type)
                analysis_created = True
                logger.info("   Fragment %d/%d analysis complete.", frag_idx, total_fragments)
            else:
                logger.warning("   WARNING: Empty analysis for fragment %d.", frag_idx)

            # Breve espera para no saturar la API (puedes ajustar o quitar)
            time.sleep(3)

    if not analysis_created:
        logger.warning("WARNING: No analysis files were created. Please check your logs or prompts.")

# =============================================
# MAIN
# =============================================
def main():
    parser = argparse.ArgumentParser(description="Analyze log files using OpenAI")
    parser.add_argument("--log-dir", required=True, help="Path to the logs directory")
    parser.add_argument("--output-dir", required=True, help="Path to save the analysis results")
    parser.add_argument("--log-type", required=True, choices=["success","failure"], help="Specify the type of logs to analyze")
    args = parser.parse_args()

    try:
        log_files = validate_logs_directory(args.log_dir)
        logger.info("Found %d log file(s) in '%s'.", len(log_files), args.log_dir)

        analyze_logs(log_files, args.output_dir, args.log_type)

        logger.info("Log analysis completed successfully.")
    except Exception as e:
        logger.error("Critical error: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
