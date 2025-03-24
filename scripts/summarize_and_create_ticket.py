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

import openai  # <- Se importa 'openai' en lugar de 'openai.error'
from openai import OpenAI
from datetime import datetime

# ================================================
# CONFIGURACIÓN DE LOGGING
# ================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# ================================================
# VERIFICACIÓN Y CONFIGURACIÓN DE OPENAI
# ================================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ================================================
# PARÁMETROS GLOBALES Y CONSTANTES
# ================================================
MAX_CHUNK_SIZE = 30000      # Tamaño de fragmento al unir logs
MAX_TOKENS_OPENAI = 2000    # Máx tokens en la respuesta de OpenAI
TEMPERATURE_OPENAI = 0.5    # Ajusta la temperatura
MAX_RETRIES = 5             # Reintentos si 429
BASE_DELAY = 2.0            # Espera inicial (segundos) para backoff

# ================================================
# FUNCIONES AUXILIARES
# ================================================
def validate_logs_directory(log_dir: str) -> List[str]:
    """
    Valida si el directorio de logs existe y contiene archivos .txt.
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")
    log_files = []
    for f in os.listdir(log_dir):
        path_ = os.path.join(log_dir, f)
        if os.path.isfile(path_) and f.endswith(".txt"):
            log_files.append(path_)
    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid .txt files found in '{log_dir}'.")
    return log_files

def clean_log_content(content: str) -> str:
    """
    Elimina líneas vacías y contenido redundante del log.
    """
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

def generate_prompt(log_type: str, language: Optional[str]) -> str:
    """
    Genera el prompt según el tipo de log y el idioma.
    """
    if log_type == "failure":
        prompt = (
            "You are an expert log analysis assistant. The provided log type is 'failure'. "
            "Your primary goal is to analyze the logs to identify issues, explain their root causes, and recommend actionable fixes. "
            "Ensure the analysis includes:\n"
            "1. **🔍 Root Cause Analysis:** Identify specific reasons for the failure, with supporting evidence from the logs.\n"
            "2. **🔧 Actionable Fixes:** Provide step-by-step recommendations, including file names, line numbers, and code examples.\n"
            "3. **⚡ Preventive Measures:** Suggest workflow or dependency changes to avoid similar issues.\n"
            "4. **🔥 Critical Issues:** Highlight urgent or blocking issues clearly.\n"
            "5. **🚫 Impact Analysis:** Describe potential consequences of unresolved issues.\n"
            "6. **🎯 Next Steps:** Summarize key actions to resolve issues and improve system reliability.\n"
        )
    else:
        prompt = (
            "You are an expert log analysis assistant. The provided log type is 'success'. "
            "Your primary goal is to confirm the success of the process and suggest optimizations for future scalability. "
            "Ensure the analysis includes:\n"
            "1. **✅ Confirmation of Success:** Clearly state that the process completed successfully.\n"
            "2. **🚀 Opportunities for Optimization:** Suggest areas for performance improvement or simplification.\n"
            "3. **📈 Scalability Recommendations:** Provide advice on extending the success to larger workloads.\n"
            "4. **🌿 Sustainability Suggestions:** Propose best practices to maintain success over time.\n"
            "5. **🎉 Positive Feedback:** Acknowledge good practices.\n"
            "6. **✨ Best Practices:** Offer actionable tips to replicate this success.\n"
        )
    if language:
        prompt += f" Generate all response text in {language}."
    return prompt

def call_openai_with_retry(system_prompt: str, user_prompt: str) -> str:
    """
    Llama a la API de OpenAI con reintentos en caso de error 429.
    Devuelve el texto resultante o cadena vacía si falla.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=MAX_TOKENS_OPENAI,
                temperature=TEMPERATURE_OPENAI
            )
            return response.choices[0].message.content.strip()

        except openai.error.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning("OpenAI RateLimitError (429). Retrying in %.1f s (Attempt %d/%d).",
                               wait_time, attempt+1, MAX_RETRIES)
                time.sleep(wait_time)
            else:
                logger.error("Max retries reached after 429. Aborting.")
                return ""

        except openai.error.APIError as e:
            if getattr(e, "http_status", None) == 429 and attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning("OpenAI APIError 429. Retrying in %.1f s (Attempt %d/%d).",
                               wait_time, attempt+1, MAX_RETRIES)
                time.sleep(wait_time)
            else:
                logger.error("APIError not recoverable or max retries exhausted: %s", e)
                return ""

        except Exception as ex:
            logger.error("Unhandled error calling OpenAI: %s", ex)
            return ""

    return ""

def create_github_issue(title: str, body: str, repo_name: str):
    """
    Crea un ticket en GitHub usando la CLI `gh`.
    """
    try:
        logger.info("Attempting to create a GitHub issue via 'gh' CLI...")
        command = [
            "gh", "issue", "create",
            "--repo", repo_name,
            "--title", title,
            "--body", body
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        logger.info("GitHub issue created successfully. Output:\n%s", result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error("Failed to create GitHub issue. CLI error output:\n%s", e.stderr)

def summarize_logs_with_openai(log_dir: str, log_type: str, language: Optional[str]) -> str:
    """
    Resume los logs en 'log_dir' usando la API de OpenAI.
    """
    log_files = validate_logs_directory(log_dir)
    logger.info("Found %d '.txt' files in '%s'.", len(log_files), log_dir)

    # Unimos todos los contenidos
    all_content = ""
    for filename in log_files:
        with open(filename, "r", encoding="utf-8") as f:
            file_content = f.read()
        cleaned = clean_log_content(file_content)
        # Separador con nombre del archivo
        base_name = os.path.basename(filename)
        all_content += f"### 📄 {base_name}\n{cleaned}\n\n"

    # Dividimos en fragmentos para evitar exceder MAX_CHUNK_SIZE
    content_fragments = [
        all_content[i:i+MAX_CHUNK_SIZE]
        for i in range(0, len(all_content), MAX_CHUNK_SIZE)
    ]

    # Generamos prompt base
    system_content = generate_prompt(log_type, language)

    consolidated_summary = ""
    for idx, fragment in enumerate(content_fragments, start=1):
        logger.info("Analyzing fragment %d/%d from log_dir...", idx, len(content_fragments))
        partial_summary = call_openai_with_retry(system_content, fragment)
        if partial_summary:
            consolidated_summary += partial_summary + "\n\n"
        else:
            logger.warning("No response for fragment %d. Continuing...", idx)

    return consolidated_summary.strip()

# ===========================================
# MAIN
# ===========================================
def main():
    parser = argparse.ArgumentParser(description="Summarize analysis and create GitHub issue.")
    parser.add_argument("--log-dir", required=True, help="Path to the directory with analysis results.")
    parser.add_argument("--output-file", required=True, help="Path to save the summary output.")
    parser.add_argument("--repo", required=True, help="GitHub repository (e.g., owner/repo).")
    parser.add_argument("--run-id", required=True, help="GitHub Actions run ID.")
    parser.add_argument("--run-url", required=False, help="URL to the GitHub Actions run.")
    parser.add_argument("--log-type", required=True, choices=["success", "failure"], help="Logs type to analyze.")
    parser.add_argument("--report-language", required=False, help="Language for the summary report.")
    parser.add_argument("--repo-and-owner", required=True, help="GitHub repository owner/repo.")
    parser.add_argument("--create-ticket", action="store_true", help="Flag to create a GitHub issue.")
    args = parser.parse_args()

    summary = summarize_logs_with_openai(args.log_dir, args.log_type, args.report_language)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Guardamos el resumen
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    # Si se desea crear ticket
    if args.create_ticket:
        # Título en base a si es success/failure
        status_emoji = "🔴 Failure" if args.log_type == "failure" else "🟢 Success"
        status_text = "Errors Found" if args.log_type == "failure" else "All Passed"

        title = f"{status_emoji} Report - Project: {args.repo} - Build ID: {args.run_id} - {status_text}"
        create_github_issue(title, summary, args.repo_and_owner)

if __name__ == "__main__":
    main()
