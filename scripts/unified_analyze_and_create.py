#!/usr/bin/env python3
import os
import sys
import re
import json
import random
import math
import argparse
import requests
import logging
import time

from datetime import datetime
from difflib import SequenceMatcher
from jira import JIRA
from openai import OpenAI
from typing import List, Optional, Dict, Any

# =====================================================
# 1) CONFIGURACIÓN DEL LOGGING (simplificar formato)
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] - %(message)s"
)
logger = logging.getLogger()

# =====================================================
# CONFIGURACIÓN GLOBAL
# =====================================================
OPENAI_MODEL = "gpt-4o"
MAX_CHAR_PER_REQUEST = 20000
BANDIT_JSON_NAME = "bandit-output.json"
MAX_FILE_SIZE_MB = 2.0
ALLOWED_EXTENSIONS = (".log", ".sarif")

# Parámetros de reintentos
MAX_RETRIES = 5
BASE_DELAY = 1.0

# Límites básicos
MAX_TICKET_TITLE_LEN = 100   # Controla la longitud máxima de títulos
MAX_CONVERSATION_TOKENS = 4000

# =====================================================
# CONFIGURACIÓN OPENAI
# =====================================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("ERROR: 'OPENAI_API_KEY' is not set.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# =====================================================
# Historial reducido (si se desea)
# =====================================================
conversation_history: List[Dict[str, str]] = []

def init_conversation(system_content: str):
    global conversation_history
    conversation_history = [{"role": "system", "content": system_content}]

def add_user_message(user_content: str):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_content})

def ensure_history_not_excessive():
    total_tokens = sum(len(m["content"].split()) for m in conversation_history)
    if total_tokens > MAX_CONVERSATION_TOKENS:
        logger.info("** Re-inicializando el historial para evitar exceso de tokens **")
        system_msg = conversation_history[0]
        conversation_history.clear()
        conversation_history.append(system_msg)

# =====================================================
# Reintentos con backoff
# =====================================================
def chat_completions_create_with_retry(
    messages: List[Dict[str, str]],
    model: str = OPENAI_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> Any:
    import openai
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response
        except openai.error.RateLimitError as e:
            if attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"OpenAI RateLimitError (429). Reintentando en {wait_time:.1f}s (Intento {attempt+1}/{MAX_RETRIES})"
                )
                time.sleep(wait_time)
            else:
                logger.error("Agotados reintentos tras un 429. Abortando.")
                raise
        except openai.error.APIError as e:
            if getattr(e, "http_status", None) == 429 and attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    f"OpenAI APIError 429. Reintentando en {wait_time:.1f}s (Intento {attempt+1}/{MAX_RETRIES})"
                )
                time.sleep(wait_time)
            else:
                logger.error(f"APIError no recuperable o reintentos agotados: {e}")
                raise
        except Exception as ex:
            logger.error(f"Error no controlado en llamada a OpenAI: {ex}")
            raise
    raise RuntimeError("Se agotaron todos los reintentos en chat_completions_create_with_retry.")

# =====================================================
# Conexión a Jira
# =====================================================
def connect_to_jira(jira_url: str, jira_user: str, jira_api_token: str) -> JIRA:
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    logger.info("Conexión establecida con Jira.")
    return jira

# =====================================================
# Funciones de sanitización
# =====================================================
def sanitize_summary(summary: str) -> str:
    summary = summary.replace("\n", " ").replace("\r", " ")
    sanitized = "".join(c for c in summary if c.isalnum() or c.isspace() or c in "-_:,./()[]{}")
    return sanitized.strip()

def clamp_title_length(title: str, max_len: int = MAX_TICKET_TITLE_LEN) -> str:
    # Forzamos la longitud máxima
    if len(title) > max_len:
        return title[:max_len].rstrip()
    return title

def preprocess_text(text: str) -> str:
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    return text_no_punct.strip().lower()

def calculate_similarity(text1: str, text2: str) -> float:
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, preprocess_text(text1), preprocess_text(text2)).ratio()
    return ratio

# =====================================================
# Manejo de JSON devuelto por la IA
# =====================================================
def safe_load_json(ai_output: str) -> Optional[dict]:
    if ai_output.startswith("```"):
        lines = ai_output.splitlines()
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        ai_output = "\n".join(lines).strip()

    try:
        return json.loads(ai_output)
    except json.JSONDecodeError:
        cleaned = re.sub(r'```.*?```', '', ai_output, flags=re.DOTALL)
        last_brace = cleaned.rfind("}")
        if last_brace != -1:
            cleaned = cleaned[:last_brace+1]
        try:
            return json.loads(cleaned)
        except:
            return None

# =====================================================
# Filtrado de logs
# =====================================================
def validate_logs_directory(log_dir: str) -> List[str]:
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: Logs dir '{log_dir}' no existe.")
    log_files = []
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if file.lower() == BANDIT_JSON_NAME.lower():
            continue
        if os.path.isfile(file_path):
            mb_size = os.path.getsize(file_path) / 1024.0 / 1024.0
            if mb_size <= MAX_FILE_SIZE_MB:
                _, ext = os.path.splitext(file.lower())
                if ext in ALLOWED_EXTENSIONS:
                    log_files.append(file_path)
    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files in '{log_dir}'.")
    return log_files

# =====================================================
# Convertir ADF -> Wiki (opcional)
# =====================================================
def convert_adf_to_wiki(adf: dict) -> str:
    def process_node(node: Dict[str, Any]) -> str:
        node_type = node.get("type", "")
        content = node.get("content", [])

        if node_type == "paragraph":
            paragraph_text = ""
            for child in content:
                if child.get("type") == "text":
                    paragraph_text += child.get("text", "")
            return paragraph_text + "\n\n"
        elif node_type == "bulletList":
            lines_ = []
            for item in content:
                item_text = ""
                for child in item.get("content", []):
                    if child.get("type") == "paragraph":
                        for subchild in child.get("content", []):
                            if subchild.get("type") == "text":
                                item_text += subchild.get("text", "")
                lines_.append(f"* {item_text.strip()}")
            return "\n".join(lines_) + "\n\n"
        elif node_type == "codeBlock":
            code_text = ""
            for child in content:
                if child.get("type") == "text":
                    code_text += child.get("text", "")
            return f"{{code}}\n{code_text}\n{{code}}\n\n"
        elif node_type == "text":
            return node.get("text", "")

        result = ""
        for c in content:
            result += process_node(c)
        return result

    if not isinstance(adf, dict):
        return str(adf)
    if "content" not in adf:
        return ""

    wiki_text = ""
    for node in adf["content"]:
        wiki_text += process_node(node)

    return wiki_text.strip()

# =====================================================
# Reintentos para IA, parsear tickets
# =====================================================
def format_ticket_content(
    project_name: str,
    branch_name: str,
    rec_summary: str,
    rec_description: str,
    ticket_category: str
) -> (str, str):
    """
    Produce un 'title' y 'description' en wiki
    a partir de un JSON devuelto por la IA.
    - Aseguramos que el título lleve project, branch, 
      y se recorta a 100 chars max.
    - Si la IA no retorna nada útil, se hace fallback.
    """
    if ticket_category.lower() in ("improvement", "tarea"):
        icon = "💡"
    else:
        icon = "🔥"

    user_prompt = (
        "You are a professional ticket formatter. "
        "Given the recommendation details, produce a JSON with 'title' and 'description'. "
        f"Project: {project_name}\nBranch: {branch_name}\nRecommendation Title: {rec_summary}\nRecommendation Details: {rec_description}\nCategory: {ticket_category}\n"
        "Title must start with the project name and show the branch in brackets if not empty, plus a short summary.\n"
        "Description must be minimal ADF with code blocks using triple backticks.\n"
        "Return only JSON. No extra text."
    )

    local_messages = [
        {"role": "system", "content": "You are a highly concise ticket-formatter."},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = chat_completions_create_with_retry(
            messages=local_messages,
            model=OPENAI_MODEL,
            max_tokens=800,
            temperature=0.3
        )
        ai_text = response.choices[0].message.content.strip()
        ticket_json = safe_load_json(ai_text)
        if not ticket_json:
            fallback_title = f"{project_name} {icon}"
            if branch_name:
                fallback_title += f" [{branch_name}]"
            fallback_title += " " + rec_summary[:40]
            desc_ = f"Fallback:\n{rec_description}"
            return clamp_title_length(fallback_title), desc_

        # Extraer
        final_title = ticket_json.get("title", "")
        if branch_name and f"[{branch_name}]" not in final_title:
            # Insertar la rama si no está
            final_title += f" [{branch_name}]"

        # Añadimos icon si no está
        if icon not in final_title:
            final_title = f"{icon} {final_title}"

        # Clampeamos longitud
        final_title = clamp_title_length(final_title, MAX_TICKET_TITLE_LEN)

        adf_description = ticket_json.get("description", {})
        wiki_text = convert_adf_to_wiki(adf_description)
        if not wiki_text.strip():
            wiki_text = f"Short fallback:\n{rec_description}"
        return final_title, wiki_text
    except Exception as e:
        logger.error(f"Error en format_ticket_content: {e}")
        fallback_title = f"{project_name} {icon}"
        if branch_name:
            fallback_title += f" [{branch_name}]"
        fallback_title += " " + rec_summary[:40]
        fallback_title = clamp_title_length(fallback_title)
        fallback_desc = f"Fallback:\n{rec_description}"
        return fallback_title, fallback_desc

# =====================================================
# Recomendaciones (success logs)
# =====================================================
def parse_recommendations(ai_text: str) -> List[dict]:
    recommendations = []
    blocks = re.split(r"\n\s*-\s+", ai_text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue

        header_match = re.match(r"\*\*(.+?)\*\*\s*:?\s*(.*)", block, re.DOTALL)
        if header_match:
            title = header_match.group(1).strip()
            remaining_text = header_match.group(2).strip()
        else:
            title = block
            remaining_text = ""

        summary_match = re.search(r"(?i)Summary:\s*(.+?)(?=\n\s*-\s*\*Description\*|$)", remaining_text, re.DOTALL)
        description_match = re.search(r"(?i)Description:\s*(.+)", remaining_text, re.DOTALL)

        if summary_match:
            summary_text = summary_match.group(1).strip()
        else:
            lines_ = remaining_text.splitlines()
            summary_text = lines_[0].strip() if lines_ else ""

        if description_match:
            description_text = description_match.group(1).strip()
        else:
            lines_ = remaining_text.splitlines()
            if len(lines_) > 1:
                description_text = "\n".join(lines_[1:]).strip()
            else:
                description_text = ""

        full_summary = f"{title}: {summary_text}" if summary_text else title
        recommendations.append({"summary": full_summary, "description": description_text})

    return recommendations

def should_skip_recommendation(summary: str, description: str) -> bool:
    skip_keywords = [
        "bandit", "npm audit", "nancy", "scan-security-vulnerabilities",
        "check-code-format", "lint code", "owasp dependency check",
        "az storage", "azure storage"
    ]
    combined = f"{summary}\n{description}".lower()
    return any(kw in combined for kw in skip_keywords)

def analyze_logs_for_recommendations(log_dir: str, report_language: str, project_name: str) -> List[dict]:
    # ...
    # (Utiliza la misma lógica que ya tienes, 
    # solo ajustado a un # de lineas max, 
    # y no guardamos la respuesta en conversation_history
    # para reducir tokens).
    # 
    # Para brevedad, re-usamos la tuya, 
    # con un MAX_LOG_LINES menor, etc.
    pass  # <--- Adapta si deseas

# =====================================================
# Lógica para logs de fallos (failure)
# =====================================================
def analyze_logs_with_ai(
    log_dir: str,
    log_type: str,
    report_language: str,
    project_name: str,
    branch_name: str = ""
) -> (Optional[str], Optional[str], Optional[str]):
    # ...
    # Similar a tu "failure logs" 
    # con un prompt más corto, 
    # si no produce nada => fallback
    pass  # <--- Adáptalo a la versión final

# =====================================================
# Búsqueda de issues similares
# =====================================================
def find_similar_issues(
    jira: JIRA,
    project_key: str,
    new_summary: str,
    new_description: str,
    issue_type: str,
    jql_extra: str
) -> List[str]:
    # ...
    pass

# =====================================================
# Creación de tickets
# =====================================================
def create_jira_ticket_unified(
    jira: JIRA,
    jira_url: str,
    jira_user: str,
    jira_api_token: str,
    project_key: str,
    summary: str,
    description: str,
    issue_type: str
) -> Optional[str]:
    # ...
    pass

# =====================================================
# MAIN
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze logs & create JIRA tickets.")
    parser.add_argument("--jira-url", required=True)
    parser.add_argument("--jira-project-key", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--log-type", required=True, choices=["success","failure"])
    parser.add_argument("--report-language", default="English")
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--branch", required=False, default="", help="Branch name")

    args = parser.parse_args()

    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user_email = os.getenv("JIRA_USER_EMAIL")
    if not jira_api_token or not jira_user_email:
        logger.error("ERROR: Missing env vars JIRA_API_TOKEN or JIRA_USER_EMAIL.")
        sys.exit(1)

    init_conversation("You are an assistant that helps analyzing logs and creating Jira tickets. Keep messages short.")

    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    if args.log_type == "failure":
        # 1) Analizar logs
        summary, description, issue_type = analyze_logs_with_ai(
            args.log_dir, args.log_type, args.report_language,
            args.project_name, branch_name=args.branch
        )
        if not summary or not description:
            logger.error("No ticket => Análisis vacío.")
            return

        # 2) Validar type
        try:
            validate_issue_type(args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, issue_type)
        except Exception as e:
            logger.error(f"No se pudo validar type => {e}")
            return

        # 3) Duplicados
        jql_states = '"To Do", "In Progress", "Open", "Reopened"'
        existing_issues = find_similar_issues(
            jira, args.jira_project_key, summary, description, issue_type, f"status IN ({jql_states})"
        )
        if existing_issues:
            logger.info(f"Duplicado => {existing_issues}")
            return

        done_issues = find_similar_issues(
            jira, args.jira_project_key, summary, description, issue_type, 'statusCategory = Done'
        )

        # 4) Crear
        logger.info("Creando ticket de tipo 'Error' en Jira...")
        ticket_key = create_jira_ticket_unified(
            jira, args.jira_url, jira_user_email, jira_api_token,
            args.jira_project_key, summary, description, issue_type
        )
        if ticket_key:
            logger.info(f"Ticket creado => {ticket_key}")
            if done_issues:
                duplicates_str = ", ".join(done_issues)
                comment_body = get_repeated_incident_comment(duplicates_str, args.report_language)
                try:
                    jira.add_comment(ticket_key, comment_body)
                except Exception as ex:
                    logger.warning(f"No se pudo añadir comentario => {ex}")
                for old_key in done_issues:
                    try:
                        jira.create_issue_link(
                            type="Relates",
                            inwardIssue=ticket_key,
                            outwardIssue=old_key
                        )
                    except Exception as ex:
                        logger.warning(f"No se pudo crear enlace => {ex}")
        else:
            logger.error("ERROR => creación fallida.")
    else:
        # success => recomendaciones
        recommendations = analyze_logs_for_recommendations(args.log_dir, args.report_language, args.project_name)
        if not recommendations:
            logger.info("No hay recomendaciones.")
            return

        issue_type = "Tarea"
        for i, rec in enumerate(recommendations, start=1):
            r_summary = rec["summary"]
            r_desc = rec["description"]
            if not r_desc.strip():
                continue
            if should_skip_recommendation(r_summary, r_desc):
                continue

            jql_states = '"To Do", "In Progress", "Open", "Reopened"'
            existing_issues = find_similar_issues(
                jira, args.jira_project_key, r_summary, r_desc, issue_type,
                f"status IN ({jql_states})"
            )
            if existing_issues:
                continue

            discard_issues = find_similar_issues(
                jira, args.jira_project_key, r_summary, r_desc, issue_type,
                'status IN ("DESCARTADO")'
            )
            if discard_issues:
                continue

            # 2) Generar title & desc for ticket
            final_title, wiki_desc = format_ticket_content(
                args.project_name,
                args.branch,
                r_summary,
                r_desc,
                "Improvement"
            )
            if not wiki_desc.strip():
                wiki_desc = f"Short fallback => {r_desc}"

            new_key = create_jira_ticket_unified(
                jira, args.jira_url, jira_user_email, jira_api_token,
                args.jira_project_key, final_title, wiki_desc, issue_type
            )
            if new_key:
                logger.info(f"Recomendación #{i} => Creado ticket => {new_key}")
            else:
                logger.error(f"Recomendación #{i} => Fallida creación.")


def get_repeated_incident_comment(duplicates_str: str, language: str) -> str:
    if language.lower().startswith("es"):
        return (
            f"Se han encontrado incidencias previas similares: {duplicates_str}. "
            "Por favor, verificar si se relacionan con esta nueva."
        )
    else:
        return (
            f"Similar incidents found: {duplicates_str}. "
            "Please check if this new issue is related to them."
        )

if __name__ == "__main__":
    main()
