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

# ======================================================
# CONFIGURACIÓN DE LOGGING (formato simplificado)
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURACIÓN GLOBAL =====================
OPENAI_MODEL = "gpt-4o"
MAX_CHAR_PER_REQUEST = 20000
BANDIT_JSON_NAME = "bandit-output.json"
MAX_FILE_SIZE_MB = 2.0
ALLOWED_EXTENSIONS = (".log", ".sarif")

# ======================================================
# Parámetros de reintentos y manejo de historial
# ======================================================
MAX_RETRIES = 5                # Reintentos ante 429
BASE_DELAY = 1.0               # Espera inicial (seg)
MAX_CONVERSATION_TOKENS = 4000 # Máx tokens totales en historial

# ======================================================
# CONFIGURACIÓN OPENAI
# ======================================================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("ERROR: 'OPENAI_API_KEY' is not set.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ======================================================
# HISTORIAL DE CONVERSACIÓN (REDUCIDO)
# ======================================================
conversation_history: List[Dict[str, str]] = []

def init_conversation(system_content: str):
    """
    Inicia historial con un mensaje de rol system.
    Si no necesitas múltiples interacciones,
    esta parte no es crucial, pero la conservamos.
    """
    global conversation_history
    conversation_history = [{"role": "system", "content": system_content}]

def add_user_message(user_content: str):
    """
    Añade un mensaje de rol user.
    """
    global conversation_history
    conversation_history.append({"role": "user", "content": user_content})

def ensure_history_not_excessive():
    """
    Recorta historial si supera cierto número de tokens.
    """
    total_tokens = sum(len(m["content"].split()) for m in conversation_history)
    if total_tokens > MAX_CONVERSATION_TOKENS:
        logger.info("** Re-inicializando el historial para evitar exceso de tokens **")
        system_msg = conversation_history[0]
        conversation_history.clear()
        conversation_history.append(system_msg)

# ======================================================
# REINTENTOS CON BACKOFF
# ======================================================
def chat_completions_create_with_retry(
    messages: List[Dict[str, str]],
    model: str = OPENAI_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> Any:
    """
    Llama a la API de OpenAI con reintentos automáticos si hay errores 429.
    """
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

        except openai.error.RateLimitError:
            if attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI RateLimitError (429). Reintentando en %.1f seg. (Intento %d/%d)",
                    wait_time, attempt + 1, MAX_RETRIES
                )
                time.sleep(wait_time)
            else:
                logger.error("Se agotaron los reintentos tras 429. Abortando.")
                raise

        except openai.error.APIError as e:
            if getattr(e, "http_status", None) == 429 and attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI APIError 429. Reintentando en %.1f seg. (Intento %d/%d)",
                    wait_time, attempt + 1, MAX_RETRIES
                )
                time.sleep(wait_time)
            else:
                logger.error("APIError no recuperable o reintentos agotados: %s", e)
                raise

        except Exception as ex:
            logger.error("Error no controlado en la llamada a OpenAI: %s", ex)
            raise

    raise RuntimeError("Se agotaron todos los reintentos en chat_completions_create_with_retry.")

# ======================================================
# CONEXIÓN A JIRA
# ======================================================
def connect_to_jira(jira_url: str, jira_user: str, jira_api_token: str) -> JIRA:
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    logger.info("Conexión establecida con Jira.")
    return jira

# ======================================================
# FUNCIONES DE SANITIZACIÓN
# ======================================================
def sanitize_summary(summary: str) -> str:
    summary = summary.replace("\n", " ").replace("\r", " ")
    sanitized = "".join(c for c in summary if c.isalnum() or c.isspace() or c in "-_:,./()[]{}")
    # Forzamos límite 255
    return sanitized[:255].strip()

def preprocess_text(text: str) -> str:
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    return text_no_punct.strip().lower()

def calculate_similarity(text1: str, text2: str) -> float:
    from difflib import SequenceMatcher
    ratio = SequenceMatcher(None, preprocess_text(text1), preprocess_text(text2)).ratio()
    return ratio

# ======================================================
# MÉTODOS AUXILIARES: JSON devuelto por IA, chunking
# ======================================================
def safe_load_json(ai_output: str) -> Optional[dict]:
    if ai_output.startswith("```"):
        lines = ai_output.splitlines()
        if lines and lines[0].strip().startswith("```"):
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

def chunk_content_if_needed(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

# ======================================================
# VALIDACIÓN DE DIRECTORIO DE LOGS
# ======================================================
def validate_logs_directory(log_dir: str) -> List[str]:
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")

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
        raise FileNotFoundError(f"ERROR: No valid files found in '{log_dir}'.")
    return log_files

# ======================================================
# CONVERTIR ADF -> WIKI
# ======================================================
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

        # Si es otro tipo, procesar recursivamente
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

# ======================================================
# FORMATEO DE TICKET (proyecto+rama+emoticono)
# ======================================================
MAX_TICKET_TITLE_LEN = 100

def clamp_title_length(title: str, max_len: int = MAX_TICKET_TITLE_LEN) -> str:
    """ Recorta la longitud del title. """
    if len(title) > max_len:
        return title[:max_len].rstrip()
    return title

def format_ticket_content(
    project_name: str,
    branch_name: str,
    rec_summary: str,
    rec_description: str,
    ticket_category: str
) -> (str, str):
    """
    Llama a la IA para formatear 'title' + 'description' en ADF => wiki.
    Si IA falla, fallback manual.
    """
    if ticket_category.lower() in ("improvement", "tarea"):
        icon = "💡"
    else:
        icon = "🔥"

    system_msg = "You are a professional ticket formatter."
    user_msg = (
        f"Project: {project_name}\n"
        f"Branch: {branch_name}\n"
        f"Recommendation Title: {rec_summary}\n"
        f"Recommendation Details: {rec_description}\n"
        f"Category: {ticket_category}\n"
        "Return JSON with 'title' and 'description'. "
        "Title must start with the project name, show the branch in brackets if not empty, "
        "and an emoji (🔥 or 💡). Description must be minimal ADF. Return only JSON, no extra text."
    )

    local_messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]
    import openai
    try:
        response = chat_completions_create_with_retry(
            messages=local_messages,
            model=OPENAI_MODEL,
            max_tokens=800,
            temperature=0.3
        )
        ai_output = response.choices[0].message.content.strip()

        ticket_json = safe_load_json(ai_output)
        if not ticket_json:
            fallback_title = f"{project_name} {icon}"
            if branch_name:
                fallback_title += f" [{branch_name}]"
            fallback_title += " " + rec_summary[:40]
            fallback_title = clamp_title_length(fallback_title)

            fallback_desc = f"Fallback description:\n{rec_description}"
            return fallback_title, fallback_desc

        final_title = ticket_json.get("title", "")
        # Asegurar el emoticon
        if icon not in final_title:
            final_title = f"{icon} {final_title}"
        # Recorta a 100
        final_title = clamp_title_length(final_title, MAX_TICKET_TITLE_LEN)

        adf_description = ticket_json.get("description", {})
        wiki_text = convert_adf_to_wiki(adf_description)
        if not wiki_text.strip():
            wiki_text = f"Short fallback:\n{rec_description}"

        return final_title, wiki_text

    except Exception as e:
        logger.error("Error en format_ticket_content: %s", e)
        fallback_title = f"{project_name} {icon}"
        if branch_name:
            fallback_title += f" [{branch_name}]"
        fallback_title += " " + rec_summary[:40]
        fallback_title = clamp_title_length(fallback_title)

        fallback_desc = f"Fallback:\n{rec_description}"
        return fallback_title, fallback_desc

# ======================================================
# RECOMENDACIONES PARA LOGS DE ÉXITO
# ======================================================
MAX_LOG_LINES = 150

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

        summary_match = re.search(
            r"(?i)Summary:\s*(.+?)(?=\n\s*-\s*\*Description\*|$)",
            remaining_text, re.DOTALL
        )
        description_match = re.search(
            r"(?i)Description:\s*(.+)",
            remaining_text, re.DOTALL
        )

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
    log_files = validate_logs_directory(log_dir)
    combined_text = []
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:MAX_LOG_LINES]
                combined_text.extend(lines)
        except UnicodeDecodeError:
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        return []

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)

    prompt_base = (
        "You are a code reviewer specialized in Python. "
        "Produce recommended improvements with minimal text. "
        "Format: - **Title**: Summary\nDescription. "
        f"Write in {report_language}."
    )

    all_recommendations = []
    for chunk in text_chunks:
        prompt = f"{prompt_base}\nLogs (truncated):\n{chunk}"
        try:
            add_user_message(prompt)
            ensure_history_not_excessive()
            response = chat_completions_create_with_retry(
                messages=conversation_history,
                model=OPENAI_MODEL,
                max_tokens=700,
                temperature=0.3
            )
            ai_text = response.choices[0].message.content.strip()
            # No agregamos al historial para no crecer
            recs = parse_recommendations(ai_text)
            all_recommendations.extend(recs)
        except Exception as e:
            logger.warning("Fallo analizando logs para recomendaciones: %s", e)
            continue
    return all_recommendations

# ======================================================
# LOGS DE FALLA
# ======================================================
def analyze_logs_with_ai(
    log_dir: str,
    log_type: str,
    report_language: str,
    project_name: str,
    branch_name: str = ""
) -> (Optional[str], Optional[str], Optional[str]):

    log_files = validate_logs_directory(log_dir)
    combined_text = []
    error_lines = []

    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines_ = f.read().splitlines()[:MAX_LOG_LINES]
                combined_text.extend(lines_)
                for ln in lines_:
                    if any(k in ln for k in ("ERROR", "Exception", "Traceback")):
                        error_lines.append(ln.strip())
        except UnicodeDecodeError:
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        return None, None, None

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)

    prompt_base = (
        f"You are a technical writer creating a concise Jira Cloud ticket from logs. "
        f"Write in {report_language}. Keep it short, minimal Markdown. Title + brief summary."
    )

    error_context = ""
    if error_lines:
        error_context = "\nSome error lines:\n" + "\n".join(error_lines[:5])

    # Tomamos solo el primer chunk
    chunk = text_chunks[0]
    final_prompt = f"{prompt_base}\n\nLogs:\n{chunk}{error_context}"

    try:
        add_user_message(final_prompt)
        ensure_history_not_excessive()

        response = chat_completions_create_with_retry(
            messages=conversation_history,
            model=OPENAI_MODEL,
            max_tokens=400,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
        lines_ = summary.splitlines()

        if not lines_:
            return None, None, None

        first_line = lines_[0].strip()
        match = re.match(r"(?i)^(?:title|summary)\s*:\s*(.*)$", first_line)
        if match:
            extracted_title = match.group(1).strip()
            lines_ = lines_[1:]
        else:
            extracted_title = first_line

        # Emoticon aleatorio para error
        icon = random.choice(["🐞", "🔥", "💥", "🐛", "⛔", "🚫"])
        if branch_name:
            extracted_title += f" [branch: {branch_name}]"

        final_title = f"{project_name} {icon} {extracted_title}"
        remaining_desc = "\n".join(lines_).strip()
        if not remaining_desc:
            remaining_desc = summary

        description_plain = remaining_desc.replace("\t", " ")
        description_plain = re.sub(r"\*\*", "*", description_plain)

        return final_title, description_plain, "Error"

    except Exception as e:
        logger.error("Error analizando logs con IA: %s", e)
        return None, None, None

# ======================================================
# FUNCIÓN PARA BUSCAR ISSUES DUPLICADAS
# ======================================================
def find_similar_issues(
    jira: JIRA,
    project_key: str,
    new_summary: str,
    new_description: str,
    issue_type: str,
    jql_extra: str
) -> List[str]:

    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.85

    jql_issue_type = "Task" if issue_type.lower() == "tarea" else issue_type
    jql_query = f'project = "{project_key}" AND issuetype = "{jql_issue_type}"'
    if jql_extra:
        jql_query += f" AND {jql_extra}"

    logger.info("Buscando tickets con la siguiente JQL: %s", jql_query)
    issues = jira.search_issues(jql_query, maxResults=1000)

    matched_keys = []
    import openai

    for issue in issues:
        existing_summary = issue.fields.summary or ""
        existing_description = issue.fields.description or ""
        summary_sim = calculate_similarity(new_summary, existing_summary)
        desc_sim = calculate_similarity(new_description, existing_description)

        logger.info(
            "Comparando con issue %s => summarySim=%.2f, descSim=%.2f",
            issue.key, summary_sim, desc_sim
        )

        # Muy bajo => skip
        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            continue

        # Muy alto => duplicado inmediato
        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            logger.info("Similitud alta con %s => duplicado inmediato.", issue.key)
            return [issue.key]

        # Rango medio => IA
        short_prompt = (
            "Check if these two issues are the same.\n\n"
            f"Existing summary: {existing_summary}\n"
            f"Existing description: {existing_description}\n\n"
            f"New summary: {new_summary}\n"
            f"New description: {new_description}\n\n"
            "Respond only 'yes' or 'no'."
        )
        local_messages = [
            {"role": "system", "content": "You are a short text similarity checker."},
            {"role": "user", "content": short_prompt}
        ]
        try:
            resp = chat_completions_create_with_retry(
                messages=local_messages,
                model=OPENAI_MODEL,
                max_tokens=50,
                temperature=0.0
            )
            ai_resp = resp.choices[0].message.content.strip().lower()
            if ai_resp.startswith("yes"):
                logger.info("IA dice que el nuevo ticket coincide con el issue %s => duplicado.", issue.key)
                matched_keys.append(issue.key)

        except Exception as ex:
            logger.warning("Fallo la IA al comparar con %s: %s", issue.key, ex)
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                logger.info("Fallback local => duplicado con %s", issue.key)
                matched_keys.append(issue.key)

    return matched_keys

# ======================================================
# CREACIÓN DE TICKETS
# ======================================================
def create_jira_ticket(
    jira: JIRA,
    project_key: str,
    summary: str,
    description: str,
    issue_type: str
) -> Optional[str]:

    summary = sanitize_summary(summary)
    if not description.strip():
        return None
    try:
        fields_ = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
        logger.info("Intentando crear ticket con summary='%s' y issueType='%s'...", summary, issue_type)
        issue = jira.create_issue(fields=fields_)
        logger.info("Ticket creado con la librería de Jira => key='%s'", issue.key)
        return issue.key
    except Exception as e:
        logger.error("Error creando ticket con la librería de Jira: %s", e)
        return None

def create_jira_ticket_via_requests(
    jira_url: str,
    jira_user: str,
    jira_api_token: str,
    project_key: str,
    summary: str,
    description: str,
    issue_type: str
) -> Optional[str]:

    summary = sanitize_summary(summary)
    if not description.strip():
        return None

    fallback_adf = {
        "version": 1,
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": description.replace('\n', ' ')}
                ]
            }
        ]
    }
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": fallback_adf,
            "issuetype": {"name": issue_type}
        }
    }
    url = f"{jira_url}/rest/api/3/issue"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)

    logger.info("Intentando crear ticket con 'requests' => summary='%s' ...", summary)
    try:
        response = requests.post(url, json=payload, headers=headers, auth=auth)
        if response.status_code == 201:
            key = response.json().get("key")
            logger.info("Ticket creado vía requests => key='%s'", key)
            return key
        else:
            logger.error("Falló la creación vía requests: %s - %s", response.status_code, response.text)
            return None
    except Exception as ex:
        logger.error("Error con requests al crear ticket: %s", ex)
        return None

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
    """
    Primero intenta crear con la librería oficial de Jira;
    si falla o no retorna key, hace fallback con requests.
    """
    key_ = create_jira_ticket(jira, project_key, summary, description, issue_type)
    if key_:
        return key_

    logger.info("Fallo con librería Jira => intentando fallback con requests...")
    return create_jira_ticket_via_requests(
        jira_url, jira_user, jira_api_token,
        project_key, summary, description, issue_type
    )

# ======================================================
# VALIDAR TIPO DE ISSUE
# ======================================================
def validate_issue_type(jira_url: str, jira_user: str, jira_api_token: str,
                        project_key: str, issue_type: str) -> None:
    url = f"{jira_url}/rest/api/3/issue/createmeta?projectKeys={project_key}"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)
    r = requests.get(url, headers=headers, auth=auth)
    if r.status_code == 200:
        valid_types = [it["name"] for it in r.json()["projects"][0]["issuetypes"]]
        if issue_type not in valid_types:
            raise ValueError(f"Invalid issue type: '{issue_type}'. Valid: {valid_types}")
    else:
        raise Exception(f"Failed to fetch issue types: {r.status_code} - {r.text}")

# ======================================================
# COMENTARIO DE INCIDENTES REPETIDOS
# ======================================================
def get_repeated_incident_comment(duplicates_str: str, language: str) -> str:
    if language.lower().startswith("es"):
        return (
            f"Se han encontrado incidencias previas similares: {duplicates_str}. "
            "Por favor, verificar si se relacionan con esta nueva."
        )
    else:
        return (
            f"Similar incidents found: {duplicates_str}. "
            "Please check if this new issue is related to any of them."
        )

# ======================================================
# MAIN
# ======================================================
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
    parser.add_argument("--branch", required=False, default="", help="Nombre de la rama actual")

    args = parser.parse_args()

    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user_email = os.getenv("JIRA_USER_EMAIL")
    if not jira_api_token or not jira_user_email:
        logger.error("ERROR: Missing env vars JIRA_API_TOKEN or JIRA_USER_EMAIL.")
        sys.exit(1)

    # Iniciamos el historial con un mensaje system
    init_conversation("You are an assistant that helps analyzing logs and creating Jira tickets. Keep messages short.")

    # Conectar a Jira
    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    # Procesar 'failure'
    if args.log_type == "failure":
        summary, description, issue_type = analyze_logs_with_ai(
            args.log_dir, args.log_type, args.report_language,
            args.project_name, branch_name=args.branch
        )
        if not summary or not description:
            logger.error("No ticket => Análisis vacío. No se creará un ticket.")
            return

        try:
            validate_issue_type(args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, issue_type)
        except Exception as e:
            logger.error("Tipo de issue no válido: %s", e)
            return

        logger.info("Verificando si existe duplicado en estado abierto (To Do, In Progress, etc.)...")
        jql_states = '"To Do", "In Progress", "Open", "Reopened"'
        existing_issues = find_similar_issues(
            jira, args.jira_project_key, summary, description, issue_type, f"status IN ({jql_states})"
        )
        if existing_issues:
            logger.info(f"Se detectan duplicados => {existing_issues}. No se creará un ticket nuevo.")
            return
        else:
            logger.info("No se encontraron duplicados abiertos. Se procederá con la creación de un ticket nuevo.")

        logger.info("Verificando si existe duplicado en estado Done/Closed...")
        done_issues = find_similar_issues(
            jira, args.jira_project_key, summary, description, issue_type, 'statusCategory = Done'
        )
        if done_issues:
            logger.info(f"Se detectan issues finalizadas similares => {done_issues} (no impide crear ticket).")

        logger.info("Creando ticket de tipo 'Error' en Jira con summary='%s'...", summary)
        ticket_key = create_jira_ticket_unified(
            jira, args.jira_url, jira_user_email, jira_api_token,
            args.jira_project_key, summary, description, issue_type
        )
        if ticket_key:
            logger.info(f"Ticket NUEVO creado => '{ticket_key}'")
            if done_issues:
                duplicates_str = ", ".join(done_issues)
                comment_body = get_repeated_incident_comment(duplicates_str, args.report_language)
                logger.info("Añadiendo comentario sobre los duplicados finalizados => %s", duplicates_str)
                try:
                    jira.add_comment(ticket_key, comment_body)
                except Exception as ex:
                    logger.warning(f"No se pudo añadir comentario => {ex}")
                for old_key in done_issues:
                    logger.info(f"Creando link 'Relates' entre '{ticket_key}' y '{old_key}'")
                    try:
                        jira.create_issue_link(
                            type="Relates",
                            inwardIssue=ticket_key,
                            outwardIssue=old_key
                        )
                    except Exception as ex:
                        logger.warning(f"No se pudo crear enlace => {ex}")
        else:
            logger.error("ERROR => creación de ticket fallida (no se obtuvo key).")

    # Procesar 'success'
    else:
        recommendations = analyze_logs_for_recommendations(
            args.log_dir, args.report_language, args.project_name
        )
        if not recommendations:
            logger.info("No hay recomendaciones derivadas de logs => no se crearán tickets.")
            return

        issue_type = "Tarea"
        for i, rec in enumerate(recommendations, start=1):
            r_summary = rec["summary"]
            r_desc = rec["description"]

            # Omitir si no hay contenido
            if not r_desc.strip():
                logger.debug("Recomendación #%d sin contenido. Saltando...", i)
                continue
            if should_skip_recommendation(r_summary, r_desc):
                logger.debug("Recomendación #%d con keywords restringidos. Saltando...", i)
                continue

            logger.info("Revisión duplicados para Recomendación #%d => '%s'...", i, r_summary)
            jql_states = '"To Do", "In Progress", "Open", "Reopened"'
            existing_issues = find_similar_issues(
                jira, args.jira_project_key, r_summary, r_desc, issue_type, f"status IN ({jql_states})"
            )
            if existing_issues:
                logger.info(f"Recomendación #%d duplicada con => %s. No se crea ticket nuevo.", i, existing_issues)
                continue

            discard_issues = find_similar_issues(
                jira, args.jira_project_key, r_summary, r_desc, issue_type, 'status IN ("DESCARTADO")'
            )
            if discard_issues:
                logger.info(f"Recomendación #%d coincide con tickets descartados => %s. No se crea ticket nuevo.", i, discard_issues)
                continue

            # Generar título + descripción en wiki
            final_title, wiki_desc = format_ticket_content(
                args.project_name,
                args.branch,
                r_summary,
                r_desc,
                "Improvement"
            )
            if not wiki_desc.strip():
                wiki_desc = f"Short fallback => {r_desc}"

            logger.info("Creando ticket de 'Tarea' para la recomendación #%d con summary='%s'...", i, final_title)
            new_key = create_jira_ticket_unified(
                jira, args.jira_url, jira_user_email, jira_api_token,
                args.jira_project_key, final_title, wiki_desc, issue_type
            )
            if new_key:
                logger.info(f"Recomendación #{i} => Creado ticket => '{new_key}'")
            else:
                logger.error(f"Recomendación #{i} => Fallida la creación de ticket (no se obtuvo key).")

if __name__ == "__main__":
    main()
