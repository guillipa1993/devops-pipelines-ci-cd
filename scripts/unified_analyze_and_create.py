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
import openai.error  # <-- para capturar RateLimitError o APIError
from typing import List, Optional, Dict, Any

# ======================================================
# CONFIGURACIÓN DE LOGGING
# ======================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== CONFIGURACIÓN GLOBAL =====================
OPENAI_MODEL = "gpt-4o" 
MAX_CHAR_PER_REQUEST = 20000
BANDIT_JSON_NAME = "bandit-output.json"
MAX_FILE_SIZE_MB = 2.0
ALLOWED_EXTENSIONS = (".log", ".sarif")

# Parámetros de reintentos
MAX_RETRIES = 5           # Número máximo de reintentos
BASE_DELAY = 1.0          # Espera base (segundos) para backoff exponencial

# ===================== CONFIGURACIÓN OPENAI =====================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("ERROR: 'OPENAI_API_KEY' is not set.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ------------------------------------------------
# HISTORIAL DE CONVERSACIÓN (SE MANTIENE EN MEMORIA)
# ------------------------------------------------
conversation_history: List[Dict[str, str]] = []
"""
Ejemplo de estructura de conversation_history:
[
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hola, IA."},
  {"role": "assistant", "content": "¡Hola! ¿En qué puedo ayudarte?"},
  ...
]
"""

def add_system_message(system_content: str) -> None:
    """
    Agrega un mensaje 'system' al historial de conversación.
    """
    conversation_history.append({"role": "system", "content": system_content})

def add_user_message(user_content: str) -> None:
    """
    Agrega un mensaje 'user' al historial de conversación.
    """
    conversation_history.append({"role": "user", "content": user_content})

def add_assistant_message(assistant_content: str) -> None:
    """
    Agrega un mensaje 'assistant' al historial de conversación.
    """
    conversation_history.append({"role": "assistant", "content": assistant_content})

def chat_completions_create_with_retry(
    messages: List[Dict[str, Any]],
    model: str = OPENAI_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 1000,
) -> Any:
    """
    Llama a client.chat.completions.create con reintentos automáticos.
    Aplica backoff exponencial si ocurre un error 429 (RateLimitError).
    """
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
            # Error específico de límite de OpenAI (429)
            if attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI RateLimitError (429). Reintentando en %.1f segundos. (Intento %d/%d)",
                    wait_time, attempt + 1, MAX_RETRIES
                )
                time.sleep(wait_time)
            else:
                logger.error("Se alcanzó el número máximo de reintentos tras un 429. Abortando.")
                raise

        except openai.error.APIError as e:
            # Otros errores de API que incluyen 429
            if e.http_status == 429 and attempt < MAX_RETRIES - 1:
                wait_time = BASE_DELAY * (2 ** attempt)
                logger.warning(
                    "OpenAI APIError 429. Reintentando en %.1f seg. (Intento %d/%d)",
                    wait_time, attempt + 1, MAX_RETRIES
                )
                time.sleep(wait_time)
            else:
                logger.error("APIError no recuperable o reintentos agotados: %s", e)
                raise

        except Exception as e:
            # Cualquier otro error se lanza sin reintentar
            logger.error("Excepción no controlada en la llamada a OpenAI: %s", e)
            raise

    raise RuntimeError("Agotados los reintentos en chat_completions_create_with_retry.")

def openai_chat_with_history(
    user_prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1000
) -> str:
    """
    1. Agrega un mensaje 'system' (opcional) y un mensaje 'user' al historial.
    2. Llama a la API con el historial completo.
    3. Agrega la respuesta ('assistant') al historial.
    4. Retorna el texto que devolvió la IA.
    """
    # Si se requiere un nuevo system_prompt, lo agregamos
    if system_prompt:
        add_system_message(system_prompt)

    # Agregamos el mensaje del usuario
    add_user_message(user_prompt)

    # Llamamos la API con TODO el historial
    response = chat_completions_create_with_retry(
        messages=conversation_history,
        model=OPENAI_MODEL,
        temperature=temperature,
        max_tokens=max_tokens
    )
    assistant_reply = response.choices[0].message.content.strip()

    # Guardamos la respuesta
    add_assistant_message(assistant_reply)

    return assistant_reply

# ===================== CONEXIÓN A JIRA =====================
def connect_to_jira(jira_url: str, jira_user: str, jira_api_token: str) -> JIRA:
    """
    Conecta a Jira usando la librería oficial de Python.
    """
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    logger.info("Conexión establecida con Jira.")
    return jira

# ===================== FUNCIONES DE SANITIZACIÓN =====================
def sanitize_summary(summary: str) -> str:
    """
    Sanitiza la cadena de resumen para Jira,
    eliminando caracteres conflictivos y limitando la longitud a 255.
    """
    summary = summary.replace("\n", " ").replace("\r", " ")
    sanitized = "".join(c for c in summary if c.isalnum() or c.isspace() or c in "-_:,./()[]{}")
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized.strip()

def preprocess_text(text: str) -> str:
    """
    Elimina puntuación y pasa a minúsculas para comparar texto.
    """
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    return text_no_punct.strip().lower()

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calcula la similitud entre dos cadenas (0.0 a 1.0).
    """
    ratio = SequenceMatcher(None, preprocess_text(text1), preprocess_text(text2)).ratio()
    return ratio

# ===================== CONVERSIÓN A WIKI (ADF -> Jira) =====================
def convert_adf_to_wiki(adf: dict) -> str:
    """
    Convierte un objeto ADF (Atlassian Document Format) a texto wiki.
    """
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
            lines = []
            for item in content:
                item_text = ""
                for child in item.get("content", []):
                    if child.get("type") == "paragraph":
                        for subchild in child.get("content", []):
                            if subchild.get("type") == "text":
                                item_text += subchild.get("text", "")
                lines.append(f"* {item_text.strip()}")
            return "\n".join(lines) + "\n\n"

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

# ===================== PARSEO DE RECOMENDACIONES =====================
def parse_recommendations(ai_text: str) -> List[dict]:
    """
    Parsea recomendaciones de la respuesta de la IA,
    esperando formato con '- **Titulo**: Summary...\nDescription...'.
    Devuelve una lista de dicts con 'summary' y 'description'.
    """
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
            lines = remaining_text.splitlines()
            summary_text = lines[0].strip() if lines else ""

        if description_match:
            description_text = description_match.group(1).strip()
        else:
            lines = remaining_text.splitlines()
            if len(lines) > 1:
                description_text = "\n".join(lines[1:]).strip()
            else:
                description_text = ""

        full_summary = f"{title}: {summary_text}" if summary_text else title
        recommendations.append({"summary": full_summary, "description": description_text})

    return recommendations

# ===================== ÍCONOS =====================
IMPROVEMENT_ICONS = ["🚀", "💡", "🔧", "🤖", "🌟", "📈", "✨"]
ERROR_ICONS = ["🐞", "🔥", "💥", "🐛", "⛔", "🚫"]

def choose_improvement_icon() -> str:
    """
    Retorna un icono aleatorio para mejoras.
    """
    return random.choice(IMPROVEMENT_ICONS)

def choose_error_icon() -> str:
    """
    Retorna un icono aleatorio para errores.
    """
    return random.choice(ERROR_ICONS)

# ===================== FILTRAR RECOMENDACIONES =====================
def should_skip_recommendation(summary: str, description: str) -> bool:
    """
    Retorna True si la recomendación debe omitirse por contener keywords restringidas.
    """
    skip_keywords = [
        "bandit", "npm audit", "nancy", "scan-security-vulnerabilities",
        "check-code-format", "lint code", "owasp dependency check",
        "az storage", "azure storage"
    ]
    combined = f"{summary}\n{description}".lower()
    return any(kw in combined for kw in skip_keywords)

# ===================== CHUNKING =====================
def chunk_content_if_needed(text: str, max_chars: int) -> List[str]:
    """
    Divide el texto en partes de longitud <= max_chars.
    """
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks

# ===================== Manejo de resultados de OpenAI =====================
def safe_load_json(ai_output: str) -> Optional[dict]:
    """
    Intenta parsear 'ai_output' como JSON.
    1. Elimina triple backticks si están presentes.
    2. Hace json.loads().
    3. Si falla, retorna None.
    """
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
            cleaned = cleaned[: last_brace+1]
        try:
            return json.loads(cleaned)
        except:
            return None

# ===================== FORMAT TICKET CONTENT =====================
def format_ticket_content(
    project_name: str,
    rec_summary: str,
    rec_description: str,
    ticket_category: str
) -> (str, str):
    """
    Dado un project_name, resumen, descripción y categoría,
    llama a OpenAI para formatear un ticket. Luego convierte ADF -> Wiki.
    Si falla el parseo JSON, retorna un fallback.
    """
    if ticket_category.lower() in ("improvement", "tarea"):
        icon = choose_improvement_icon()
    else:
        icon = choose_error_icon()

    prompt = (
        "You are a professional technical writer formatting Jira tickets for developers. "
        "Given the following recommendation details, produce a JSON object with two keys: 'title' and 'description'.\n\n"
        f"- The 'title' must be a single concise sentence that starts with the project name as a prefix and includes "
        f"an appropriate emoticon for {ticket_category} (choose from: {IMPROVEMENT_ICONS + ERROR_ICONS}).\n\n"
        "- The 'description' must be a valid Atlassian Document Format (ADF) object with code blocks using triple backticks.\n"
        "Do not include labels like 'Summary:' or 'Description:' in the output.\n\n"
        f"Project: {project_name}\n"
        f"Recommendation Title: {rec_summary}\n"
        f"Recommendation Details: {rec_description}\n"
        f"Ticket Category: {ticket_category}\n\n"
        "Return only a valid JSON object."
    )

    try:
        # Llamamos a la IA con historial. 
        # Añadimos un "system_prompt" para que sea un rol system distinto, y un user_prompt con la solicitud.
        ai_output = openai_chat_with_history(
            user_prompt=prompt,
            system_prompt="You are a professional technical writer."
        )

        ticket_json = safe_load_json(ai_output)
        if not ticket_json:
            fallback_summary = sanitize_summary(rec_summary)
            fallback_summary = f"{icon} {fallback_summary}"
            fallback_desc = f"Fallback description:\n\n{rec_description}"
            return fallback_summary, fallback_desc

        final_title = ticket_json.get("title", "")
        adf_description = ticket_json.get("description", {})

        if not any(ic in final_title for ic in (IMPROVEMENT_ICONS + ERROR_ICONS)):
            final_title = f"{icon} {final_title}"

        if len(final_title) > 255:
            final_title = final_title[:255]

        wiki_text = convert_adf_to_wiki(adf_description)
        return final_title, wiki_text

    except Exception as e:
        logger.error("Error en format_ticket_content: %s", e)
        fallback_summary = sanitize_summary(rec_summary)
        fallback_summary = f"{icon} {fallback_summary}"
        wiki_text = f"Fallback description:\n\n{rec_description}"
        return fallback_summary, wiki_text

# ===================== FUNCIÓN UNIFICADA PARA BUSCAR ISSUES =====================
def find_similar_issues(
    jira: JIRA,
    project_key: str,
    new_summary: str,
    new_description: str,
    issue_type: str,
    jql_extra: str
) -> List[str]:
    """
    Busca en Jira (mediante JQL) issues que puedan ser similares
    a un nuevo summary/description. Usa:
      - Comparación local (SequenceMatcher).
      - Comparación con IA (si la similitud local es moderada).
    Retorna la lista de keys encontradas como similares.
    
    :param jira: Instancia conectada a JIRA
    :param project_key: Proyecto a consultar
    :param new_summary: Resumen del nuevo ticket
    :param new_description: Descripción del nuevo ticket
    :param issue_type: Tipo de issue (e.g. "Error" o "Task")
    :param jql_extra: Cláusula extra para filtrar (e.g. 'status IN ("To Do")')
    :return: Lista de issue keys que se consideren similares
    """
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9

    jql_issue_type = "Task" if issue_type.lower() == "tarea" else issue_type
    jql_query = f'project = "{project_key}" AND issuetype = "{jql_issue_type}"'
    if jql_extra:
        jql_query += f" AND {jql_extra}"

    logger.info("Buscando tickets con la siguiente JQL: %s", jql_query)
    issues = jira.search_issues(jql_query, maxResults=1000)

    matched_keys = []
    for issue in issues:
        existing_summary = issue.fields.summary or ""
        existing_description = issue.fields.description or ""
        summary_sim = calculate_similarity(new_summary, existing_summary)
        desc_sim = calculate_similarity(new_description, existing_description)

        logger.info(
            "Comparando con issue %s => summarySim=%.2f, descSim=%.2f",
            issue.key, summary_sim, desc_sim
        )

        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            continue
        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            logger.info("La similitud con %s supera %.1f. Lo consideramos duplicado inmediato.", issue.key, LOCAL_SIM_HIGH)
            return [issue.key]

        try:
            # Preparamos un prompt breve para la IA
            check_prompt = (
                "We have two issues:\n\n"
                f"Existing issue:\nSummary: {existing_summary}\nDescription: {existing_description}\n\n"
                f"New issue:\nSummary: {new_summary}\nDescription: {new_description}\n\n"
                "Do they represent essentially the same issue? Respond 'yes' or 'no'."
            )

            ai_result = openai_chat_with_history(
                user_prompt=check_prompt,
                system_prompt="You are an assistant specialized in analyzing text similarity. Respond only 'yes' or 'no'.",
                temperature=0.3,
                max_tokens=200
            ).lower()

            if ai_result.startswith("yes"):
                logger.info("IA considera que el nuevo ticket coincide con el issue %s", issue.key)
                matched_keys.append(issue.key)

        except Exception as e:
            logger.warning("Fallo la llamada IA al comparar con %s: %s", issue.key, e)
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                logger.info("Fallback local: la similitud con %s >= 0.8. Lo consideramos duplicado.", issue.key)
                matched_keys.append(issue.key)

    return matched_keys

# ===================== Unificar creación de tickets (librería + requests) =====================
def create_jira_ticket(
    jira: JIRA,
    project_key: str,
    summary: str,
    description: str,
    issue_type: str
) -> Optional[str]:
    """
    Crea un ticket mediante la librería oficial de Python para Jira.
    Retorna la clave (issue.key) o None en caso de error.
    """
    summary = sanitize_summary(summary)
    if not description.strip():
        return None

    try:
        issue_dict = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
        issue = jira.create_issue(fields=issue_dict)
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
    """
    Crea un ticket usando la API REST de Jira (requests).
    Retorna la clave o None en caso de error.
    """
    summary = sanitize_summary(summary)
    if not description.strip():
        return None

    if isinstance(description, str):
        fallback_adf = {
            "version": 1,
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": description.replace('\n', ' ').replace('\r', ' ')}
                    ]
                }
            ]
        }
        adf_description = fallback_adf
    elif isinstance(description, dict):
        adf_description = description
    else:
        fallback_adf = {
            "version": 1,
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": str(description)}
                    ]
                }
            ]
        }
        adf_description = fallback_adf

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": adf_description,
            "issuetype": {"name": issue_type}
        }
    }

    url = f"{jira_url}/rest/api/3/issue"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)

    try:
        response = requests.post(url, json=payload, headers=headers, auth=auth)
        if response.status_code == 201:
            return response.json().get("key")
        else:
            logger.error("Falló la creación de ticket vía requests: %s - %s", response.status_code, response.text)
            return None
    except Exception as e:
        logger.error("Excepción usando requests para crear ticket: %s", e)
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
    Función unificada: primero intenta crear con librería de Jira,
    si falla o retorna None, hace fallback con requests.
    """
    key = create_jira_ticket(jira, project_key, summary, description, issue_type)
    if key:
        return key
    logger.info("Fallo la creación con la librería de Jira. Intentando con requests...")
    return create_jira_ticket_via_requests(
        jira_url, jira_user, jira_api_token,
        project_key, summary, description, issue_type
    )

# ===================== VALIDACIÓN DE LOGS =====================
def validate_logs_directory(log_dir: str) -> List[str]:
    """
    Verifica la existencia del directorio de logs y
    retorna la lista de archivos válidos a procesar.
    
    :param log_dir: Ruta al directorio de logs
    :return: Lista de rutas a archivos válidos
    :raises FileNotFoundError: Si no existe la carpeta o no hay archivos válidos
    """
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")

    log_files = []
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if file.lower() == BANDIT_JSON_NAME.lower():
            continue
        mb_size = os.path.getsize(file_path) / 1024.0 / 1024.0
        if mb_size > MAX_FILE_SIZE_MB:
            continue
        _, ext = os.path.splitext(file.lower())
        if ext not in ALLOWED_EXTENSIONS:
            continue
        if os.path.isfile(file_path):
            log_files.append(file_path)

    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files found in '{log_dir}'.")
    return log_files

def unify_double_to_single_asterisks(description: str) -> str:
    """
    Reemplaza '**' por '*' en la descripción para unificar estilos.
    """
    while '**' in description:
        description = description.replace('**', '*')
    return description

def generate_prompt(log_type: str, language: str) -> (str, str):
    """
    Genera un prompt base y un issue_type por defecto.
    """
    if log_type == "failure":
        details = (
            "You are a technical writer creating a concise Jira Cloud ticket from logs. "
            "Keep it short, minimal Markdown. "
            "Use headings like: *Summary*, *Root Cause Analysis*, *Proposed Solutions*, etc. "
            "Use minimal triple backticks for code. "
            f"Write in {language}. Avoid enumerations like '1., a., i.'."
        )
        issue_type = "Error"
    else:
        details = (
            "You are a code reviewer specialized in Python. "
            "Below are logs from a successful build. Produce improvements with format: "
            "- Title (bold)\n- Summary\n- Description. "
            f"Use emojis for variety. Write in {language} with concise language."
        )
        issue_type = "Tarea"
    return details, issue_type

def clean_log_content(content: str) -> str:
    """
    Quita líneas vacías en blanco.
    """
    lines = content.splitlines()
    return "\n".join([line for line in lines if line.strip()])

def validate_issue_type(jira_url: str, jira_user: str, jira_api_token: str,
                        project_key: str, issue_type: str) -> None:
    """
    Valida que issue_type exista en el proyecto especificado.
    Lanza excepción si no existe.
    """
    url = f"{jira_url}/rest/api/3/issue/createmeta?projectKeys={project_key}"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)
    response = requests.get(url, headers=headers, auth=auth)
    if response.status_code == 200:
        valid_types = [it["name"] for it in response.json()["projects"][0]["issuetypes"]]
        if issue_type not in valid_types:
            raise ValueError(f"Invalid issue type: '{issue_type}'. Valid types: {valid_types}")
    else:
        raise Exception(f"Failed to fetch issue types: {response.status_code} - {response.text}")

# ===================== MÉTODOS DE ANÁLISIS =====================
def analyze_logs_for_recommendations(log_dir: str, report_language: str, project_name: str) -> List[dict]:
    """
    Analiza logs de 'success' para extraer recomendaciones.
    Retorna una lista de dicts con 'summary' y 'description'.
    """
    log_files = validate_logs_directory(log_dir)
    combined_text = []
    max_lines = 300
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:max_lines]
                combined_text.extend(lines)
        except UnicodeDecodeError:
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        return []

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)
    prompt_base, _ = generate_prompt("success", report_language)

    all_recommendations = []
    for chunk in text_chunks:
        prompt = f"{prompt_base}\n\nLogs:\n{chunk}"
        try:
            ai_text = openai_chat_with_history(
                user_prompt=prompt,
                system_prompt="You are a helpful assistant.",
                temperature=0.3,
                max_tokens=1000
            )
            recs = parse_recommendations(ai_text)
            all_recommendations.extend(recs)
        except Exception as e:
            logger.warning("Fallo analizando logs para recomendaciones: %s", e)
            continue
    return all_recommendations

def analyze_logs_with_ai(
    log_dir: str,
    log_type: str,
    report_language: str,
    project_name: str,
    branch_name: str = None
) -> (Optional[str], Optional[str], Optional[str]):
    """
    Analiza logs de 'failure' y retorna (summary_title, description_plain, issue_type).
    """
    log_files = validate_logs_directory(log_dir)
    combined_text = []
    max_lines = 300
    error_lines = []

    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:max_lines]
                combined_text.extend(lines)
                for ln in lines:
                    if any(keyword in ln for keyword in ("ERROR", "Exception", "Traceback")):
                        error_lines.append(ln.strip())
        except UnicodeDecodeError:
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        return None, None, None

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)
    prompt_base, issue_type = generate_prompt(log_type, report_language)

    error_context = ""
    if error_lines:
        few_error_lines = error_lines[:5]
        error_context = "\n\nHere are some specific error lines found:\n" + "\n".join(f"- {l}" for l in few_error_lines)

    chunk = text_chunks[0]
    final_prompt = f"{prompt_base}\n\nLogs:\n{chunk}{error_context}"

    def sanitize_title(t: str) -> str:
        return t.replace("\n", " ").replace("\r", " ").strip()

    try:
        summary = openai_chat_with_history(
            user_prompt=final_prompt,
            system_prompt=(
                "You are a helpful assistant generating concise Jira tickets. "
                "Use short statements, some emojis, minimal markdown. "
                "Make sure the title references the most relevant error."
            ),
            temperature=0.4,
            max_tokens=600
        )
        lines = summary.splitlines()

        first_line = lines[0].strip() if lines else "No Title"
        match = re.match(r"(?i)^(?:title|summary)\s*:\s*(.*)$", first_line)
        if match:
            extracted_title = match.group(1).strip()
            lines = lines[1:]
        else:
            extracted_title = first_line

        cleaned_title_line = sanitize_title(extracted_title)
        icon = choose_error_icon()
        if branch_name:
            cleaned_title_line += f" [branch: {branch_name}]"

        summary_title = f"{project_name} {icon} {cleaned_title_line}"
        remaining_desc = "\n".join(lines).strip()
        if not remaining_desc:
            remaining_desc = summary

        description_plain = unify_double_to_single_asterisks(remaining_desc.replace("\t", " "))
        return summary_title, description_plain, issue_type

    except Exception as e:
        logger.error("Error analizando logs con IA: %s", e)
        return None, None, None

# ===================== MAIN =====================
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

    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    if args.log_type == "failure":
        summary, description, issue_type = analyze_logs_with_ai(
            args.log_dir, args.log_type, args.report_language,
            args.project_name, branch_name=args.branch
        )
        if not summary or not description:
            logger.error("ERROR: No ticket will be created (analysis empty).")
            return

        try:
            validate_issue_type(args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, issue_type)
        except Exception as e:
            logger.error("ERROR: %s", e)
            return

        # Buscar tickets abiertos (ToDo, InProgress, etc.)
        jql_states = '"To Do", "In Progress", "Open", "Reopened"'
        existing_issues = find_similar_issues(
            jira, args.jira_project_key,
            summary, description, issue_type,
            f"status IN ({jql_states})"
        )
        if existing_issues:
            logger.info(f"INFO: Ticket(s) {existing_issues} parecen ya representar el mismo problema. Skipping.")
            return

        # Buscar tickets finalizados
        done_issues = find_similar_issues(
            jira, args.jira_project_key,
            summary, description, issue_type,
            'statusCategory = Done'
        )

        logger.info("Creando ticket de tipo 'Error' en Jira...")
        ticket_key = create_jira_ticket_unified(
            jira, args.jira_url, jira_user_email, jira_api_token,
            args.jira_project_key, summary, description, issue_type
        )
        if ticket_key:
            logger.info(f"Ticket creado: {ticket_key}")
            if done_issues:
                duplicates_str = ", ".join(done_issues)
                comment_body = get_repeated_incident_comment(duplicates_str, args.report_language)
                try:
                    jira.add_comment(ticket_key, comment_body)
                except Exception as e:
                    logger.warning("No se pudo añadir comentario de incidencias repetidas: %s", e)
                for old_key in done_issues:
                    try:
                        jira.create_issue_link(
                            type="Relates",
                            inwardIssue=ticket_key,
                            outwardIssue=old_key
                        )
                    except Exception as e:
                        logger.warning("No se pudo crear issue link: %s", e)
        else:
            logger.error("ERROR: Creación de ticket fallida.")

    else:
        # logs de success => recomendaciones
        recommendations = analyze_logs_for_recommendations(
            args.log_dir, args.report_language, args.project_name
        )
        if not recommendations:
            logger.info("INFO: No hay recomendaciones generadas por la IA.")
            return

        issue_type = "Tarea"
        for i, rec in enumerate(recommendations, start=1):
            r_summary = rec["summary"]
            r_desc = rec["description"]
            if not r_desc.strip():
                continue
            if should_skip_recommendation(r_summary, r_desc):
                continue

            # Chequear tickets abiertos
            jql_states = '"To Do", "In Progress", "Open", "Reopened"'
            existing_issues = find_similar_issues(
                jira, args.jira_project_key,
                r_summary, r_desc, issue_type,
                f"status IN ({jql_states})"
            )
            if existing_issues:
                continue

            # Chequear tickets descartados
            discard_issues = find_similar_issues(
                jira, args.jira_project_key,
                r_summary, r_desc, issue_type,
                'status IN ("DESCARTADO")'
            )
            if discard_issues:
                continue

            final_title, wiki_desc = format_ticket_content(
                args.project_name, r_summary, r_desc, "Improvement"
            )
            if not wiki_desc.strip():
                continue

            new_key = create_jira_ticket_unified(
                jira, args.jira_url, jira_user_email, jira_api_token,
                args.jira_project_key, final_title, wiki_desc, issue_type
            )
            if new_key:
                logger.info(f"Recomendación #{i} => Creado ticket: {new_key}")
            else:
                logger.error(f"Recomendación #{i} => Creación de ticket fallida.")

def get_repeated_incident_comment(duplicates_str: str, language: str) -> str:
    """
    Construye un comentario indicando que se encontraron
    incidencias previas similares.
    """
    if language.lower().startswith("es"):
        return (
            f"Se han encontrado incidencias previas que podrían ser similares: {duplicates_str}. "
            "Por favor, verificar si esta incidencia está relacionada con alguna de ellas."
        )
    else:
        return (
            f"Similar incidents found: {duplicates_str}. "
            "Please check if this new issue is related to any of them."
        )

if __name__ == "__main__":
    main()
