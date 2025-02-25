#!/usr/bin/env python3
import os
import sys
import re
import json
import random
import math
import argparse
import requests
from datetime import datetime
from difflib import SequenceMatcher
from jira import JIRA
from openai import OpenAI

# ===================== CONFIGURACIÓN GLOBAL =====================
OPENAI_MODEL = "gpt-4o"  # Cambia aquí el modelo que quieras usar
MAX_CHAR_PER_REQUEST = 20000  # Límite aproximado de caracteres a enviar al prompt
BANDIT_JSON_NAME = "bandit-output.json"
MAX_FILE_SIZE_MB = 2.0
ALLOWED_EXTENSIONS = (".log", ".sarif")  # <-- AÑADIDO: permitimos .log y .sarif

# ===================== CONFIGURACIÓN OPENAI =====================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set.")
    sys.exit(1)

print("DEBUG: Initializing OpenAI client...")
client = OpenAI(api_key=api_key)

# ===================== CONEXIÓN A JIRA =====================
def connect_to_jira(jira_url, jira_user, jira_api_token):
    print(f"DEBUG: Connecting to Jira at {jira_url} with user {jira_user}...")
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    print("DEBUG: Successfully connected to Jira.")
    return jira

# ===================== FUNCIONES DE SANITIZACIÓN =====================
def sanitize_summary(summary):
    """
    Elimina caracteres problemáticos y saltos de linea,
    además trunca a 255.
    """
    # Sustitución de saltos de línea por espacio
    summary = summary.replace("\n", " ").replace("\r", " ")

    # Eliminamos (o escapamos) cualquier otro caracter extraño
    # y dejamos algunos símbolos
    sanitized = "".join(
        c for c in summary
        if c.isalnum() or c.isspace() or c in "-_:,./()[]{}"
    )

    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized.strip()

def preprocess_text(text: str) -> str:
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    lowered = text_no_punct.strip().lower()
    return lowered

def calculate_similarity(text1: str, text2: str) -> float:
    t1 = preprocess_text(text1)
    t2 = preprocess_text(text2)
    ratio = SequenceMatcher(None, t1, t2).ratio()
    return ratio

# ===================== CONVERSIÓN A WIKI (ADF -> Jira) =====================
def convert_adf_to_wiki(adf) -> str:
    """Convierte un ADF simplificado a wiki markup de Jira."""
    def process_node(node):
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

        # fallback recursivo
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
def parse_recommendations(ai_text: str) -> list:
    recommendations = []
    print("DEBUG: Raw AI output for recommendations:\n", ai_text)

    blocks = re.split(r"\n\s*-\s+", ai_text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        print("DEBUG: Processing block:\n", block)

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
        print(f"DEBUG: Extracted - Title: '{title}' | Summary: '{summary_text}' | Description: '{description_text}'")

        recommendations.append({
            "summary": full_summary,
            "description": description_text
        })

    print(f"DEBUG: Parsed {len(recommendations)} recommendation(s).")
    return recommendations

# ===================== ÍCONOS =====================
IMPROVEMENT_ICONS = ["🚀", "💡", "🔧", "🤖", "🌟", "📈", "✨"]
ERROR_ICONS = ["🐞", "🔥", "💥", "🐛", "⛔", "🚫"]

def choose_improvement_icon() -> str:
    return random.choice(IMPROVEMENT_ICONS)

def choose_error_icon() -> str:
    return random.choice(ERROR_ICONS)

# ===================== FILTRAR RECOMENDACIONES NO DESEADAS =====================
def should_skip_recommendation(summary: str, description: str) -> bool:
    skip_keywords = [
        "bandit", "npm audit", "nancy", "scan-security-vulnerabilities",
        "check-code-format", "lint code", "owasp dependency check",
        "az storage", "azure storage"
    ]
    combined = f"{summary}\n{description}".lower()
    for kw in skip_keywords:
        if kw in combined:
            print(f"DEBUG: Recommendation references tool '{kw}'. Skipping.")
            return True
    return False

# ===================== FORMATEO FINAL (IA) =====================
def format_ticket_content(project_name: str, rec_summary: str, rec_description: str, ticket_category: str) -> tuple:
    """
    Llama a la IA para intentar formatear (title, description) en JSON con ADF.
    En caso de error JSON, fallback a un texto simple.
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
    print("DEBUG: format_ticket_content prompt:\n", prompt)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional technical writer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3
        )
        ai_output = response.choices[0].message.content.strip()
        print("DEBUG: Raw AI output from format_ticket_content:\n", ai_output)

        # Intentar quitar backticks en caso de que el JSON venga con fences
        if ai_output.startswith("```"):
            lines = ai_output.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            ai_output = "\n".join(lines).strip()

        print("DEBUG: AI output after stripping code fences:\n", ai_output)

        # ======== INTENTO 1: parsear JSON ====
        try:
            ticket_json = json.loads(ai_output)
        except json.JSONDecodeError as e:
            print(f"WARNING: Primary JSON parse error => {e}")
            # ======== INTENTO 2: si hay "Unterminated string" u otros problemas, 
            # intentamos un mini-limpieza adicional (por ejemplo, recortar tras la última llave)
            cleaned = re.sub(r'```.*?```', '', ai_output, flags=re.DOTALL)  # elimina triple backticks en medio
            # O con un truncado heurístico:
            last_brace = cleaned.rfind("}")
            if last_brace != -1:
                cleaned = cleaned[: last_brace+1]
            # reintenta parsear
            try:
                ticket_json = json.loads(cleaned)
                print("DEBUG: Successfully parsed JSON after second attempt cleaning.")
            except Exception as e2:
                print(f"WARNING: second parse attempt also failed => {e2}")
                # fallback total
                fallback_summary = sanitize_summary(rec_summary)
                fallback_summary = f"{icon} {fallback_summary}"
                fallback_desc = f"Fallback description:\n\n{rec_description}"
                return fallback_summary, fallback_desc

        # Si llegamos aquí => parseado OK
        final_title = ticket_json.get("title", "")
        adf_description = ticket_json.get("description", {})

        # Asegura que tenga ícono en caso de que no viniera
        if not any(ic in final_title for ic in (IMPROVEMENT_ICONS + ERROR_ICONS)):
            final_title = f"{icon} {final_title}"

        if len(final_title) > 255:
            final_title = final_title[:255]

        wiki_text = convert_adf_to_wiki(adf_description)
        return final_title, wiki_text

    except Exception as e:
        print(f"WARNING: Failed to format ticket content with AI: {e}")
        fallback_summary = sanitize_summary(rec_summary)
        fallback_summary = f"{icon} {fallback_summary}"
        wiki_text = f"Fallback description:\n\n{rec_description}"
        return fallback_summary, wiki_text

# ===================== BÚSQUEDA DE TICKETS (LOCAL+IA) =====================
def check_existing_tickets_local_and_ia_summary_desc(jira, project_key, new_summary, new_description, issue_type):
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9
    sanitized_sum = sanitize_summary(new_summary)
    print(f"DEBUG: sanitized_summary='{sanitized_sum}'")

    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)
    jql_issue_type = "Task" if issue_type.lower() == "tarea" else issue_type
    jql_query = f'project = "{project_key}" AND issuetype = "{jql_issue_type}" AND status IN ({states_str})'
    print(f"DEBUG: JQL -> {jql_query}")

    try:
        issues = jira.search_issues(jql_query)
        print(f"DEBUG: Found {len(issues)} candidate issue(s).")
    except Exception as e:
        print(f"ERROR: Failed to execute JQL query: {e}")
        return None

    for issue in issues:
        issue_key = issue.key
        existing_summary = issue.fields.summary or ""
        existing_description = issue.fields.description or ""
        print(f"DEBUG: Analyzing Issue {issue_key}")

        summary_sim = calculate_similarity(new_summary, existing_summary)
        print(f"DEBUG: summary_sim with {issue_key} = {summary_sim:.2f}")
        desc_sim = calculate_similarity(new_description, existing_description)
        print(f"DEBUG: description_sim with {issue_key} = {desc_sim:.2f}")

        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            continue
        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            print(f"INFO: Found duplicate ticket {issue_key} (high local similarity).")
            return issue_key

        # Check con IA
        print(f"DEBUG: Intermediate range for {issue_key}. Asking IA for final check...")
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant specialized in analyzing text similarity. Respond only 'yes' or 'no'."},
                    {"role": "user", "content": (
                        "We have two issues:\n\n"
                        f"Existing issue:\nSummary: {existing_summary}\nDescription: {existing_description}\n\n"
                        f"New issue:\nSummary: {new_summary}\nDescription: {new_description}\n\n"
                        "Do they represent essentially the same issue? Respond 'yes' or 'no'."
                    )}
                ],
                max_tokens=200,
                temperature=0.3
            )
            ai_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: AI result for {issue_key}: '{ai_result}'")
            if ai_result.startswith("yes"):
                print(f"INFO: Found duplicate ticket (IA confirms) -> {issue_key}")
                return issue_key
        except:
            # fallback
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                print(f"INFO: Fallback local: similarity >= 0.8 for {issue_key}; marking as duplicate.")
                return issue_key

    print("DEBUG: No duplicate ticket found after local+IA approach.")
    return None

# ===================== CREACIÓN DE TICKETS =====================
def create_jira_ticket(jira, project_key, summary, description, issue_type):
    summary = sanitize_summary(summary)
    if not description.strip():
        print("DEBUG: description is empty; skipping ticket creation.")
        return None
    try:
        issue_dict = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
        print(f"DEBUG: Issue fields -> {issue_dict}")
        issue = jira.create_issue(fields=issue_dict)
        print("DEBUG: Ticket created successfully via JIRA library.")
        return issue.key
    except Exception as e:
        print(f"ERROR: Could not create ticket via JIRA library: {e}")
        return None

def create_jira_ticket_via_requests(
    jira_url, jira_user, jira_api_token,
    project_key, summary, description, issue_type
):
    summary = sanitize_summary(summary)
    if not description.strip():
        print("DEBUG: description is empty; skipping ticket creation via API.")
        return None

    # Asegurarnos de que 'description' sea un ADF válido si tu Jira Cloud lo requiere
    # Si 'description' ya es un ADF dict =>  OK
    # Si NO, creamos un fallback ADF mínimo
    if isinstance(description, str):
        # creamos un doc con un solo párrafo
        fallback_adf = {
            "version": 1,
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            # Reemplazamos saltos, etc.
                            "text": description.replace('\n', ' ').replace('\r', ' ')
                        }
                    ]
                }
            ]
        }
        adf_description = fallback_adf
    elif isinstance(description, dict):
        # Asumimos que ya viene en formato ADF
        adf_description = description
    else:
        # fallback total
        fallback_adf = {
            "version": 1,
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {
                            "type": "text",
                            "text": str(description)
                        }
                    ]
                }
            ]
        }
        adf_description = fallback_adf

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": adf_description,  # ADF en JSON
            "issuetype": {"name": issue_type}
        }
    }
    print(f"DEBUG: Payload -> {json.dumps(payload, indent=2)}")

    url = f"{jira_url}/rest/api/3/issue"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)

    response = requests.post(url, json=payload, headers=headers, auth=auth)
    if response.status_code == 201:
        print("DEBUG: Ticket created successfully via API.")
        print("Ticket created successfully:", response.json())
        return response.json().get("key")
    else:
        print(
            f"ERROR: Failed to create ticket via API: {response.status_code} - {response.text}"
        )
        return None

# ===================== VALIDACIÓN DE LOGS =====================
def validate_logs_directory(log_dir: str) -> list:
    print(f"DEBUG: Validating logs directory -> {log_dir}")
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")

    log_files = []
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)

        # Descarta el bandit-output.json
        if file.lower() == BANDIT_JSON_NAME.lower():
            print(f"DEBUG: Skipping {file}, as it's bandit-output.json.")
            continue

        # Descarta si pasa cierto tamaño
        mb_size = os.path.getsize(file_path) / 1024.0 / 1024.0
        if mb_size > MAX_FILE_SIZE_MB:
            print(f"DEBUG: Skipping {file} because it's {mb_size:.2f} MB > {MAX_FILE_SIZE_MB:.2f} MB limit.")
            continue

        # AÑADIDO: Si quieres **solo** .log y .sarif, por ejemplo:
        _, ext = os.path.splitext(file.lower())
        if ext not in ALLOWED_EXTENSIONS:
            print(f"DEBUG: Skipping {file} because extension {ext} is not in {ALLOWED_EXTENSIONS}.")
            continue

        # Lo añadimos a la lista
        if os.path.isfile(file_path):
            log_files.append(file_path)

    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files found in the directory '{log_dir}'.")
    print(f"DEBUG: Found {len(log_files)} log file(s) in total.")
    return log_files

def clean_log_content(content: str) -> str:
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

# ===================== VALIDACIÓN DE TIPO DE INCIDENCIA =====================
def validate_issue_type(jira_url, jira_user, jira_api_token, project_key, issue_type):
    print(f"DEBUG: Validating issue type '{issue_type}' for project '{project_key}'...")
    url = f"{jira_url}/rest/api/3/issue/createmeta?projectKeys={project_key}"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)
    response = requests.get(url, headers=headers, auth=auth)
    if response.status_code == 200:
        valid_types = [it["name"] for it in response.json()["projects"][0]["issuetypes"]]
        print(f"DEBUG: Valid issue types -> {valid_types}")
        if issue_type not in valid_types:
            raise ValueError(f"Invalid issue type: '{issue_type}'. Valid types: {valid_types}")
    else:
        raise Exception(f"Failed to fetch issue types: {response.status_code} - {response.text}")

# ===================== GENERACIÓN DEL PROMPT =====================
def generate_prompt(log_type: str, language: str) -> tuple:
    print(f"DEBUG: Generating prompt for log_type='{log_type}', language='{language}'...")
    if log_type == "failure":
        details = (
            f"You are a technical writer creating a concise Jira Cloud ticket from logs. "
            f"Keep the format short and professional, using minimal Markdown. "
            f"Use these headings or bullet points (without numeric or lettered sub-lists):\n\n"
            f"*Summary* ❗ (1 line)\n"
            f"*Root Cause Analysis* 🔍 (brief explanation)\n"
            f"*Proposed Solutions* 🛠️ (a few bullet points)\n"
            f"*Preventive Measures* ⛑️ (another bullet list or short paragraphs)\n"
            f"*Impact Analysis* ⚠️ (consequences if not resolved)\n\n"
            f"Use minimal triple backticks for code/log snippets if needed, "
            f"Use emojis like {ERROR_ICONS} for variety. Write in {language} with concise language. "
            f"and avoid enumerations like '1. a. i.'."
        )
        issue_type = "Error"
    else:
        details = (
            f"You are a code reviewer specialized in Python. "
            f"Below are logs from a successful build, possibly including minor warnings. "
            f"Analyze them thoroughly and produce specific code improvements or refactors with this format:\n\n"
            f"- Title (bold)\n"
            f"- Summary (1 line)\n"
            f"- Description (detailed explanation, referencing lines or snippets, with small code examples)\n\n"
            f"Your suggestions MUST derive from real warnings or code smells in the logs. "
            f"Use emojis like {IMPROVEMENT_ICONS} for variety. Write in {language} with concise language. "
            f"Avoid triple backticks unless needed."
        )
        issue_type = "Tarea"

    print(f"DEBUG: Prompt generated. Issue type = {issue_type}")
    return details, issue_type

def unify_double_to_single_asterisks(description: str) -> str:
    while '**' in description:
        description = description.replace('**', '*')
    return description

def sanitize_title(title: str) -> str:
    title = re.sub(r"[\*`]+", "", title).strip()
    if len(title) > 255:
        title = title[:255]
    return title

# ===================== RECORTAR EL CONTENIDO SI EXCEDE =====================
def chunk_content_if_needed(combined_logs: str, max_chars: int = MAX_CHAR_PER_REQUEST) -> list:
    """Devuelve una lista de trozos de texto (cada uno <= max_chars)."""
    if len(combined_logs) <= max_chars:
        return [combined_logs]
    chunks = []
    start = 0
    while start < len(combined_logs):
        end = start + max_chars
        chunk = combined_logs[start:end]
        chunks.append(chunk)
        start = end
    return chunks

# ===================== MÉTODOS DE ANÁLISIS (SUCCESS/FAILURE) =====================
def analyze_logs_for_recommendations(log_dir: str, report_language: str, project_name: str) -> list:
    print("DEBUG: analyze_logs_for_recommendations... (with chunking)")

    log_files = validate_logs_directory(log_dir)

    combined_text = []
    max_lines = 300
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:max_lines]
                combined_text.extend(lines)
                print(f"DEBUG: Reading '{file}', took up to {max_lines} lines.")
        except UnicodeDecodeError:
            print(f"WARNING: Could not read file {file}. Skipping.")
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        print("ERROR: No relevant logs found for analysis.")
        return []

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)
    print(f"DEBUG: We have {len(text_chunks)} chunk(s) to send to the AI.")

    prompt_base, _ = generate_prompt("success", report_language)

    all_recommendations = []
    for idx, chunk in enumerate(text_chunks, start=1):
        prompt = f"{prompt_base}\n\nLogs:\n{chunk}"
        print(f"DEBUG: Sending chunk {idx}/{len(text_chunks)} to OpenAI (length={len(chunk)})")

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            ai_text = response.choices[0].message.content.strip()
            print("DEBUG: AI returned text for chunk:\n", ai_text)
            recs = parse_recommendations(ai_text)
            all_recommendations.extend(recs)
        except Exception as e:
            print(f"ERROR: chunk {idx} -> {e}")
            continue

    print(f"DEBUG: Returning {len(all_recommendations)} recommendation(s) total.")
    return all_recommendations

def analyze_logs_with_ai(log_dir: str, log_type: str, report_language: str, project_name: str) -> tuple:
    print("DEBUG: analyze_logs_with_ai... (with chunking)")

    log_files = validate_logs_directory(log_dir)

    combined_text = []
    max_lines = 300
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:max_lines]
                combined_text.extend(lines)
                print(f"DEBUG: Reading '{file}', took up to {max_lines} lines.")
        except UnicodeDecodeError:
            print(f"WARNING: Could not read file {file}. Skipping.")
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        print("ERROR: No relevant logs found for analysis.")
        return None, None, None

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)
    print(f"DEBUG: We have {len(text_chunks)} chunk(s) for failure logs.")

    prompt_base, issue_type = generate_prompt(log_type, report_language)

    # Usamos sólo el primer chunk para el ticket de error
    chunk = text_chunks[0]
    prompt = f"{prompt_base}\n\nLogs:\n{chunk}"
    print("DEBUG: Sending chunk 1 for failure to OpenAI...")

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant generating concise Jira tickets. "
                        "Use short, direct statements, some emojis, minimal markdown. "
                        "Avoid triple backticks for code unless strictly necessary."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
        lines = summary.splitlines()

        # --- NUEVA LÓGICA: buscar 'Title:' o 'title:' en la primera línea ---
        if lines:
            first_line = lines[0].strip()
        else:
            first_line = "No Title"

        # Ejemplo de regex que busca 'Title:' al inicio (insensible a mayúsculas):
        match = re.match(r"(?i)^title\s*:\s*(.*)$", first_line)
        if match:
            # Si encontramos 'Title: ...'
            extracted_title = match.group(1).strip()
            # Quitamos la línea para que no quede duplicada en la descripción:
            lines = lines[1:]
        else:
            # Fallback si no matchea:
            extracted_title = first_line

        cleaned_title_line = sanitize_title(extracted_title)
        icon = choose_error_icon()
        summary_title = f"{project_name} {icon} {cleaned_title_line}"

        # El resto de líneas van como descripción
        remaining_desc = "\n".join(lines).strip()
        if not remaining_desc:
            remaining_desc = summary

        description_plain = unify_double_to_single_asterisks(remaining_desc.replace("\t", " "))

        print(f"DEBUG: Final summary title -> {summary_title}")
        print(f"DEBUG: Description length -> {len(description_plain)} chars.")
        return summary_title, description_plain, issue_type

    except Exception as e:
        print(f"ERROR: Failed to analyze logs with AI: {e}")
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
    args = parser.parse_args()

    print("DEBUG: Starting main process with arguments:", args)

    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user_email = os.getenv("JIRA_USER_EMAIL")
    if not jira_api_token or not jira_user_email:
        print("ERROR: Missing env vars JIRA_API_TOKEN or JIRA_USER_EMAIL.")
        sys.exit(1)

    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    if args.log_type == "failure":
        summary, description, issue_type = analyze_logs_with_ai(
            args.log_dir, args.log_type, args.report_language, args.project_name
        )
        if not summary or not description:
            print("ERROR: No ticket will be created (analysis empty).")
            return
        try:
            validate_issue_type(args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, issue_type)
        except Exception as e:
            print(f"ERROR: {e}")
            return
        dup_key = check_existing_tickets_local_and_ia_summary_desc(
            jira, args.jira_project_key, summary, description, issue_type
        )
        if dup_key:
            print(f"INFO: Ticket {dup_key} already exists. Skipping.")
            return

        print("DEBUG: Creating failure ticket in Jira...")
        ticket_key = create_jira_ticket(jira, args.jira_project_key, summary, description, issue_type)
        if ticket_key:
            print(f"INFO: JIRA Ticket Created: {ticket_key}")
        else:
            print("WARNING: Falling back to create via REST API...")
            fallback_key = create_jira_ticket_via_requests(
                args.jira_url, jira_user_email, jira_api_token, args.jira_project_key,
                summary, description, issue_type
            )
            if fallback_key:
                print(f"INFO: JIRA Ticket Created via REST: {fallback_key}")
            else:
                print("ERROR: Failed to create JIRA ticket.")

    else:
        # "success"
        recommendations = analyze_logs_for_recommendations(
            args.log_dir, args.report_language, args.project_name
        )
        if not recommendations:
            print("INFO: No recommendations generated by the AI.")
            return

        print(f"DEBUG: {len(recommendations)} recommendation(s) total.")
        issue_type = "Tarea"

        for i, rec in enumerate(recommendations, start=1):
            r_summary = rec["summary"]
            r_desc = rec["description"]

            if not r_desc.strip():
                print(f"DEBUG: Recommendation #{i} has empty desc. Skipping.")
                continue

            if should_skip_recommendation(r_summary, r_desc):
                print(f"INFO: Recommendation #{i} references existing tool. Skipping.")
                continue

            dup_key = check_existing_tickets_local_and_ia_summary_desc(
                jira, args.jira_project_key, r_summary, r_desc, issue_type
            )
            if dup_key:
                print(f"INFO: Recommendation #{i} => ticket {dup_key} already exists.")
                continue

            final_title, wiki_desc = format_ticket_content(
                args.project_name, r_summary, r_desc, "Improvement"
            )
            if not wiki_desc.strip():
                print(f"DEBUG: recommendation {i} => empty wiki desc. skip.")
                continue

            new_key = create_jira_ticket(jira, args.jira_project_key, final_title, wiki_desc, issue_type)
            if new_key:
                print(f"INFO: Created recommendation #{i} => {new_key}")
            else:
                print(f"WARNING: fallback creation for #{i} via REST...")
                fallback_key = create_jira_ticket_via_requests(
                    args.jira_url, jira_user_email, jira_api_token, args.jira_project_key,
                    final_title, wiki_desc, issue_type
                )
                if fallback_key:
                    print(f"INFO: Created recommendation #{i} => {fallback_key}")
                else:
                    print(f"ERROR: Could not create ticket for recommendation #{i}.")

    print("DEBUG: Process finished.")

if __name__ == "__main__":
    main()
