#!/usr/bin/env python3
import os
import argparse
import tarfile
import requests
import re
import json
import random
from jira import JIRA
from openai import OpenAI
from datetime import datetime
from difflib import SequenceMatcher

# ============ CONFIGURACIÓN OPENAI ============
print("DEBUG: create_jira_ticket.py")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)
print("DEBUG: Initializing OpenAI client...")
client = OpenAI(api_key=api_key)

# ============ CONEXIÓN A JIRA ============
def connect_to_jira(jira_url, jira_user, jira_api_token):
    print(f"DEBUG: Connecting to Jira at {jira_url} with user {jira_user}...")
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    print("DEBUG: Successfully connected to Jira.")
    return jira

# ============ FUNCIONES DE SANITIZACIÓN Y SIMILITUD ============

def sanitize_summary(summary):
    """
    Elimina caracteres problemáticos y, además, trunca el summary si supera 255 caracteres.
    """
    print(f"DEBUG: Sanitizing summary: '{summary}'")
    # Permitimos algunos signos para no recortar todo
    sanitized = "".join(c for c in summary if c.isalnum() or c.isspace() or c in "-_:,./()[]{}")
    if len(sanitized) > 255:
        print("DEBUG: summary too long, truncating to 255 chars.")
        sanitized = sanitized[:255]
    print(f"DEBUG: Resulting sanitized summary: '{sanitized}'")
    return sanitized.strip()

def preprocess_text(text):
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    lowered = text_no_punct.strip().lower()
    return lowered

def calculate_similarity(text1, text2):
    t1 = preprocess_text(text1)
    t2 = preprocess_text(text2)
    ratio = SequenceMatcher(None, t1, t2).ratio()
    return ratio

# ============ CONVERSIÓN DE BLOQUES DE CÓDIGO (ADF -> Wiki) ============

def convert_adf_to_wiki(adf):
    """
    Convierte un ADF simplificado a wiki markup de Jira:
      - codeBlock -> {code}...{code}
      - bulletList -> '* ' items
      - paragraphs -> saltos de línea
    """
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

        # Cualquier otro tipo se procesa recursivamente (fallback)
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

# ============ PARSEO DE RECOMENDACIONES ============
def parse_recommendations(ai_text):
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

# ============ LISTAS DE ÍCONOS ============

IMPROVEMENT_ICONS = ["🚀", "💡", "🔧", "🤖", "🌟", "📈", "✨"]
ERROR_ICONS = ["🐞", "🔥", "💥", "🐛", "⛔", "🚫"]

def choose_improvement_icon():
    return random.choice(IMPROVEMENT_ICONS)

def choose_error_icon():
    return random.choice(ERROR_ICONS)

# ============ FILTRAR RECOMENDACIONES NO DESEADAS ============

def should_skip_recommendation(summary, description):
    """
    Retorna True si en la recomendación se mencionan herramientas que ya tenemos,
    o si no aporta contenido real. Ajusta el set de palabras clave según tu preferencia.
    """
    # Palabras clave que NO queremos tickets que las sugieran
    skip_keywords = [
        "bandit", "npm audit", "nancy", "scan-security-vulnerabilities",
        "check-code-format", "lint code", "owasp dependency check", "az storage", "azure storage"
    ]
    combined = f"{summary}\n{description}".lower()

    for kw in skip_keywords:
        if kw in combined:
            print(f"DEBUG: Recommendation references tool '{kw}'. Skipping.")
            return True

    return False

# ============ FORMATEO FINAL DEL CONTENIDO DEL TICKET ============

def format_ticket_content(project_name, rec_summary, rec_description, ticket_category):
    """
    Llama a la IA para generar un objeto JSON con 'title' y 'description' en ADF,
    luego convierte 'description' a wiki markup. Aseguramos que haya un ícono.
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
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional technical writer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3
        )
        ai_output = response.choices[0].message.content.strip()
        print("DEBUG: Raw AI output from format_ticket_content:\n", ai_output)

        if ai_output.startswith("```"):
            lines = ai_output.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            ai_output = "\n".join(lines).strip()

        print("DEBUG: AI output after stripping code fences:\n", ai_output)

        ticket_json = json.loads(ai_output)
        final_title = ticket_json.get("title", "")
        adf_description = ticket_json.get("description", {})

        # Verificamos que tenga un ícono. Si no lo tiene, se lo añadimos.
        if not any(icon_char in final_title for icon_char in (IMPROVEMENT_ICONS + ERROR_ICONS)):
            # Insertamos en la parte inicial
            final_title = f"{icon} {final_title}"

        # limitamos a 255
        if len(final_title) > 255:
            final_title = final_title[:255]

        # Convertir ADF a wiki
        wiki_text = convert_adf_to_wiki(adf_description)
        return final_title, wiki_text

    except Exception as e:
        print(f"WARNING: Failed to format ticket content with AI: {e}")
        # fallback con icono
        fallback_summary = sanitize_summary(rec_summary)
        fallback_summary = f"{choose_improvement_icon()} {fallback_summary}"  # Para improvements
        wiki_text = f"Fallback description:\n\n{rec_description}"
        return fallback_summary, wiki_text

# ============ BÚSQUEDA DE TICKETS EXISTENTES (LOCAL + IA) ============

def check_existing_tickets_local_and_ia_summary_desc(jira, project_key, new_summary, new_description, issue_type):
    print("DEBUG: Checking for existing tickets (local + IA, summary + description)")
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
            print(f"DEBUG: Both summary_sim and description_sim < {LOCAL_SIM_LOW:.2f}; ignoring {issue_key}.")
            continue
        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            print(f"INFO: Found duplicate ticket {issue_key} (high local similarity).")
            return issue_key

        # IA check
        print(f"DEBUG: Intermediate range for {issue_key}. Asking IA for final check...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
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
            elif ai_result.startswith("no"):
                print(f"DEBUG: AI says 'no' for {issue_key}. Continue.")
                continue
            else:
                print(f"WARNING: Ambiguous AI response '{ai_result}' for {issue_key}; continuing.")
                continue
        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with AI for {issue_key}: {e}")
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                print(f"INFO: Fallback local: similarity >= 0.8 for {issue_key}; marking as duplicate.")
                return issue_key
            else:
                print(f"DEBUG: Not high enough for fallback; ignoring.")
                continue

    print("DEBUG: No duplicate ticket found after local+IA approach.")
    return None

# ============ CREACIÓN DE TICKETS ============

def create_jira_ticket(jira, project_key, summary, description, issue_type):
    """
    Se crea el ticket con summary y description en wiki (string).
    No se crea si la descripción está vacía o casi vacía.
    """
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

def create_jira_ticket_via_requests(jira_url, jira_user, jira_api_token, project_key, summary, description, issue_type):
    """
    Crea ticket via API REST. No se crea si description está vacío.
    """
    summary = sanitize_summary(summary)
    if not description.strip():
        print("DEBUG: description is empty; skipping ticket creation via API.")
        return None

    url = f"{jira_url}/rest/api/3/issue"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type}
        }
    }
    print(f"DEBUG: Payload -> {json.dumps(payload, indent=2)}")
    response = requests.post(url, json=payload, headers=headers, auth=auth)
    if response.status_code == 201:
        print("DEBUG: Ticket created successfully via API.")
        print("Ticket created successfully:", response.json())
        return response.json().get("key")
    else:
        print(f"ERROR: Failed to create ticket via API: {response.status_code} - {response.text}")
        return None

# ============ VALIDACIÓN DE LOGS ============

def validate_logs_directory(log_dir):
    print(f"DEBUG: Validating logs directory -> {log_dir}")
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")

    log_files = []
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if file.endswith(".tar.gz"):
            print(f"DEBUG: Extracting tar.gz -> {file_path}")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=log_dir)
                log_files.extend(os.path.join(log_dir, member.name) for member in tar.getmembers() if member.isfile())
        elif os.path.isfile(file_path):
            log_files.append(file_path)

    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files found in the directory '{log_dir}'.")
    print(f"DEBUG: Found {len(log_files)} log file(s) in total.")
    return log_files

def clean_log_content(content):
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

# ============ VALIDACIÓN DE TIPO DE INCIDENCIA ============

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

# ============ GENERACIÓN DEL PROMPT PARA LA IA ============

def generate_prompt(log_type, language):
    print(f"DEBUG: Generating prompt for log_type='{log_type}', language='{language}'...")
    if log_type == "failure":
        details = (
            "You are a technical writer creating a concise Jira Cloud ticket from logs. "
            "Keep the format short and professional, using minimal Markdown. "
            "Focus on these sections:\n\n"
            "1) *Summary* ❗: A single-sentence overview of the main issue.\n"
            "2) *Root Cause Analysis* 🔍: Briefly state the cause. Include log snippets if crucial.\n"
            "3) *Proposed Solutions* 🛠️: List concrete steps to fix the issue, using bullets or short paragraphs.\n"
            "4) *Preventive Measures* ⛑️: Suggest ways to avoid recurrence. Keep it succinct.\n"
            "5) *Impact Analysis* ⚠️: What happens if it's not addressed?\n\n"
            "Use emojis (e.g. 🐞, 🔥, 💥, etc.) but keep them minimal. Avoid triple backticks unless strictly necessary."
        )
        issue_type = "Error"
    else:
        details = (
            f"You are a code reviewer specialized in Python. "
            f"Below are logs from a successful build, possibly including minor warnings. "
            f"Analyze them thoroughly and produce specific code improvements or refactors with this format:\n\n"
            f"- Title (bold)\n"
            f"- Summary (1 line)\n"
            f"- Description (detailed explanation, referencing actual lines or snippets, showing small code examples)\n\n"
            f"Your suggestions MUST derive from real warnings or code smells in the logs. "
            f"Use emojis like {IMPROVEMENT_ICONS} for variety. Provide short, direct bullet points. "
            f"Write in {language} with concise technical language. Avoid triple backticks unless needed.\n"
        )
        issue_type = "Tarea"

    print(f"DEBUG: Prompt generated. Issue type = {issue_type}")
    return details, issue_type

# ============ UNIFICAR ASTERISCOS EN DESCRIPCIONES ============

def unify_double_to_single_asterisks(description):
    print("DEBUG: Unifying double asterisks to single...")
    while '**' in description:
        description = description.replace('**', '*')
    return description

def sanitize_title(title):
    print(f"DEBUG: Sanitizing title '{title}'...")
    title = re.sub(r"[\*`]+", "", title).strip()
    if len(title) > 255:
        title = title[:255]
    print(f"DEBUG: Title after sanitize -> '{title}'")
    return title

# ============ ANALIZAR LOGS (SUCCESS) ============

def analyze_logs_for_recommendations(log_dir, report_language, project_name):
    print("DEBUG: analyze_logs_for_recommendations...")
    log_files = validate_logs_directory(log_dir)
    combined_logs = []
    max_lines = 300

    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                combined_logs.extend(lines[:max_lines])
                print(f"DEBUG: Reading '{file}', taking up to {max_lines} lines.")
        except UnicodeDecodeError:
            print(f"WARNING: Could not read file {file} due to encoding issues. Skipping.")
            continue

    logs_content = "\n".join(combined_logs)
    if not logs_content.strip():
        print("ERROR: No relevant logs found for analysis.")
        return []
    
    prompt_base, _ = generate_prompt("success", report_language)
    prompt = f"{prompt_base}\n\nLogs:\n{logs_content}"
    print("DEBUG: Sending prompt for recommendations to OpenAI...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )
        ai_text = response.choices[0].message.content.strip()
        print("DEBUG: AI returned text for recommendations:\n", ai_text)

        recs = parse_recommendations(ai_text)
        print(f"DEBUG: Returning {len(recs)} recommendation(s) (including those with empty descriptions).")
        return recs

    except Exception as e:
        print(f"ERROR: Failed to analyze logs for recommendations: {e}")
        return []

# ============ ANALIZAR LOGS (FAILURE) ============

def analyze_logs_with_ai(log_dir, log_type, report_language, project_name):
    print(f"DEBUG: analyze_logs_with_ai(log_dir={log_dir}, log_type={log_type}, language='{report_language}', project='{project_name}')")
    log_files = validate_logs_directory(log_dir)
    combined_logs = []
    max_lines = 300

    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                combined_logs.extend(lines[:max_lines])
                print(f"DEBUG: Reading '{file}', taking up to {max_lines} lines.")
        except UnicodeDecodeError:
            print(f"WARNING: Could not read file {file} due to encoding issues. Skipping.")
            continue

    logs_content = "\n".join(combined_logs)
    if not logs_content.strip():
        print("ERROR: No relevant logs found for analysis.")
        return None, None, None

    print("DEBUG: Generating prompt and calling OpenAI for error ticket...")
    prompt, issue_type = generate_prompt(log_type, report_language)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant generating concise Jira tickets. "
                    "Use short, direct statements, some emojis (e.g. 🐞, 💥, etc.), minimal markdown. "
                    "Avoid triple backticks for code unless strictly necessary."
                )},
                {"role": "user", "content": f"{prompt}\n\nLogs:\n{logs_content}"}
            ],
            max_tokens=600,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
        print(f"DEBUG: AI returned summary of length {len(summary)} chars.")

        lines = summary.splitlines()
        first_line = lines[0] if lines else "No Title"
        cleaned_title_line = sanitize_title(first_line.replace("```markdown", "").replace("```", "").strip())

        icon = choose_error_icon()
        summary_title = f"{project_name} {icon} {cleaned_title_line}"

        description_plain = unify_double_to_single_asterisks(summary.replace("\t", " "))
        print(f"DEBUG: Final summary title -> {summary_title}")
        print(f"DEBUG: Description length -> {len(description_plain)} chars.")

        return summary_title, description_plain, issue_type

    except Exception as e:
        print(f"ERROR: Failed to analyze logs with AI: {e}")
        return None, None, None

# ============ PROCESO PRINCIPAL ============

def main():
    parser = argparse.ArgumentParser(description="Crear tickets en JIRA desde logs analizados.")
    parser.add_argument("--jira-url", required=True, help="URL de la instancia de JIRA.")
    parser.add_argument("--jira-project-key", required=True, help="Clave del proyecto en JIRA.")
    parser.add_argument("--log-dir", required=True, help="Directorio con los logs.")
    parser.add_argument("--log-type", required=True, choices=["success", "failure"], help="Tipo de log.")
    parser.add_argument("--report-language", default="English", help="Idioma para el resumen del reporte.")
    parser.add_argument("--project-name", required=True, help="Nombre del repositorio en GitHub.")
    parser.add_argument("--run-id", required=True, help="ID de la ejecución en GitHub Actions.")
    parser.add_argument("--repo", required=True, help="Nombre completo del repositorio (owner/repo).")
    args = parser.parse_args()

    print("DEBUG: Starting main process with arguments:", args)
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user_email = os.getenv("JIRA_USER_EMAIL")
    if not all([jira_api_token, jira_user_email]):
        print("ERROR: Missing required environment variables JIRA_API_TOKEN or JIRA_USER_EMAIL.")
        exit(1)

    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    if args.log_type == "failure":
        # Caso de fallo
        summary, description, issue_type = analyze_logs_with_ai(
            args.log_dir, args.log_type, args.report_language, args.project_name
        )
        if not summary or not description or not issue_type:
            print("ERROR: Log analysis failed or invalid issue type. No ticket will be created.")
            return

        print(f"DEBUG: Proposed summary -> '{summary}'\nDEBUG: Proposed issue_type -> '{issue_type}'")
        try:
            validate_issue_type(args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, issue_type)
        except Exception as e:
            print(f"ERROR: {e}")
            return

        existing_ticket_key = check_existing_tickets_local_and_ia_summary_desc(
            jira, args.jira_project_key, summary, description, issue_type
        )
        if existing_ticket_key:
            print(f"INFO: Ticket already exists: {existing_ticket_key}. Skipping creation.")
            return

        print("DEBUG: Creating ticket for failure in Jira...")
        ticket_key = create_jira_ticket(jira, args.jira_project_key, summary, description, issue_type)
        if ticket_key:
            print(f"INFO: JIRA Ticket Created: {ticket_key}")
        else:
            print("WARNING: Falling back to creating ticket via API...")
            ticket_key = create_jira_ticket_via_requests(
                args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, summary, description, issue_type
            )
            if ticket_key:
                print(f"INFO: JIRA Ticket Created via API: {ticket_key}")
            else:
                print("ERROR: Failed to create JIRA ticket.")

    else:
        # Caso "success"
        recommendations = analyze_logs_for_recommendations(
            args.log_dir, args.report_language, args.project_name
        )
        if not recommendations:
            print("INFO: No recommendations generated by the AI.")
            return

        print(f"DEBUG: {len(recommendations)} recommendation(s) parsed from AI response.")
        for i, rec in enumerate(recommendations, start=1):
            r_summary = rec["summary"]
            r_desc = rec["description"]

            # Si la descripción está realmente vacía, no creamos
            if not r_desc.strip():
                print(f"DEBUG: Recommendation #{i} has empty description. Skipping.")
                continue

            # Filtrar si es algo que ya usamos (por ejemplo Bandit, npm audit, etc)
            if should_skip_recommendation(r_summary, r_desc):
                print(f"INFO: Recommendation #{i} references existing tool. Skipping creation.")
                continue

            issue_type = "Tarea"
            print(f"DEBUG: Processing recommendation #{i} -> Summary: '{r_summary}'")

            dup_key = check_existing_tickets_local_and_ia_summary_desc(
                jira, args.jira_project_key, r_summary, r_desc, issue_type
            )
            if dup_key:
                print(f"INFO: Recommendation #{i} already exists in ticket {dup_key}. Skipping creation.")
                continue

            final_title, wiki_description = format_ticket_content(
                args.project_name, r_summary, r_desc, "Improvement"
            )
            print("DEBUG: Final title:", final_title)
            print("DEBUG: Final wiki description:\n", wiki_description)

            if not wiki_description.strip():
                print(f"DEBUG: Recommendation #{i} final wiki description is empty. Skipping ticket creation.")
                continue

            new_key = create_jira_ticket(
                jira, args.jira_project_key, final_title, wiki_description, issue_type
            )
            if new_key:
                print(f"INFO: Created ticket for recommendation #{i}: {new_key}")
            else:
                print(f"WARNING: Failed to create ticket for recommendation #{i} via library, attempting fallback...")
                fallback_key = create_jira_ticket_via_requests(
                    args.jira_url, jira_user_email, jira_api_token,
                    args.jira_project_key, final_title, wiki_description, issue_type
                )
                if fallback_key:
                    print(f"INFO: Created ticket for recommendation #{i} via API: {fallback_key}")
                else:
                    print(f"ERROR: Could not create ticket for recommendation #{i}.")

    print("DEBUG: Process finished.")

if __name__ == "__main__":
    main()
