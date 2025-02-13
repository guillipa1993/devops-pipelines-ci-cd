#!/usr/bin/env python3
import os
import argparse
import tarfile
import requests
import re
import json
from jira import JIRA
from openai import OpenAI
from datetime import datetime
from difflib import SequenceMatcher

# ============ CONFIGURACIÓN OPENAI ============
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
    print(f"DEBUG: Sanitizing summary: '{summary}'")
    sanitized = "".join(c for c in summary if c.isalnum() or c.isspace())
    print(f"DEBUG: Resulting sanitized summary: '{sanitized}'")
    return sanitized

def preprocess_text(text):
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    lowered = text_no_punct.strip().lower()
    return lowered

def calculate_similarity(text1, text2):
    t1 = preprocess_text(text1)
    t2 = preprocess_text(text2)
    ratio = SequenceMatcher(None, t1, t2).ratio()
    return ratio

# ============ FUNCIÓN PARA CONVERTIR ADF A TEXTO PLANO ============
def convert_adf_to_plain_text(adf):
    """
    Recorre la estructura ADF y extrae el contenido de texto de cada nodo,
    devolviendo un string formateado en Markdown.
    """
    def process_node(node):
        if "text" in node and "type" not in node:  # nodo de texto simple
            return node["text"]
        node_type = node.get("type", "")
        if node_type == "text":
            return node.get("text", "")
        elif node_type == "paragraph":
            texts = [process_node(child) for child in node.get("content", [])]
            return " ".join(texts)
        elif node_type == "bulletList":
            items = []
            for item in node.get("content", []):
                item_texts = [process_node(child) for child in item.get("content", [])]
                items.append("- " + " ".join(item_texts))
            return "\n".join(items)
        elif node_type == "codeBlock":
            code_texts = [process_node(child) for child in node.get("content", [])]
            return "```\n" + "\n".join(code_texts) + "\n```"
        elif "content" in node:  # caso genérico para nodos con 'content'
            return " ".join(process_node(child) for child in node["content"])
        else:
            return ""

    if adf.get("content"):
        paragraphs = [process_node(node) for node in adf["content"]]
        return "\n\n".join(paragraphs)
    return ""

def adf_to_plain_text(description):
    """
    Convierte un objeto ADF (dict) a un string legible en formato Markdown.
    Si 'description' ya es una cadena, se devuelve tal cual.
    """
    if isinstance(description, str):
        return description
    try:
        return convert_adf_to_plain_text(description)
    except Exception as e:
        print(f"WARNING: Failed to convert ADF to plain text: {e}")
        return json.dumps(description, ensure_ascii=False)

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

# ============ FORMATEO FINAL DEL CONTENIDO DEL TICKET ============
def format_ticket_content(project_name, rec_summary, rec_description, ticket_category):
    """
    Llama a la IA para formatear el contenido final del ticket.
    Se espera que la IA devuelva un JSON con dos claves:
      - "title": Una oración concisa que comience con el nombre del proyecto y contenga un emoticono adecuado.
      - "description": Un contenido formateado en Atlassian Document Format (ADF); se usarán triple backticks sin lenguaje.
    """
    prompt = (
        "You are a professional technical writer formatting Jira tickets for developers. "
        "Given the following recommendation details, produce a JSON object with two keys: 'title' and 'description'.\n\n"
        "The 'title' must be a single concise sentence that starts with the project name as a prefix and includes an appropriate emoticon based on the ticket category "
        "(for example, use '🚀' for improvements, '🐞' for bugs, '💡' for suggestions, etc.).\n\n"
        "The 'description' must be formatted in Atlassian Document Format (ADF). It should be a JSON object with:\n"
        "  - \"type\": \"doc\",\n"
        "  - \"version\": 1,\n"
        "  - \"content\": an array of nodes (e.g., paragraphs, code blocks, bullet lists) that include detailed technical information and code examples. "
        "When formatting code blocks, use triple backticks without a language specifier.\n\n"
        "Do not include redundant labels like 'Summary:' or 'Description:' in the output.\n\n"
        f"Project: {project_name}\n"
        f"Recommendation Title: {rec_summary}\n"
        f"Recommendation Details: {rec_description}\n"
        f"Ticket Category: {ticket_category}\n\n"
        "Return only a valid JSON object."
    )
    print("DEBUG: format_ticket_content prompt:\n", prompt)
    try:
        # CHANGED: raised max_tokens to allow more text if needed
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional technical writer."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,  # CHANGED: slightly higher
            temperature=0.3
        )
        ai_output = response.choices[0].message.content.strip()
        print("DEBUG: Raw AI output from format_ticket_content:\n", ai_output)

        # Eliminar backticks si la IA los devolviese
        if ai_output.startswith("```"):
            lines = ai_output.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            ai_output = "\n".join(lines).strip()
        print("DEBUG: AI output after stripping Markdown delimiters:\n", ai_output)

        # Convertir la respuesta a JSON
        ticket_json = json.loads(ai_output)

        # Extraer título y descripción ADF
        final_title = ticket_json.get("title", f"{project_name}: {rec_summary}")
        final_description = ticket_json.get("description", rec_description)

        if not final_description:
            print("WARNING: Final description is empty. Using original recommendation description as fallback.")
            final_description = rec_description

        return final_title, final_description

    except Exception as e:
        print(f"WARNING: Failed to format ticket content with AI: {e}")
        return f"{project_name}: {rec_summary}", rec_description

# ============ FUNCIÓN PARA CONVERTIR DESCRIPCIÓN A TEXTO LEGIBLE ============
# (Nota: se mantiene para posibles usos, pero ya NO se usa antes de crear el ticket)
def adf_to_plain_text(description):
    if isinstance(description, str):
        return description
    try:
        return convert_adf_to_plain_text(description)
    except Exception as e:
        print(f"WARNING: Failed to convert ADF to plain text: {e}")
        return json.dumps(description, ensure_ascii=False)

# ============ BÚSQUEDA DE TICKETS EXISTENTES (LOCAL + IA) ============
def check_existing_tickets_local_and_ia_summary_desc(jira, project_key, new_summary, new_description, issue_type):
    print("DEBUG: Checking for existing tickets (local + IA, summary + description)")
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9
    sanitized_summary = sanitize_summary(new_summary)
    print(f"DEBUG: sanitized_summary='{sanitized_summary}'")

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

        # Filtros de similaridad básica
        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            print(f"DEBUG: Both summary_sim and description_sim < {LOCAL_SIM_LOW:.2f}; ignoring {issue_key}.")
            continue

        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            print(f"INFO: Found duplicate ticket {issue_key} (high local similarity).")
            return issue_key

        # Usar IA para confirmar
        print(f"DEBUG: Intermediate range for {issue_key}. Asking IA for final check...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
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
            # fallback local check
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
    Crea el ticket en Jira usando la librería python-jira. 
    AHORA NO convertimos la descripción a texto plano de forma automática.
    """
    try:
        issue_dict = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,  # Se admite ADF si tu Jira Cloud lo soporta
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
    Crea un ticket en Jira vía API REST. 
    Si description es un dict (ADF), se pasa tal cual en JSON. 
    Si es string, también se pasa como string.
    """
    if isinstance(description, dict):
        # Se pasa el dict como JSON en el payload
        desc_payload = description
    else:
        desc_payload = description

    url = f"{jira_url}/rest/api/3/issue"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": desc_payload,
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
            "Avoid triple backticks unless strictly necessary, and keep the use of emojis minimal but clear."
        )
        issue_type = "Error"
    else:
        # CHANGED: Ajuste para recalcar la necesidad de detallar ejemplos
        details = f"""
You are a code reviewer and linter specialized in Python. 
Below are the combined logs from a successful build, which may include minor warnings or best practices. 
Analyze them thoroughly and propose *specific code improvements or refactors* with the following format:
  
- Title (bold)
- Summary (1 line)
- Description (detailed explanation, referencing actual lines or snippets where possible, and showing small code examples)

Your suggestions MUST be derived from any warnings, messages, or potential optimization hints seen in the logs. 
Focus on real issues: PEP 8 violations, performance bottlenecks, or readability improvements. 
Avoid generic placeholders. 
Write in {language} using clear and concise technical language.
"""
        issue_type = "Tarea"
    print(f"DEBUG: Prompt generated. Issue type = {issue_type}")
    return details, issue_type

# ============ FUNCIÓN PARA UNIFICAR ASTERISCOS ============
def unify_double_to_single_asterisks(description):
    print("DEBUG: Unifying double asterisks to single...")
    while '**' in description:
        description = description.replace('**', '*')
    return description

def sanitize_title(title):
    print(f"DEBUG: Sanitizing title '{title}'...")
    title = re.sub(r"[\*`]+", "", title)
    title = title.strip()
    print(f"DEBUG: Title after sanitize -> '{title}'")
    return title

# ============ MÉTODO PARA ANALIZAR LOGS EN CASO DE ÉXITO (RECOMENDACIONES) ============
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
            model="gpt-4o-mini",
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

# ============ MÉTODO PARA ANALIZAR LOGS EN CASO DE ERROR ============
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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": (
                    "You are a helpful assistant generating concise Jira tickets. "
                    "Use short, direct statements and minimal markdown formatting. "
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

        label = "Error" if log_type == "failure" else "Success"
        summary_title = f"{project_name}: {label} - {cleaned_title_line}"

        # Unificar asteriscos duplicados para no romper el markdown
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

        print("DEBUG: Checking for existing tickets (failure)...")
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
        recommendations = analyze_logs_for_recommendations(args.log_dir, args.report_language, args.project_name)
        if not recommendations:
            print("INFO: No recommendations generated by the AI.")
            return

        print(f"DEBUG: {len(recommendations)} recommendation(s) parsed from AI response.")
        for i, rec in enumerate(recommendations, start=1):
            r_summary = rec["summary"]
            r_desc = rec["description"]
            issue_type = "Tarea"
            print(f"DEBUG: Processing recommendation #{i} -> Summary: '{r_summary}'")

            dup_key = check_existing_tickets_local_and_ia_summary_desc(
                jira, args.jira_project_key, r_summary, r_desc, issue_type
            )
            if dup_key:
                print(f"INFO: Recommendation #{i} already exists in ticket {dup_key}. Skipping creation.")
            else:
                final_title, final_description = format_ticket_content(
                    args.project_name, r_summary, r_desc, "Improvement"
                )
                # ADDED: Si la librería python-jira no soporta dict en "description",
                # podría usarse la vía requests. Aquí creamos directamente con la librería:
                print("DEBUG: Final title:", final_title)
                print("DEBUG: Final description (ADF or str):", final_description)

                # En caso de querer texto plano, puedes descomentar:
                # plain_description = adf_to_plain_text(final_description)

                new_key = create_jira_ticket(jira, args.jira_project_key, final_title, final_description, issue_type)
                if new_key:
                    print(f"INFO: Created ticket for recommendation #{i}: {new_key}")
                else:
                    print(f"WARNING: Failed to create ticket for recommendation #{i} via library, attempting fallback...")
                    fallback_key = create_jira_ticket_via_requests(
                        args.jira_url, jira_user_email, jira_api_token,
                        args.jira_project_key, final_title, final_description, issue_type
                    )
                    if fallback_key:
                        print(f"INFO: Created ticket for recommendation #{i} via API: {fallback_key}")
                    else:
                        print(f"ERROR: Could not create ticket for recommendation #{i}.")

    print("DEBUG: Process finished.")

if __name__ == "__main__":
    main()
