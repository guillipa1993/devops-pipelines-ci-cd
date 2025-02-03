#!/usr/bin/env python3
import os
import argparse
import tarfile
import requests
import re
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
    """
    Limpia el resumen para eliminar caracteres que puedan causar problemas en el JQL.
    """
    print(f"DEBUG: Sanitizing summary: '{summary}'")
    sanitized = "".join(c for c in summary if c.isalnum() or c.isspace())
    print(f"DEBUG: Resulting sanitized summary: '{sanitized}'")
    return sanitized

def preprocess_text(text):
    """
    Quita puntuación y espacios extra, lleva a minúsculas.
    Se usa para comparar similitud con SequenceMatcher.
    """
    print("DEBUG: Preprocessing text for similarity comparison...")
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    lowered = text_no_punct.strip().lower()
    return lowered

def calculate_similarity(text1, text2):
    """
    Calcula la similitud usando SequenceMatcher, retornando un valor entre 0 y 1.
    """
    print("DEBUG: Calculating local similarity with SequenceMatcher...")
    t1 = preprocess_text(text1)
    t2 = preprocess_text(text2)
    ratio = SequenceMatcher(None, t1, t2).ratio()
    print(f"DEBUG: Similarity ratio = {ratio:.2f}")
    return ratio

# ============ PARSEO DE RECOMENDACIONES ============
def parse_recommendations(ai_text):
    """
    Parses the AI text (for log_type 'success') and extracts a list of recommendations.
    Expected format is a series of bullet points starting with a line like:
    
    - **Title**: 
      - Summary: <summary text>
      - Description: <description text>
    
    However, if the block does not contain explicit "Summary:" and "Description:" labels,
    this function will instead split the block into lines, take the first non-empty line as the summary,
    and the rest as the description.
    
    Returns a list of dictionaries with keys "summary" and "description".
    """
    recommendations = []
    
    # Primero, eliminamos cualquier encabezado como "### Recommendations" si existe.
    ai_text = ai_text.strip()
    ai_text = re.sub(r"^#+\s*Recommendations\s*", "", ai_text, flags=re.IGNORECASE).strip()
    
    # Patrón que captura cada bloque que empieza con "- **Title**:" y su contenido hasta el siguiente bloque o el final del texto.
    pattern = re.compile(
        r"(?ms)^\s*-\s*\*\*(?P<title>.+?)\*\*:\s*(?:\n\s*)?(?P<content>.*?)(?=^\s*-\s*\*\*|\Z)",
        re.MULTILINE | re.DOTALL
    )
    
    for match in pattern.finditer(ai_text):
        title = match.group("title").strip()
        content = match.group("content").strip()
        
        # Intentar extraer explícitamente las etiquetas "Summary:" y "Description:" si están presentes
        summary_match = re.search(r"(?i)^\s*-\s*Summary:\s*(?P<summary>.+)", content, re.MULTILINE)
        description_match = re.search(r"(?i)^\s*-\s*Description:\s*(?P<description>.+)", content, re.MULTILINE | re.DOTALL)
        
        if summary_match and description_match:
            rec_summary = summary_match.group("summary").strip()
            rec_description = description_match.group("description").strip()
        else:
            # Si no hay etiquetas explícitas, separamos el bloque en líneas no vacías
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            if lines:
                rec_summary = lines[0]
                rec_description = " ".join(lines[1:]).strip() if len(lines) > 1 else ""
            else:
                rec_summary = ""
                rec_description = ""
        
        # Combinar el título con el resumen si éste existe, para formar un resumen completo
        full_summary = f"{title}: {rec_summary}" if rec_summary else title
        
        if full_summary or rec_description:
            recommendations.append({
                "summary": full_summary,
                "description": rec_description
            })
        else:
            print("WARNING: Could not parse a recommendation block:", match.group(0))
    
    print(f"DEBUG: Parsed {len(recommendations)} recommendation(s).")
    return recommendations

# ============ BÚSQUEDA DE TICKETS EXISTENTES (COMBINANDO LOCAL + IA) ============
def check_existing_tickets_local_and_ia_summary_desc(jira, project_key, new_summary, new_description, issue_type):
    """
    Verifica duplicados combinando comparaciones locales (título y descripción)
    con IA de manera escalonada:
      - Si la similitud local en summary y description es muy alta (>= 0.9) => duplicado inmediato.
      - Si ambas son muy bajas (< 0.3) => se descarta el candidato.
      - En rango intermedio, se pregunta a la IA.
    Devuelve la key del primer ticket duplicado o None si no se encontró ninguno.
    """
    print("DEBUG: Checking for existing tickets (local + IA, summary + description)")
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9

    sanitized_summary = sanitize_summary(new_summary)
    print(f"DEBUG: sanitized_summary='{sanitized_summary}'")
    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)
    jql_query = (
        f'project = "{project_key}" '
        f'AND issuetype = "{issue_type}" '
        f'AND status IN ({states_str})'
    )
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
        print(f"DEBUG: desc_sim with {issue_key} = {desc_sim:.2f}")
        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            print(f"DEBUG: Both summary_sim and desc_sim < {LOCAL_SIM_LOW:.2f}; ignoring {issue_key}.")
            continue
        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            print(f"INFO: Found duplicate ticket {issue_key} with high local similarity (summary_sim={summary_sim:.2f}, desc_sim={desc_sim:.2f}).")
            return issue_key
        print(f"DEBUG: Intermediate range for {issue_key} (summary_sim={summary_sim:.2f}, desc_sim={desc_sim:.2f}). Asking IA...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an assistant specialized in analyzing text similarity. Respond only 'yes' or 'no'."},
                    {"role": "user",
                     "content": (
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
                print(f"DEBUG: AI says 'no' for {issue_key}. Continuing with next candidate.")
                continue
            else:
                print(f"WARNING: Ambiguous AI response '{ai_result}' for {issue_key}; continuing.")
                continue
        except Exception as e:
            print(f"WARNING: IA call failed for {issue_key}: {e}")
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                print(f"INFO: Fallback local: similarity >= 0.8 for {issue_key}; marking as duplicate.")
                return issue_key
            else:
                print(f"DEBUG: Similarity for {issue_key} not high enough for fallback; ignoring.")
                continue
    print("DEBUG: No duplicate ticket found after local+IA approach.")
    return None

def check_existing_tickets_ia_only(jira, project_key, summary, description, issue_type):
    """
    Verifica si existe un ticket con un resumen parecido y luego llama a la IA para comparar
    las descripciones en detalle. Retorna la key del primer ticket duplicado o None.
    """
    print("DEBUG: Checking for existing tickets with IA as primary comparator...")
    sanitized_summary = sanitize_summary(summary)
    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)
    jql_query = (
        f'project = "{project_key}" '
        f'AND issuetype = "{issue_type}" '
        f'AND summary ~ "{sanitized_summary}" '
        f'AND status IN ({states_str})'
    )
    print(f"DEBUG: JQL -> {jql_query}")
    try:
        issues = jira.search_issues(jql_query)
        print(f"DEBUG: Found {len(issues)} candidate issue(s).")
    except Exception as e:
        print(f"ERROR: Failed to execute JQL query: {e}")
        return None
    for issue in issues:
        existing_description = issue.fields.description or ""
        print(f"DEBUG: Using IA to compare with Issue {issue.key}...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an assistant specialized in analyzing text similarity. Respond only 'yes' or 'no'."},
                    {"role": "user",
                     "content": (
                         "Compare the following two descriptions in terms of meaning, ignoring language differences.\n\n"
                         f"Existing Description:\n{existing_description}\n\n"
                         f"New Description:\n{description}"
                     )}
                ],
                max_tokens=200,
                temperature=0.3
            )
            ai_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: AI comparison result for {issue.key}: '{ai_result}'")
            if ai_result.startswith("yes"):
                print(f"INFO: Found duplicate ticket (IA confirms) -> {issue.key}")
                return issue.key
            elif ai_result.startswith("no"):
                print(f"DEBUG: AI says 'no' for {issue.key}. Continuing.")
                continue
            else:
                print(f"WARNING: Ambiguous AI response '{ai_result}' for {issue.key}. Continuing.")
                continue
        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with IA for {issue.key}: {e}")
            continue
    print("DEBUG: No duplicate ticket found after IA comparisons.")
    return None

def check_existing_tickets(jira, project_key, summary, description, issue_type):
    """
    Verifica si existe un ticket con un resumen o descripción similar en Jira,
    combinando una comparación local y el uso de IA.
    Retorna la key del primer ticket duplicado o None si no se encontró ninguno.
    """
    print("DEBUG: Checking for existing tickets (local + IA approach)...")
    LOCAL_SIMILARITY_THRESHOLD = 0.75
    LOCAL_FALLBACK_THRESHOLD = 0.70

    sanitized_summary = sanitize_summary(summary)
    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)
    jql_query = (
        f'project = "{project_key}" '
        f'AND issuetype = "{issue_type}" '
        f'AND summary ~ "{sanitized_summary}" '
        f'AND status IN ({states_str})'
    )
    print(f"DEBUG: JQL Query -> {jql_query}")
    try:
        issues = jira.search_issues(jql_query)
        print(f"DEBUG: Found {len(issues)} candidate issue(s).")
    except Exception as e:
        print(f"ERROR: Failed to execute JQL query: {e}")
        return None

    for issue in issues:
        existing_description = issue.fields.description or ""
        print(f"DEBUG: Analyzing Issue {issue.key} with local similarity check...")
        local_similarity = calculate_similarity(description, existing_description)
        print(f"DEBUG: local_similarity with {issue.key} = {local_similarity:.2f}")
        if local_similarity >= LOCAL_SIMILARITY_THRESHOLD:
            print(f"INFO: Found duplicate ticket (local similarity {local_similarity:.2f}) -> {issue.key}")
            return issue.key
        print(f"DEBUG: Using IA to compare with Issue {issue.key} if local similarity < {LOCAL_SIMILARITY_THRESHOLD:.2f}")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are an assistant specialized in analyzing text similarity. Respond only 'yes' or 'no'."},
                    {"role": "user",
                     "content": (
                         "Compare these two descriptions in terms of meaning, ignoring language differences. "
                         "If they describe the same or very similar issue, respond with 'yes'. Otherwise, respond with 'no'.\n\n"
                         f"Existing description:\n{existing_description}\n\n"
                         f"New description:\n{description}"
                     )}
                ],
                max_tokens=200,
                temperature=0.3
            )
            ai_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: AI comparison result for {issue.key}: '{ai_result}'")
            if ai_result.startswith("yes"):
                print(f"INFO: Found duplicate ticket (AI confirms) -> {issue.key}")
                return issue.key
            elif ai_result.startswith("no"):
                print(f"DEBUG: AI says 'no' for {issue.key}. Continuing with next candidate.")
                continue
            else:
                print(f"WARNING: AI ambiguous response '{ai_result}' for {issue.key}.")
                if local_similarity >= LOCAL_FALLBACK_THRESHOLD:
                    print(f"WARNING: Fallback local: similarity {local_similarity:.2f} => Marking {issue.key} as duplicate.")
                    return issue.key
        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with AI for {issue.key}: {e}")
            if local_similarity >= LOCAL_SIMILARITY_THRESHOLD:
                print(f"INFO: Fallback local (similarity {local_similarity:.2f}) => {issue.key}")
                return issue.key
    print("DEBUG: No duplicate ticket found after checking all candidates.")
    return None

# ============ CREACIÓN DE TICKETS ============
def create_jira_ticket_via_requests(jira_url, jira_user, jira_api_token, project_key, summary, description, issue_type):
    print("DEBUG: Creating Jira ticket via REST API (requests)...")
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
    print(f"DEBUG: Payload -> {payload}")
    response = requests.post(url, json=payload, headers=headers, auth=auth)
    if response.status_code == 201:
        print("DEBUG: Ticket created successfully via API.")
        print("Ticket created successfully:", response.json())
        return response.json().get("key")
    else:
        print(f"ERROR: Failed to create ticket via API: {response.status_code} - {response.text}")
        return None

def create_jira_ticket(jira, project_key, summary, description, issue_type):
    print("DEBUG: Creating Jira ticket via JIRA library...")
    try:
        issue_dict = {
            'project': {'key': project_key},
            'summary': summary,
            'description': description,
            'issuetype': {'name': issue_type}
        }
        print(f"DEBUG: Issue fields -> {issue_dict}")
        issue = jira.create_issue(fields=issue_dict)
        print("DEBUG: Ticket created successfully via JIRA library.")
        return issue.key
    except Exception as e:
        print(f"ERROR: Could not create ticket via JIRA library: {e}")
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
        valid_types = [issue["name"] for issue in response.json()["projects"][0]["issuetypes"]]
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
            "1) **Summary** ❗: A single-sentence overview of the main issue.\n"
            "2) **Root Cause Analysis** 🔍: Briefly state the cause. Include log snippets if crucial.\n"
            "3) **Proposed Solutions** 🛠️: List concrete steps to fix the issue, using bullets or short paragraphs.\n"
            "4) **Preventive Measures** ⛑️: Suggest ways to avoid recurrence. Keep it succinct.\n"
            "5) **Impact Analysis** ⚠️: What happens if it's not addressed?\n\n"
            "Avoid triple backticks unless strictly necessary, and keep the use of emojis minimal but clear."
        )
        issue_type = "Error"
    else:
        details = (
            "You are a technical writer creating one or more recommendations in a concise Jira Cloud ticket from logs. "
            "Please list separate recommendations as bullet points. Each bullet should have a short summary and a brief description. "
            "Keep the format short and professional, using minimal Markdown. "
            "Focus on these sections:\n\n"
            "1) **Summary** ✅: A single-sentence overview of the successful outcome.\n"
            "2) **Success Details** 🚀: Important tasks or milestones achieved.\n"
            "3) **Recommendations** 💡: Suggested optimizations or scalability measures.\n"
            "4) **Impact** 🌟: Positive effects or benefits of this success.\n\n"
            "Avoid triple backticks unless strictly necessary, and keep the use of emojis minimal but clear."
        )
        issue_type = "Tarea"
    prompt = (
        f"{details}\n\n"
        f"Be concise, professional, and compatible with Jira Cloud's Markdown. "
        f"Write the ticket in {language}."
    )
    print(f"DEBUG: Prompt generated. Issue type = {issue_type}")
    return prompt, issue_type

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
    
    # Prompt especial para generar recomendaciones
    prompt = (
        f"You are a helpful assistant. The build logs indicate a successful build. "
        f"Please list separate recommendations as bullet points. Each bullet should have a short summary and a brief description. "
        f"Write in {report_language}. Avoid triple backticks.\n\nLogs:\n{logs_content}"
    )
    print("DEBUG: Sending prompt for recommendations to OpenAI...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.5
        )
        ai_text = response.choices[0].message.content.strip()
        print("DEBUG: AI returned text for recommendations:\n", ai_text)
        recs = parse_recommendations(ai_text)
        print(f"DEBUG: Parsed {len(recs)} recommendation(s).")
        return recs
    except Exception as e:
        print(f"ERROR: Failed to analyze logs for recommendations: {e}")
        return []

# ============ MÉTODO PARA ANALIZAR LOGS EN CASO DE ERROR ============
def analyze_logs_with_ai(log_dir, log_type, report_language, project_name):
    print(f"DEBUG: analyze_logs_with_ai(log_dir={log_dir}, log_type={log_type}, language={report_language}, project={project_name})")
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
        cleaned_title_line = (
            first_line
            .replace("```markdown", "")
            .replace("```", "")
            .strip()
        )
        cleaned_title_line = sanitize_title(cleaned_title_line)
        label = "Error" if log_type == "failure" else "Success"
        summary_title = f"{project_name}: {label} - {cleaned_title_line}"
        description_plain = summary.replace("\t", " ")
        description_plain = unify_double_to_single_asterisks(description_plain)
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
        ticket_key = create_jira_ticket(
            jira, args.jira_project_key, summary, description, issue_type
        )
        if ticket_key:
            print(f"INFO: JIRA Ticket Created: {ticket_key}")
        else:
            print("WARNING: Falling back to creating ticket via API...")
            ticket_key = create_jira_ticket_via_requests(
                args.jira_url, jira_user_email, jira_api_token, args.jira_project_key,
                summary, description, issue_type
            )
            if ticket_key:
                print(f"INFO: JIRA Ticket Created via API: {ticket_key}")
            else:
                print("ERROR: Failed to create JIRA ticket.")
    else:
        # Caso "success": generar recomendaciones y crear tickets para cada mejora
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
            issue_type = "Tarea"
            print(f"DEBUG: Processing recommendation #{i} -> Summary: '{r_summary}'")
            dup_key = check_existing_tickets_local_and_ia_summary_desc(
                jira, args.jira_project_key, r_summary, r_desc, issue_type
            )
            if dup_key:
                print(f"INFO: Recommendation #{i} already exists in ticket {dup_key}. Skipping creation.")
            else:
                new_key = create_jira_ticket(
                    jira, args.jira_project_key, r_summary, r_desc, issue_type
                )
                if new_key:
                    print(f"INFO: Created ticket for recommendation #{i}: {new_key}")
                else:
                    print(f"WARNING: Failed to create ticket for recommendation #{i} via library, attempting fallback...")
                    fallback_key = create_jira_ticket_via_requests(
                        args.jira_url, jira_user_email, jira_api_token, args.jira_project_key,
                        r_summary, r_desc, issue_type
                    )
                    if fallback_key:
                        print(f"INFO: Created ticket for recommendation #{i} via API: {fallback_key}")
                    else:
                        print(f"ERROR: Could not create ticket for recommendation #{i}.")
    print("DEBUG: Process finished.")

if __name__ == "__main__":
    main()
