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
    print(f"DEBUG: Preprocessing text for similarity comparison...")
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

# ============ BÚSQUEDA DE TICKETS EXISTENTES (FILTRANDO POR ISSUE_TYPE) ============

def check_existing_tickets_local_and_ia_summary_desc(
    jira,
    project_key,
    new_summary,       # El summary propuesto para el nuevo ticket
    new_description,   # La descripción generada para el nuevo ticket
    issue_type
):
    """
    Verifica duplicados combinando comparaciones locales (título y descripción)
    con IA de manera escalonada:

      1) JQL para filtrar issues con un summary parecido ('summary ~ sanitized_summary').
      2) Comparación local en summary y description:
         - Si la similitud en ambos es >= 0.9 => duplicado inmediato
         - Si ambos < 0.3 => descartar
         - Resto => IA decide.
      3) IA: se le pasa la descripción actual y la del ticket existente. Responde 'yes' / 'no'.

    Devuelve la key del primer ticket considerado duplicado, o None si no se encontró.
    Con debug para entender el proceso.
    """

    print("DEBUG: Checking for existing tickets (local + IA, summary + description)")

    # Umbrales
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9

    # 1) JQL: filtrar por summary ~ sanitized_summary
    sanitized_summary = sanitize_summary(new_summary)
    print(f"DEBUG: sanitized_summary='{sanitized_summary}'")

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

    # 2) Para cada candidate issue, comparamos
    for issue in issues:
        issue_key = issue.key
        existing_summary = issue.fields.summary or ""
        existing_description = issue.fields.description or ""

        print(f"DEBUG: Analyzing Issue {issue_key}")

        # 2A) Similaridad local en summary
        summary_sim = calculate_similarity(new_summary, existing_summary)
        print(f"DEBUG: summary_sim with {issue_key} = {summary_sim:.2f}")

        # 2B) Similaridad local en description
        desc_sim = calculate_similarity(new_description, existing_description)
        print(f"DEBUG: desc_sim with {issue_key} = {desc_sim:.2f}")

        # CASO A: Ambos muy bajos => descartar
        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            print(
                f"DEBUG: Both summary_sim={summary_sim:.2f} and desc_sim={desc_sim:.2f} "
                f"< {LOCAL_SIM_LOW}, ignoring {issue_key}."
            )
            continue

        # CASO B: Ambos muy altos => duplicado sin IA
        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            print(
                f"INFO: Found existing ticket {issue_key} (summary_sim={summary_sim:.2f}, desc_sim={desc_sim:.2f} >= {LOCAL_SIM_HIGH}). "
                "Marking as duplicate without IA."
            )
            return issue_key

        # CASO C: Resto => IA decide
        print(
            f"DEBUG: Intermediate range -> summary_sim={summary_sim:.2f}, "
            f"desc_sim={desc_sim:.2f}. Asking IA about {issue_key}..."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant specialized in analyzing text similarity. "
                            "You respond only with 'yes' or 'no' to indicate whether the two issues match in meaning."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "We have two issues:\n\n"
                            f"Existing issue (summary + description):\n"
                            f"Summary:\n{existing_summary}\n\n"
                            f"Description:\n{existing_description}\n\n"
                            "New issue (summary + description):\n"
                            f"Summary:\n{new_summary}\n\n"
                            f"Description:\n{new_description}\n\n"
                            "If they describe essentially the same issue, respond 'yes'. Otherwise, respond 'no'."
                        )
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )

            ai_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: AI result for {issue_key}: '{ai_result}'")

            if ai_result.startswith("yes"):
                print(f"INFO: Found an existing ticket (AI says 'yes') -> {issue_key}")
                return issue_key
            elif ai_result.startswith("no"):
                print(f"DEBUG: AI says 'no' => Not considering {issue_key} a duplicate.")
                continue
            else:
                print(
                    f"WARNING: AI gave ambiguous response '{ai_result}'. "
                    f"Continuing with next candidate."
                )
                continue

        except Exception as e:
            print(f"WARNING: IA call failed for {issue_key}: {e}")
            # fallback local
            # si summary o desc sim >= 0.8 => consideramos duplicado
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                print(
                    f"INFO: local fallback => summary_sim={summary_sim:.2f}, "
                    f"desc_sim={desc_sim:.2f} => Marking {issue_key} as duplicate."
                )
                return issue_key
            else:
                print(f"DEBUG: local similarity not high enough for fallback => ignoring {issue_key}.")
                continue

    print("DEBUG: No duplicate ticket found after checking summary & description with local + IA approach.")
    return None

def check_existing_tickets_local_and_ia(
    jira,
    project_key,
    summary,
    description,
    issue_type
):
    """
    Combina comparación local y IA de manera escalonada para evitar duplicados:
      1) Similitud local < 0.3 => descarta como no duplicado, sin IA.
      2) Similitud local >= 0.9 => lo considera duplicado sin preguntar a la IA.
      3) 0.3 <= similitud local < 0.9 => IA decide con 'yes'/'no'.

    Más información de debug para comprender qué camino está tomando.
    """

    print("DEBUG: Checking for existing tickets (local + IA approach)...")

    # Umbrales de similitud local
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9

    # 1) Sanitiza el summary para la query JQL
    sanitized_summary = sanitize_summary(summary)
    print(f"DEBUG: sanitized_summary='{sanitized_summary}'")

    # 2) Preparamos el query: states y issue_type
    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)

    jql_query = (
        f'project = "{project_key}" '
        f'AND issuetype = "{issue_type}" '
        f'AND summary ~ "{sanitized_summary}" '
        f'AND status IN ({states_str})'
    )
    print(f"DEBUG: JQL -> {jql_query}")

    # 3) Ejecutamos la búsqueda en Jira
    try:
        issues = jira.search_issues(jql_query)
        print(f"DEBUG: Found {len(issues)} candidate issue(s).")
    except Exception as e:
        print(f"ERROR: Failed to execute JQL query: {e}")
        return None

    # 4) Para cada candidate issue, analizamos:
    for issue in issues:
        existing_description = issue.fields.description or ""
        issue_key = issue.key
        print(f"DEBUG: Analyzing Issue {issue_key} with local similarity check...")

        # 4A) Calculamos similitud local
        local_similarity = calculate_similarity(description, existing_description)
        print(f"DEBUG: local_similarity with {issue_key} = {local_similarity:.2f}")

        # CASO A: Similitud muy baja -> descartamos
        if local_similarity < LOCAL_SIM_LOW:
            print(f"DEBUG: local_similarity < {LOCAL_SIM_LOW:.2f}, ignoring {issue_key} as candidate.")
            continue

        # CASO B: Similitud muy alta -> duplicado inmediato
        if local_similarity >= LOCAL_SIM_HIGH:
            print(
                f"INFO: Found existing ticket {issue_key} with local_similarity {local_similarity:.2f} >= {LOCAL_SIM_HIGH:.2f}. " 
                "Marking as duplicate without IA."
            )
            return issue_key

        # CASO C: Rango intermedio => preguntar a la IA
        print(
            f"DEBUG: local_similarity in [{LOCAL_SIM_LOW},{LOCAL_SIM_HIGH}). "
            f"Using IA to confirm duplication for {issue_key}..."
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant specialized in analyzing text similarity. "
                            "You must respond only with 'yes' or 'no' to indicate whether the two descriptions match "
                            "in meaning or not."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Compare these two descriptions in terms of meaning, ignoring language differences. "
                            "If they describe the same or very similar issue, respond with 'yes'. Otherwise, respond with 'no'.\n\n"
                            f"Existing description:\n{existing_description}\n\n"
                            f"New description:\n{description}"
                        )
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )

            ai_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: AI comparison result for {issue_key}: '{ai_result}'")

            if ai_result.startswith("yes"):
                print(f"INFO: Found an existing ticket (AI says match) -> {issue_key}")
                return issue_key
            elif ai_result.startswith("no"):
                print(
                    f"DEBUG: AI says 'no' => Not considering {issue_key} a duplicate. " 
                    "Continuing with next candidate."
                )
                continue
            else:
                # Si la IA no empieza con 'yes' o 'no', lo consideramos ambiguo
                print(f"WARNING: AI gave ambiguous response '{ai_result}' => ignoring and continuing.")
                continue

        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with AI: {e}")
            # CASO D: Si la IA falla, podemos hacer un fallback local más indulgente
            if local_similarity >= 0.8:
                print(f"INFO: Fallback local -> similarity {local_similarity:.2f} => Marking {issue_key} as duplicate.")
                return issue_key
            else:
                print(f"DEBUG: local_similarity={local_similarity:.2f} not high enough for fallback => ignoring {issue_key}.")
                continue

    # 5) Si no encontramos duplicado en ninguno de los issues:
    print("DEBUG: No duplicate ticket found after local+IA approach.")
    return None

def check_existing_tickets_ia_only(jira, project_key, summary, description, issue_type):
    """
    Verifica si existe un ticket con un resumen parecido
    y luego llama a la IA para comparar descripciones en detalle.
    Retorna la key del primer ticket que la IA considere duplicado
    o None si no hay coincidencias.
    """
    print("DEBUG: Checking for existing tickets with IA as primary comparator...")

    sanitized_summary = sanitize_summary(summary)

    # Ajusta tus estados
    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)

    # Filtra por issuetype
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

    # Para cada issue, comparamos con la IA
    for issue in issues:
        existing_description = issue.fields.description or ""
        print(f"DEBUG: Using IA to compare with Issue {issue.key}...")

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant specialized in analyzing text similarity. "
                            "You must respond only with 'yes' or 'no' to indicate whether the two descriptions match "
                            "in meaning or not."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Compare these two descriptions in terms of meaning, ignoring language differences. "
                            "If they describe the same or very similar issue, respond with 'yes'. Otherwise, respond with 'no'.\n\n"
                            f"Existing ticket description:\n{existing_description}\n\n"
                            f"New ticket description:\n{description}"
                        )
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            ai_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: AI comparison result for {issue.key}: '{ai_result}'")

            if ai_result.startswith("yes"):
                print(f"INFO: Found an existing ticket (IA indicates match) -> {issue.key}")
                return issue.key
            elif ai_result.startswith("no"):
                print(f"DEBUG: AI says 'no' -> continuing with next issue candidate.")
                continue
            else:
                print(f"WARNING: AI gave ambiguous response '{ai_result}' -> continuing with next candidate.")
                # No concluimos duplicado si la IA es ambigua
                continue

        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with AI: {e}")
            # Si la IA falla, podrías hacer un fallback local o simplemente ignorar
            pass

    print("DEBUG: No duplicate ticket found after IA comparisons.")
    return None

def check_existing_tickets(jira, project_key, summary, description, issue_type):
    """
    Verifica si existe un ticket con un resumen o descripción similar en Jira,
    filtrando por el 'issue_type' y estados relevantes.
    1) Sanitiza el resumen para la búsqueda JQL.
    2) Busca tickets en múltiples estados (por defecto: "To Do", "In Progress", "Open", "Reopened").
    3) Aplica comparación local (SequenceMatcher) con un umbral ajustable.
    4) Si no supera el umbral local, usa IA para confirmar similitud.
    5) Retorna la key del primer ticket que considera duplicado o None si no hay coincidencias.
    """
    print("DEBUG: Checking for existing tickets...")
    LOCAL_SIMILARITY_THRESHOLD = 0.75
    LOCAL_FALLBACK_THRESHOLD = 0.70

    # 1. Sanitizar el resumen para JQL
    sanitized_summary = sanitize_summary(summary)

    # 2. Ajusta los estados que quieras incluir
    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)

    # Filtrar por issuetype (p.ej., "Error" o "Tarea"), dependiendo de log_type
    # si log_type es 'failure' => issue_type = "Error", si es 'success' => "Tarea"
    # (Ya lo manejas en generate_prompt, pero aquí nos aseguramos de filtrar la búsqueda)
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

        # Comparación local
        similarity = calculate_similarity(description, existing_description)
        if similarity >= LOCAL_SIMILARITY_THRESHOLD:
            print(f"INFO: Found an existing ticket (local similarity {similarity:.2f}) -> {issue.key}")
            return issue.key

        # Si no supera el umbral local, usar IA
        print(f"DEBUG: Using IA to compare with Issue {issue.key} if local similarity < {LOCAL_SIMILARITY_THRESHOLD:.2f}")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant specialized in analyzing text similarity. "
                            "You must respond only with 'yes' or 'no' to indicate whether the two descriptions match."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            "Compare these two descriptions in terms of meaning, ignoring language differences. "
                            "If they describe the same or very similar issue, respond with 'yes'. Otherwise, respond with 'no'.\n\n"
                            f"Existing description:\n{existing_description}\n\n"
                            f"New description:\n{description}"
                        )
                    }
                ],
                max_tokens=200,
                temperature=0.3
            )
            ai_result = response.choices[0].message.content.strip().lower()
            print(f"DEBUG: AI comparison result for {issue.key}: '{ai_result}'")

            if ai_result.startswith("yes"):
                print(f"INFO: Found an existing ticket (AI indicates match) -> {issue.key}")
                return issue.key
            elif ai_result.startswith("no"):
                print(f"DEBUG: AI says 'no' -> continuing with next issue candidate.")
                continue
            else:
                print(f"WARNING: AI gave ambiguous response '{ai_result}'")
                if similarity >= LOCAL_FALLBACK_THRESHOLD:
                    print(f"WARNING: Using local fallback, similarity {similarity:.2f} -> Marking {issue.key} as duplicate.")
                    return issue.key

        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with AI: {e}")
            if similarity >= LOCAL_SIMILARITY_THRESHOLD:
                print(f"INFO: Found an existing ticket (fallback local {similarity:.2f}) -> {issue.key}")
                return issue.key

    print("DEBUG: No duplicate ticket found after checking all candidates.")
    return None

# ============ CREACIÓN DE TICKETS ============

def create_jira_ticket_via_requests(jira_url, jira_user, jira_api_token, project_key, summary, description, issue_type):
    print("DEBUG: Creating Jira ticket via REST API (requests)...")
    url = f"{jira_url}/rest/api/3/issue"
    headers = {
        "Content-Type": "application/json"
    }
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
    """
    Crea un ticket en JIRA usando la librería JIRA.
    """
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
                log_files.extend(
                    os.path.join(log_dir, member.name) for member in tar.getmembers() if member.isfile()
                )
        elif os.path.isfile(file_path):
            log_files.append(file_path)

    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files found in the directory '{log_dir}'.")

    print(f"DEBUG: Found {len(log_files)} log file(s) in total.")
    return log_files

def clean_log_content(content):
    """
    Quita líneas vacías y retorna contenido limpio.
    """
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

# ============ VALIDACIÓN DE TIPO DE INCIDENCIA ============

def validate_issue_type(jira_url, jira_user, jira_api_token, project_key, issue_type):
    """
    Valida si el tipo de incidencia es válido para el proyecto especificado.
    """
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
    """
    Genera un prompt refinado para la IA, con instrucciones claras sobre
    cómo estructurar el ticket de Jira en Markdown y añadiendo emojis.
    Retorna el prompt y el tipo de incidencia (Error / Tarea).
    """
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
            "You are a technical writer creating a concise Jira Cloud ticket from logs. "
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
    """
    Reemplaza de forma iterativa cualquier aparición de '**' por '*'.
    Esto simplifica todos los casos en los que la IA use doble asterisco.
    """
    print("DEBUG: Unifying double asterisks to single...")
    while '**' in description:
        description = description.replace('**', '*')
    return description

def sanitize_title(title):
    """
    Elimina asteriscos (*) y backticks (`) del título,
    incluyendo posibles triples (```).
    Mantiene letras, números, espacios y signos de puntuación básicos.
    Luego recorta espacios al inicio/fin.
    """
    print(f"DEBUG: Sanitizing title '{title}'...")
    title = re.sub(r"[\*`]+", "", title)
    title = title.strip()
    print(f"DEBUG: Title after sanitize -> '{title}'")
    return title

# ============ PROCESO PRINCIPAL ============

def analyze_logs_with_ai(log_dir, log_type, report_language, project_name):
    """
    Analiza los logs en log_dir, genera un prompt para la IA y obtiene
    un resumen y descripción para crear el ticket de Jira.
    """
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

    print("DEBUG: Generating prompt and calling OpenAI...")
    prompt, issue_type = generate_prompt(log_type, report_language)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant generating concise Jira tickets. "
                        "Use short, direct statements and minimal markdown formatting. "
                        "Avoid triple backticks for code unless strictly necessary."
                    )
                },
                {
                    "role": "user",
                    "content": f"{prompt}\n\nLogs:\n{logs_content}"
                }
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

    # 1. Conectar a Jira
    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    # 2. Analizar logs y obtener summary, description, issue_type
    summary, description, issue_type = analyze_logs_with_ai(
        args.log_dir,
        args.log_type,
        args.report_language,
        args.project_name
    )

    if not summary or not description or not issue_type:
        print("ERROR: Log analysis failed or invalid issue type. No ticket will be created.")
        return

    print(f"DEBUG: Proposed summary -> '{summary}'\nDEBUG: Proposed issue_type -> '{issue_type}'")

    # 3. Validar si el issue_type es válido para el proyecto
    try:
        validate_issue_type(args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, issue_type)
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to validate issue type: {e}")
        return

    # 4. Revisar si existe un ticket similar (filtrando por summary e issuetype)
    print("DEBUG: Checking for existing tickets...")
    #existing_ticket_key = check_existing_tickets(jira, args.jira_project_key, summary, description, issue_type)
    #existing_ticket_key = check_existing_tickets_local_and_ia(jira, args.jira_project_key, summary, description, issue_type)
    existing_ticket_key = check_existing_tickets_local_and_ia_summary_desc(jira, args.jira_project_key, summary, description, issue_type)

    if existing_ticket_key:
        print(f"INFO: Ticket already exists: {existing_ticket_key}. Skipping creation.")
        return
    
    # 5. Crear ticket en Jira
    print("DEBUG: Creating ticket in Jira...")
    ticket_key = create_jira_ticket(
        jira,
        args.jira_project_key,
        summary,
        description,
        issue_type
    )

    if ticket_key:
        print(f"JIRA Ticket Created using JIRA library: {ticket_key}")
    else:
        print("Falling back to creating ticket via API...")
        ticket_key = create_jira_ticket_via_requests(
            args.jira_url,
            jira_user_email,
            jira_api_token,
            args.jira_project_key,
            summary,
            description,
            issue_type
        )
        if ticket_key:
            print(f"JIRA Ticket Created via API: {ticket_key}")
        else:
            print("Failed to create JIRA ticket.")

    print("DEBUG: Process finished.")

if __name__ == "__main__":
    main()
