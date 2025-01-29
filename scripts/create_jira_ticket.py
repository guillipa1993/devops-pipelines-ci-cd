import os
import argparse
import tarfile
import requests
import re
from jira import JIRA
from openai import OpenAI
from datetime import datetime
from difflib import SequenceMatcher

# Verificar si la clave de API está configurada
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

# Inicializar la API de OpenAI
client = OpenAI(api_key=api_key)

def connect_to_jira(jira_url, jira_user, jira_api_token):
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    return jira

def sanitize_summary(summary):
    """
    Limpia el resumen para eliminar caracteres que puedan causar problemas en el JQL.
    """
    return "".join(c for c in summary if c.isalnum() or c.isspace())

def preprocess_text(text):
    # Quita puntuación y espacios extra
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip().lower()

def calculate_similarity(text1, text2):
    t1 = preprocess_text(text1)
    t2 = preprocess_text(text2)
    return SequenceMatcher(None, t1, t2).ratio()

def check_existing_tickets(jira, project_key, summary, description):
    """
    Verifica si existe un ticket con un resumen o descripción similar en Jira.
    1) Sanitiza el resumen para la búsqueda JQL.
    2) Busca tickets en múltiples estados (puedes ajustar el listado).
    3) Aplica comparación local (SequenceMatcher) con un umbral ajustable.
    4) Si no supera el umbral local, usa IA para confirmar similitud.
    5) Retorna la key del primer ticket que considera duplicado o None si no hay coincidencias.
    """
    
    # Ajusta el umbral de similitud local (SequenceMatcher)
    LOCAL_SIMILARITY_THRESHOLD = 0.75  # Por ejemplo, 0.75
    # Ajusta el umbral "casi alto" para fallback en caso de respuesta IA ambigua
    LOCAL_FALLBACK_THRESHOLD = 0.70
    
    # 1. Sanitizar el resumen para JQL
    sanitized_summary = sanitize_summary(summary)  # Usa tu método sanitize_summary
    
    # 2. Preparar una lista de estados que quieras incluir
    #    (Si prefieres uno en concreto, ajusta la lista)
    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']  
    states_str = ", ".join(jql_states)
    
    # 3. Construir la query JQL
    jql_query = (
        f'project = "{project_key}" AND summary ~ "{sanitized_summary}" '
        f'AND status IN ({states_str})'
    )
    
    try:
        # 4. Ejecutar la búsqueda en Jira
        issues = jira.search_issues(jql_query)
        print(f"DEBUG: Found {len(issues)} candidate issues with JQL: {jql_query}")
    except Exception as e:
        print(f"ERROR: Failed to execute JQL query: {e}")
        return None

    # 5. Revisar cada ticket candidato
    for issue in issues:
        existing_description = issue.fields.description or ""
        
        # 5A. Comparación local
        similarity = calculate_similarity(description, existing_description)
        print(f"DEBUG: Comparing with Issue {issue.key} -> Local similarity: {similarity:.2f}")
        
        if similarity >= LOCAL_SIMILARITY_THRESHOLD:
            # Si supera el umbral local alto, se considera duplicado
            print(f"INFO: Found an existing ticket (local similarity {similarity:.2f}) -> {issue.key}")
            return issue.key
        
        # 5B. Si la similitud local es menor, usar IA para un segundo filtro
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

            # 5C. Interpretar la respuesta de la IA
            if ai_result.startswith("yes"):
                print(f"INFO: Found an existing ticket (AI indicates match) -> {issue.key}")
                return issue.key
            elif ai_result.startswith("no"):
                # Continúa buscando en otros tickets
                continue
            else:
                # Respuesta ambigua -> fallback local
                if similarity >= LOCAL_FALLBACK_THRESHOLD:
                    print(
                        f"WARNING: AI gave ambiguous response '{ai_result}', "
                        f"but local similarity is {similarity:.2f} -> {issue.key}"
                    )
                    return issue.key

        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with AI: {e}")
            # Fallback: si la similitud local es alta (>= LOCAL_SIMILARITY_THRESHOLD)
            # de todos modos considerarlo duplicado
            if similarity >= LOCAL_SIMILARITY_THRESHOLD:
                print(
                    f"INFO: Found an existing ticket (fallback local {similarity:.2f}) -> {issue.key}"
                )
                return issue.key

    # 6. Si no se encontró ningún duplicado, retornar None
    return None

def create_jira_ticket_via_requests(jira_url, jira_user, jira_api_token, project_key, summary, description, issue_type):
    url = f"{jira_url}/rest/api/3/issue"
    headers = {
        "Content-Type": "application/json"
    }
    auth = (jira_user, jira_api_token)
    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,  # Usar texto plano aquí
            "issuetype": {"name": issue_type}
        }
    }
    response = requests.post(url, json=payload, headers=headers, auth=auth)

    if response.status_code == 201:
        print("Ticket created successfully:", response.json())
        return response.json().get("key")
    else:
        print(f"Failed to create ticket: {response.status_code} - {response.text}")
        return None

def create_jira_ticket(jira, project_key, summary, description, issue_type):
    """
    Crea un ticket en JIRA usando la librería JIRA.
    """
    try:
        issue_dict = {
            'project': {'key': project_key},
            'summary': summary,
            'description': description,  # Usar texto plano aquí
            'issuetype': {'name': issue_type}
        }
        issue = jira.create_issue(fields=issue_dict)
        return issue.key
    except Exception as e:
        print(f"ERROR: No se pudo crear el ticket en JIRA: {e}")
        return None

def validate_logs_directory(log_dir):
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")
    log_files = []

    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if file.endswith(".tar.gz"):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=log_dir)
                log_files.extend(
                    os.path.join(log_dir, member.name) for member in tar.getmembers() if member.isfile()
                )
        elif os.path.isfile(file_path):
            log_files.append(file_path)

    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files found in the directory '{log_dir}'.")
    return log_files

def clean_log_content(content):
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

def validate_issue_type(jira_url, jira_user, jira_api_token, project_key, issue_type):
    """
    Valida si el tipo de incidencia es válido para el proyecto especificado.
    """
    url = f"{jira_url}/rest/api/3/issue/createmeta?projectKeys={project_key}"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)
    response = requests.get(url, headers=headers, auth=auth)
    if response.status_code == 200:
        valid_types = [
            issue["name"] for issue in response.json()["projects"][0]["issuetypes"]
        ]
        if issue_type not in valid_types:
            raise ValueError(f"Invalid issue type: '{issue_type}'. Valid types: {valid_types}")
    else:
        raise Exception(f"Failed to fetch issue types: {response.status_code} - {response.text}")

def generate_prompt(log_type, language):
    """
    Genera un prompt refinado para la IA, con instrucciones claras sobre
    cómo estructurar el ticket de Jira en Markdown y añadiendo emojis.
    Retorna el prompt y el tipo de incidencia (Error / Task).
    """

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
        issue_type = "Task"

    # Añadimos un recordatorio de concisión y del idioma deseado
    prompt = (
        f"{details}\n\n"
        f"Be concise, professional, and compatible with Jira Cloud's Markdown. "
        f"Write the ticket in {language}."
    )
    return prompt, issue_type

def unify_double_to_single_asterisks(description):
    """
    Reemplaza de forma iterativa cualquier aparición de '**' por '*'.
    Esto simplifica todos los casos en los que la IA use doble asterisco,
    sin importar el idioma o el contexto.
    """
    # Mientras sigamos encontrando '**', las reducimos a un '*'
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
    # Sustituye cualquier aparición de `*` o `` ` `` (en bloque o individual) por nada
    # Ejemplo: "```markdown" -> "markdown", "***Error***" -> "Error"
    title = re.sub(r"[\*`]+", "", title)

    # Elimina espacios en exceso
    title = title.strip()

    return title

def analyze_logs_with_ai(log_dir, log_type, report_language, project_name):
    """
    Analiza los logs en log_dir, genera un prompt para la IA y obtiene
    un resumen y descripción para crear el ticket de Jira.
    """
    # 1. Validar y cargar archivos de logs
    log_files = validate_logs_directory(log_dir)
    combined_logs = []
    
    # 2. Filtrado básico de logs (Ejemplo: limitar a 300 líneas totales)
    max_lines = 300
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                # Opcional: filtrar solo secciones relevantes
                # lines = [ln for ln in lines if "ERROR" in ln or "Exception" in ln]
                
                # Agregamos hasta 'max_lines' para que no sea muy extenso
                combined_logs.extend(lines[:max_lines])
        except UnicodeDecodeError:
            print(f"WARNING: Could not read file {file} due to encoding issues. Skipping.")
            continue

    # Si quieres separar cada archivo en el prompt, puedes crear secciones
    logs_content = "\n".join(combined_logs)
    if not logs_content.strip():
        print("ERROR: No relevant logs found for analysis.")
        return None, None, None

    # 3. Generar el prompt para la IA
    prompt, issue_type = generate_prompt(log_type, report_language)

    # 4. Llamar a la IA con un rol system que fuerce respuestas breves y concisas
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
            max_tokens=600,  # Límite menor para no generar texto muy largo
            temperature=0.4   # Menor temperatura para ser más directa y predecible
        )
        
        # 5. Procesar la respuesta de la IA
        summary = response.choices[0].message.content.strip()
        
        # 6. Limpiar la primera línea para evitar secuencias no deseadas como ```markdown
        lines = summary.splitlines()
        first_line = lines[0] if lines else "No Title"
        
        # Eliminar triple backticks y la palabra 'markdown'
        cleaned_title_line = (
            first_line
            .replace("```markdown", "")
            .replace("```", "")
            .strip()
        )
        
        # Opcional: quitar doble asteriscos si no los quieres en el título
        cleaned_title_line = sanitize_title(cleaned_title_line)

        # Armar el título final
        label = "Error" if log_type == "failure" else "Success"
        summary_title = f"{project_name}: {label} - {cleaned_title_line}"
        
        # 7. Descripción final (posprocesado opcional para quitar tabulaciones, etc.)
        description_plain = summary
        # Eliminar tabulaciones y espacios repetidos
        # (Si usas HTML en Jira, podrías convertir markdown a HTML, etc.)
        description_plain = description_plain.replace("\t", " ")

        # Reemplazar doble asterisco por uno solo
        description_plain = unify_double_to_single_asterisks(description_plain)

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

    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user_email = os.getenv("JIRA_USER_EMAIL")

    if not all([jira_api_token, jira_user_email]):
        print("ERROR: Missing required environment variables JIRA_API_TOKEN or JIRA_USER_EMAIL.")
        exit(1)

    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    summary, description, issue_type = analyze_logs_with_ai(args.log_dir, args.log_type, args.report_language, args.project_name)

    if not summary or not description or not issue_type:
        print("ERROR: Log analysis failed or invalid issue type. No ticket will be created.")
        return

    try:
        validate_issue_type(args.jira_url, jira_user_email, jira_api_token, args.jira_project_key, issue_type)
    except ValueError as e:
        print(f"ERROR: {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to validate issue type: {e}")
        return

    existing_ticket_key = check_existing_tickets(jira, args.jira_project_key, summary, description)
    if existing_ticket_key:
        print(f"INFO: Ticket already exists: {existing_ticket_key}. Skipping creation.")
        return

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

if __name__ == "__main__":
    main()
