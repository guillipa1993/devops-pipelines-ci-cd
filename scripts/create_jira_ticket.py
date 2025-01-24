import os
import argparse
from jira import JIRA
from datetime import datetime
import openai

# Verificar si la clave de API está configurada
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

# Inicializar la API de OpenAI
openai.api_key = openai_api_key

# Conectar a JIRA
def connect_to_jira(jira_url, jira_user, jira_api_token):
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    return jira

# Consultar si existe un ticket similar en JIRA
def check_existing_tickets(jira, project_key, summary):
    jql_query = f'project = "{project_key}" AND summary ~ "{summary}"'
    issues = jira.search_issues(jql_query)
    return issues

# Crear un ticket en JIRA
def create_jira_ticket(jira, project_key, summary, description, issue_type):
    """
    Crea un ticket en JIRA.
    """
    try:
        issue_dict = {
            'project': {'key': project_key},
            'summary': summary,
            'description': description,
            'issuetype': {'name': issue_type}
        }
        issue = jira.create_issue(fields=issue_dict)
        return issue.key
    except Exception as e:
        print(f"ERROR: No se pudo crear el ticket en JIRA: {e}")
        return None

# Validar los logs
def validate_logs_directory(log_dir):
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")
    log_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, f)) and f.endswith(".txt")
    ]
    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid .txt files found in the directory '{log_dir}'.")
    return log_files

# Limpiar el contenido de los logs
def clean_log_content(content):
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

# Generar un prompt para OpenAI
def generate_prompt(log_type, language):
    """
    Genera un prompt para analizar los logs con OpenAI.
    """
    base_prompt = "Analyze the logs provided and generate a detailed report."
    if log_type == "failure":
        details = "Identify issues, recommend fixes, and provide preventive measures."
        issue_type = "Bug"
    else:
        details = "Confirm success, suggest optimizations, and provide scalability recommendations."
        issue_type = "Task"
    prompt = f"{base_prompt} {details} Ensure the summary is in {language}."
    return prompt, issue_type

# Resumir los logs con OpenAI
def summarize_logs_with_openai(log_dir, log_type, language):
    log_files = validate_logs_directory(log_dir)
    all_content = ""
    for filename in log_files:
        with open(filename, "r") as f:
            file_content = f.read()
            all_content += f"### 📄 {os.path.basename(filename)}\n{clean_log_content(file_content)}\n\n"

    max_chunk_size = 30000
    content_fragments = [all_content[i:i + max_chunk_size] for i in range(0, len(all_content), max_chunk_size)]
    consolidated_summary = ""
    for idx, fragment in enumerate(content_fragments, 1):
        role_content, issue_type = generate_prompt(log_type, language)
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": role_content},
                {"role": "user", "content": fragment}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        consolidated_summary += response.choices[0].message.content.strip() + "\n\n"
    return consolidated_summary, issue_type

# Flujo principal
def main():
    parser = argparse.ArgumentParser(description="Crear tickets en JIRA desde logs analizados.")
    parser.add_argument("--jira-url", required=True, help="URL de la instancia de JIRA.")
    parser.add_argument("--jira-project-key", required=True, help="Clave del proyecto en JIRA.")
    parser.add_argument("--log-dir", required=True, help="Directorio con los logs.")
    parser.add_argument("--log-type", required=True, choices=["success", "failure"], help="Tipo de log.")
    parser.add_argument("--language", default="English", help="Idioma para el resumen.")
    parser.add_argument("--project-name", required=False, help="Nombre del repositorio en GitHub.")
    parser.add_argument("--run-id", required=False, help="ID de la ejecución en GitHub Actions.")
    parser.add_argument("--report-language", required=False, help="Idioma para el resumen del reporte.")
    parser.add_argument("--repo", required=False, help="Nombre completo del repositorio (owner/repo).")

    args = parser.parse_args()

    # Variables de entorno
    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user_email = os.getenv("JIRA_USER_EMAIL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not all([jira_api_token, jira_user_email, openai_api_key]):
        print("ERROR: Missing required environment variables. Please set them before running the script.")
        exit(1)

    # Conectar a JIRA
    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    # Procesar los logs y generar el resumen
    summary, issue_type = summarize_logs_with_openai(args.log_dir, args.log_type, args.language)

    # Verificar si el ticket ya existe
    existing_tickets = check_existing_tickets(jira, args.jira_project_key, f"Log Analysis - {args.log_type}")
    if existing_tickets:
        print(f"INFO: Found existing tickets: {[ticket.key for ticket in existing_tickets]}. Skipping ticket creation.")
        return

    # Crear un ticket en JIRA si no existe
    ticket_key = create_jira_ticket(jira, args.jira_project_key, f"Log Analysis - {args.log_type}", summary, issue_type)
    print(f"JIRA Ticket Created: {ticket_key}")

if __name__ == "__main__":
    main()

