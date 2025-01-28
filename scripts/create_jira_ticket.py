import os
import argparse
from jira import JIRA
from openai import OpenAI
from datetime import datetime

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

def check_existing_tickets(jira, project_key, summary, description):
    """
    Verifica si existe un ticket con un resumen o descripción similar en Jira.
    """
    jql_query = f'project = "{project_key}" AND summary ~ "{summary}"'
    issues = jira.search_issues(jql_query)

    for issue in issues:
        if description.strip() in issue.fields.description:
            print(f"INFO: Found an existing ticket with similar description: {issue.key}")
            return issue.key
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
            "description": {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": description}
                        ]
                    }
                ]
            },
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
            'description': {
                "type": "doc",
                "version": 1,
                "content": [
                    {
                        "type": "paragraph",
                        "content": [
                            {"type": "text", "text": description}
                        ]
                    }
                ]
            },
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
    log_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, f))
    ]
    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files found in the directory '{log_dir}'.")
    return log_files

def clean_log_content(content):
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

def generate_prompt(log_type, language):
    if log_type == "failure":
        details = "Identify issues, recommend fixes, and provide preventive measures."
        issue_type = "Bug"
    else:
        details = "Confirm success, suggest optimizations, and provide scalability recommendations."
        issue_type = "Task"
    prompt = f"Analyze the logs provided and generate a detailed report in {language}. {details}"
    return prompt, issue_type

def analyze_logs_with_ai(log_dir, log_type, report_language):
    log_files = validate_logs_directory(log_dir)
    combined_logs = ""
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                combined_logs += f"\n### {os.path.basename(file)}\n" + clean_log_content(f.read())
        except UnicodeDecodeError:
            print(f"WARNING: Could not read file {file} due to encoding issues. Skipping.")
            continue

    prompt, issue_type = generate_prompt(log_type, report_language)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{prompt}\n{combined_logs}"}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        summary = response["choices"][0]["message"]["content"].strip()
        return summary, issue_type
    except Exception as e:
        print(f"ERROR: Failed to analyze logs with AI: {e}")
        return None, None

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

    summary = f"Log Analysis - {args.log_type}"
    description, issue_type = analyze_logs_with_ai(args.log_dir, args.log_type, args.report_language)

    if not description:
        print("ERROR: Log analysis failed. No ticket will be created.")
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
        "Bug" if args.log_type == "failure" else "Task"
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
            "Bug" if args.log_type == "failure" else "Task"
        )
        if ticket_key:
            print(f"JIRA Ticket Created via API: {ticket_key}")
        else:
            print("Failed to create JIRA ticket.")

if __name__ == "__main__":
    main()
