import os
import argparse
from jira import JIRA
from datetime import datetime
import requests
import openai

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

def analyze_logs_with_ai(log_dir, log_type, report_language):
    """
    Analiza los logs usando OpenAI y devuelve un resumen.
    """
    log_files = [
        os.path.join(log_dir, f)
        for f in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, f))
    ]

    combined_logs = ""
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                combined_logs += f"\n### {file}\n" + f.read()
        except UnicodeDecodeError:
            print(f"WARNING: Could not read file {file} due to encoding issues. Skipping.")
            continue

    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = (
        f"Analyze the following logs and generate a detailed report in {report_language}:\n"
        f"Log Type: {log_type}\n"
        f"Logs:\n{combined_logs}"
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        summary = response["choices"][0]["message"]["content"]
        return summary
    except Exception as e:
        print(f"ERROR: Failed to analyze logs with AI: {e}")
        return None

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
    description = analyze_logs_with_ai(args.log_dir, args.log_type, args.report_language)

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
