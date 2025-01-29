import os
import argparse
from jira import JIRA
import tarfile
import requests
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
    Compara también el contenido del ticket usando IA para evitar duplicados.
    Solo busca tickets en estado "To Do" o "In Progress".
    """
    # Buscar tickets en el estado especificado con un resumen similar
    jql_query = f'project = "{project_key}" AND summary ~ "{summary}" AND status IN ("To Do", "In Progress")'
    issues = jira.search_issues(jql_query)

    for issue in issues:
        existing_description = issue.fields.description or ""

        # Usar IA para determinar similitud entre descripciones
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an assistant specialized in analyzing text similarity."},
                    {
                        "role": "user",
                        "content": f"Does the following description match this one?\n\n"
                                   f"Existing description:\n{existing_description}\n\n"
                                   f"New description:\n{description}"
                    }
                ],
                max_tokens=5000,
                temperature=0.4
            )
            ai_result = response.choices[0].message.content.strip().lower()
            if "yes" in ai_result or "match" in ai_result:
                print(f"INFO: Found an existing ticket with similar content: {issue.key}")
                return issue.key

        except Exception as e:
            print(f"WARNING: Failed to analyze similarity with AI: {e}")
            # Fallback to simple string comparison
            if description.strip().lower() in existing_description.lower():
                print(f"INFO: Found an existing ticket with similar description (fallback): {issue.key}")
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
    if log_type == "failure":
        details = (
            "You are an expert technical writer. Generate a Jira Cloud ticket based on the provided logs. "
            "The ticket should be structured as follows:\n\n"
            "1. **Summary**: Provide a concise summary of the detected issue, clearly highlighting the problem. Use relevant emojis like `🔍`.\n"
            "2. **Root Cause Analysis**: Analyze the logs to identify the primary cause of the issue. Include critical error messages or log snippets in code blocks (``` ```).\n"
            "3. **Proposed Solutions**: List actionable steps to resolve the issue. Use numbered lists and provide code examples in blocks. Emphasize key actions with emojis like 💡 or 🔧.\n"
            "4. **Preventive Measures**: Suggest strategies to avoid similar issues in the future. Use bullet points (`-`) and actionable advice.\n"
            "5. **Critical Issues Highlight**: Explain the risks of not addressing the issue. Use impactful emojis like 💣 or ⚠️ to stress criticality.\n"
            "6. **Impact Analysis**: Describe the implications of the issue on development workflows, users, or systems.\n\n"
        )
        issue_type = "Error"
    else:
        details = (
            "You are an expert technical writer. Generate a Jira Cloud ticket based on the provided logs. "
            "The ticket should be structured as follows:\n\n"
            "1. **Summary**: Provide a concise summary of the successful state of the build. Use relevant emojis like `✅`.\n"
            "2. **Success Confirmation**: Outline the tasks completed successfully and the technical milestones achieved.\n"
            "3. **Proposed Optimizations**: Recommend potential improvements or adjustments to enhance performance or scalability. Use bullet points (`-`) and provide examples in code blocks.\n"
            "4. **Scalability Recommendations**: Suggest ideas to make the system ready for higher loads, updates, or new functionalities.\n"
            "5. **Solution Benefits**: Highlight the positive outcomes of the successful build for the project, users, or system stability.\n\n"
        )
        issue_type = "Task"
    prompt = (
        f"{details} Ensure the ticket is clear, professional, and formatted to fit Jira Cloud's requirements. "
        f"The ticket content must be generated in {language}."
    )
    return prompt, issue_type

def analyze_logs_with_ai(log_dir, log_type, report_language, project_name):
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
        summary = response.choices[0].message.content.strip()

        # Generar un título más descriptivo
        summary_title = f"{project_name}: {log_type.capitalize()} Error - {summary.splitlines()[0]}"

        # Formatear la descripción en Markdown para Jira
        description_plain = f"""
# Informe de Análisis de Logs: Error en `{project_name}`

## Resumen del Problema
{summary}

## Logs Analizados
```
{combined_logs}
```

## Soluciones Recomendadas 💡
- [Incluir pasos sugeridos aquí]

## Análisis de Impacto 🚫⚠️
- [Incluir impacto aquí]
"""
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
