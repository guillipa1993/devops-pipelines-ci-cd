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
OPENAI_MODEL = "gpt-4o" 
MAX_CHAR_PER_REQUEST = 20000
BANDIT_JSON_NAME = "bandit-output.json"
MAX_FILE_SIZE_MB = 2.0
ALLOWED_EXTENSIONS = (".log", ".sarif")

# ===================== CONFIGURACIÓN OPENAI =====================
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set.")
    sys.exit(1)

client = OpenAI(api_key=api_key)

# ===================== CONEXIÓN A JIRA =====================
def connect_to_jira(jira_url, jira_user, jira_api_token):
    options = {'server': jira_url}
    jira = JIRA(options, basic_auth=(jira_user, jira_api_token))
    print("Conexión establecida con Jira.")
    return jira

# ===================== FUNCIONES DE SANITIZACIÓN =====================
def sanitize_summary(summary):
    summary = summary.replace("\n", " ").replace("\r", " ")
    sanitized = "".join(c for c in summary if c.isalnum() or c.isspace() or c in "-_:,./()[]{}")
    if len(sanitized) > 255:
        sanitized = sanitized[:255]
    return sanitized.strip()

def preprocess_text(text: str) -> str:
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    return text_no_punct.strip().lower()

def calculate_similarity(text1: str, text2: str) -> float:
    ratio = SequenceMatcher(None, preprocess_text(text1), preprocess_text(text2)).ratio()
    return ratio

# ===================== CONVERSIÓN A WIKI (ADF -> Jira) =====================
def convert_adf_to_wiki(adf) -> str:
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
    blocks = re.split(r"\n\s*-\s+", ai_text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

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
        recommendations.append({"summary": full_summary, "description": description_text})

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
    return any(kw in combined for kw in skip_keywords)

# ===================== FORMAT TICKET CONTENT =====================
def format_ticket_content(project_name: str, rec_summary: str, rec_description: str, ticket_category: str) -> tuple:
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

        if ai_output.startswith("```"):
            lines = ai_output.splitlines()
            if lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            ai_output = "\n".join(lines).strip()

        try:
            ticket_json = json.loads(ai_output)
        except json.JSONDecodeError:
            cleaned = re.sub(r'```.*?```', '', ai_output, flags=re.DOTALL)
            last_brace = cleaned.rfind("}")
            if last_brace != -1:
                cleaned = cleaned[: last_brace+1]
            try:
                ticket_json = json.loads(cleaned)
            except:
                fallback_summary = sanitize_summary(rec_summary)
                fallback_summary = f"{icon} {fallback_summary}"
                fallback_desc = f"Fallback description:\n\n{rec_description}"
                return fallback_summary, fallback_desc

        final_title = ticket_json.get("title", "")
        adf_description = ticket_json.get("description", {})

        if not any(ic in final_title for ic in (IMPROVEMENT_ICONS + ERROR_ICONS)):
            final_title = f"{icon} {final_title}"

        if len(final_title) > 255:
            final_title = final_title[:255]

        wiki_text = convert_adf_to_wiki(adf_description)
        return final_title, wiki_text

    except:
        fallback_summary = sanitize_summary(rec_summary)
        fallback_summary = f"{icon} {fallback_summary}"
        wiki_text = f"Fallback description:\n\n{rec_description}"
        return fallback_summary, wiki_text

# ===================== BÚSQUEDA DE TICKETS =====================
def check_existing_tickets_local_and_ia_summary_desc(jira, project_key, new_summary, new_description, issue_type):
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9
    sanitized_sum = sanitize_summary(new_summary)

    jql_states = ['"To Do"', '"In Progress"', '"Open"', '"Reopened"']
    states_str = ", ".join(jql_states)
    jql_issue_type = "Task" if issue_type.lower() == "tarea" else issue_type
    jql_query = f'project = "{project_key}" AND issuetype = "{jql_issue_type}" AND status IN ({states_str})'
    issues = jira.search_issues(jql_query)

    for issue in issues:
        existing_summary = issue.fields.summary or ""
        existing_description = issue.fields.description or ""
        summary_sim = calculate_similarity(new_summary, existing_summary)
        desc_sim = calculate_similarity(new_description, existing_description)

        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            continue
        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            return issue.key

        # Chequeo con IA
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
            if ai_result.startswith("yes"):
                return issue.key
        except:
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                return issue.key
    return None

def check_discarded_tickets_local_and_ia_summary_desc(jira, project_key, new_summary, new_description, issue_type):
    LOCAL_SIM_LOW = 0.2
    LOCAL_SIM_HIGH = 0.85

    jql_issue_type = "Task" if issue_type.lower() == "tarea" else issue_type
    jql_query = (
        f'project = "{project_key}" AND issuetype = "{jql_issue_type}" '
        f'AND status IN ("DESCARTADO", "Rejected", "Closed", "Done")'
    )
    issues = jira.search_issues(jql_query, maxResults=1000)

    for issue in issues:
        existing_summary = issue.fields.summary or ""
        existing_description = issue.fields.description or ""
        summary_sim = calculate_similarity(new_summary, existing_summary)
        desc_sim = calculate_similarity(new_description, existing_description)

        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            continue

        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            return issue.key

        # Chequeo con IA
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant specialized in analyzing text similarity. Respond only 'yes' or 'no'."},
                    {"role": "user", "content": (
                        "We have two issues:\n\n"
                        f"Existing (discarded) issue:\nSummary: {existing_summary}\nDescription: {existing_description}\n\n"
                        f"New issue:\nSummary: {new_summary}\nDescription: {new_description}\n\n"
                        "Do they represent essentially the same issue? Respond 'yes' or 'no'."
                    )}
                ],
                max_tokens=200,
                temperature=0.3
            )
            ai_result = response.choices[0].message.content.strip().lower()
            if ai_result.startswith("yes"):
                return issue.key
        except:
            if summary_sim >= 0.75 or desc_sim >= 0.75:
                return issue.key
    return None

def check_finalized_tickets_local_and_ia_summary_desc(jira, project_key, new_summary, new_description, issue_type):
    LOCAL_SIM_LOW = 0.3
    LOCAL_SIM_HIGH = 0.9
    jql_issue_type = "Task" if issue_type.lower() == "tarea" else issue_type
    jql_query = (
        f'project = "{project_key}" AND issuetype = "{jql_issue_type}" '
        f'AND statusCategory = Done'
    )
    matched_keys = []
    issues = jira.search_issues(jql_query, maxResults=1000)

    for issue in issues:
        existing_summary = issue.fields.summary or ""
        existing_description = issue.fields.description or ""
        summary_sim = calculate_similarity(new_summary, existing_summary)
        desc_sim = calculate_similarity(new_description, existing_description)

        if summary_sim < LOCAL_SIM_LOW and desc_sim < LOCAL_SIM_LOW:
            continue

        if summary_sim >= LOCAL_SIM_HIGH and desc_sim >= LOCAL_SIM_HIGH:
            matched_keys.append(issue.key)
            continue

        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an assistant specialized in analyzing text similarity. Respond only 'yes' or 'no'."},
                    {"role": "user", "content": (
                        "We have two issues:\n\n"
                        f"Existing (closed) issue:\nSummary: {existing_summary}\nDescription: {existing_description}\n\n"
                        f"New issue:\nSummary: {new_summary}\nDescription: {new_description}\n\n"
                        "Do they represent essentially the same issue? Respond 'yes' or 'no'."
                    )}
                ],
                max_tokens=200,
                temperature=0.3
            )
            ai_result = response.choices[0].message.content.strip().lower()
            if ai_result.startswith("yes"):
                matched_keys.append(issue.key)
        except:
            if summary_sim >= 0.8 or desc_sim >= 0.8:
                matched_keys.append(issue.key)

    return matched_keys

# ===================== CREACIÓN DE TICKETS =====================
def create_jira_ticket(jira, project_key, summary, description, issue_type):
    summary = sanitize_summary(summary)
    if not description.strip():
        return None
    try:
        issue_dict = {
            "project": {"key": project_key},
            "summary": summary,
            "description": description,
            "issuetype": {"name": issue_type},
        }
        issue = jira.create_issue(fields=issue_dict)
        return issue.key
    except:
        return None

def create_jira_ticket_via_requests(jira_url, jira_user, jira_api_token, project_key, summary, description, issue_type):
    summary = sanitize_summary(summary)
    if not description.strip():
        return None

    if isinstance(description, str):
        fallback_adf = {
            "version": 1,
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": description.replace('\n', ' ').replace('\r', ' ')}
                    ]
                }
            ]
        }
        adf_description = fallback_adf
    elif isinstance(description, dict):
        adf_description = description
    else:
        fallback_adf = {
            "version": 1,
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": str(description)}
                    ]
                }
            ]
        }
        adf_description = fallback_adf

    payload = {
        "fields": {
            "project": {"key": project_key},
            "summary": summary,
            "description": adf_description,
            "issuetype": {"name": issue_type}
        }
    }

    url = f"{jira_url}/rest/api/3/issue"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)

    response = requests.post(url, json=payload, headers=headers, auth=auth)
    if response.status_code == 201:
        return response.json().get("key")
    else:
        return None

# ===================== VALIDACIÓN DE LOGS =====================
def validate_logs_directory(log_dir: str) -> list:
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"ERROR: The logs directory '{log_dir}' does not exist.")

    log_files = []
    for file in os.listdir(log_dir):
        file_path = os.path.join(log_dir, file)
        if file.lower() == BANDIT_JSON_NAME.lower():
            continue
        mb_size = os.path.getsize(file_path) / 1024.0 / 1024.0
        if mb_size > MAX_FILE_SIZE_MB:
            continue
        _, ext = os.path.splitext(file.lower())
        if ext not in ALLOWED_EXTENSIONS:
            continue
        if os.path.isfile(file_path):
            log_files.append(file_path)

    if not log_files:
        raise FileNotFoundError(f"ERROR: No valid files found in '{log_dir}'.")
    return log_files

def unify_double_to_single_asterisks(description: str) -> str:
    while '**' in description:
        description = description.replace('**', '*')
    return description

def generate_prompt(log_type: str, language: str) -> tuple:
    if log_type == "failure":
        details = (
            "You are a technical writer creating a concise Jira Cloud ticket from logs. "
            "Keep it short, minimal Markdown. "
            "Use headings like: *Summary*, *Root Cause Analysis*, *Proposed Solutions*, etc. "
            "Use minimal triple backticks for code. "
            f"Write in {language}. Avoid enumerations like '1., a., i.'."
        )
        issue_type = "Error"
    else:
        details = (
            "You are a code reviewer specialized in Python. "
            "Below are logs from a successful build. Produce improvements with format: "
            "- Title (bold)\n- Summary\n- Description. "
            f"Use emojis for variety. Write in {language} with concise language."
        )
        issue_type = "Tarea"
    return details, issue_type

def clean_log_content(content: str) -> str:
    lines = content.splitlines()
    return "\n".join([line for line in lines if line.strip()])

def validate_issue_type(jira_url, jira_user, jira_api_token, project_key, issue_type):
    url = f"{jira_url}/rest/api/3/issue/createmeta?projectKeys={project_key}"
    headers = {"Content-Type": "application/json"}
    auth = (jira_user, jira_api_token)
    response = requests.get(url, headers=headers, auth=auth)
    if response.status_code == 200:
        valid_types = [it["name"] for it in response.json()["projects"][0]["issuetypes"]]
        if issue_type not in valid_types:
            raise ValueError(f"Invalid issue type: '{issue_type}'. Valid types: {valid_types}")
    else:
        raise Exception(f"Failed to fetch issue types: {response.status_code} - {response.text}")

# ===================== MÉTODOS DE ANÁLISIS =====================
def analyze_logs_for_recommendations(log_dir: str, report_language: str, project_name: str) -> list:
    log_files = validate_logs_directory(log_dir)
    combined_text = []
    max_lines = 300
    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:max_lines]
                combined_text.extend(lines)
        except UnicodeDecodeError:
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        return []

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)
    prompt_base, _ = generate_prompt("success", report_language)

    all_recommendations = []
    for chunk in text_chunks:
        prompt = f"{prompt_base}\n\nLogs:\n{chunk}"
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
            recs = parse_recommendations(ai_text)
            all_recommendations.extend(recs)
        except:
            continue
    return all_recommendations

def analyze_logs_with_ai(log_dir: str, log_type: str, report_language: str, project_name: str, branch_name: str = None) -> tuple:
    log_files = validate_logs_directory(log_dir)
    combined_text = []
    max_lines = 300
    error_lines = []

    for file in log_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()[:max_lines]
                combined_text.extend(lines)
                for ln in lines:
                    if any(keyword in ln for keyword in ("ERROR", "Exception", "Traceback")):
                        error_lines.append(ln.strip())
        except UnicodeDecodeError:
            continue

    joined_text = "\n".join(combined_text).strip()
    if not joined_text:
        return None, None, None

    text_chunks = chunk_content_if_needed(joined_text, MAX_CHAR_PER_REQUEST)
    prompt_base, issue_type = generate_prompt(log_type, report_language)

    error_context = ""
    if error_lines:
        few_error_lines = error_lines[:5]
        error_context = "\n\nHere are some specific error lines found:\n" + "\n".join(f"- {l}" for l in few_error_lines)

    chunk = text_chunks[0]
    final_prompt = f"{prompt_base}\n\nLogs:\n{chunk}{error_context}"

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant generating concise Jira tickets. "
                        "Use short statements, some emojis, minimal markdown. "
                        "Make sure the title references the most relevant error."
                    )
                },
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=600,
            temperature=0.4
        )
        summary = response.choices[0].message.content.strip()
        lines = summary.splitlines()

        first_line = lines[0].strip() if lines else "No Title"
        match = re.match(r"(?i)^(?:title|summary)\s*:\s*(.*)$", first_line)
        if match:
            extracted_title = match.group(1).strip()
            lines = lines[1:]
        else:
            extracted_title = first_line

        cleaned_title_line = sanitize_title(extracted_title)
        icon = choose_error_icon()
        if branch_name:
            cleaned_title_line += f" [branch: {branch_name}]"

        summary_title = f"{project_name} {icon} {cleaned_title_line}"
        remaining_desc = "\n".join(lines).strip()
        if not remaining_desc:
            remaining_desc = summary

        description_plain = unify_double_to_single_asterisks(remaining_desc.replace("\t", " "))
        return summary_title, description_plain, issue_type

    except:
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
    parser.add_argument("--branch", required=False, default="", help="Nombre de la rama actual")

    args = parser.parse_args()

    jira_api_token = os.getenv("JIRA_API_TOKEN")
    jira_user_email = os.getenv("JIRA_USER_EMAIL")
    if not jira_api_token or not jira_user_email:
        print("ERROR: Missing env vars JIRA_API_TOKEN or JIRA_USER_EMAIL.")
        sys.exit(1)

    jira = connect_to_jira(args.jira_url, jira_user_email, jira_api_token)

    if args.log_type == "failure":
        summary, description, issue_type = analyze_logs_with_ai(
            args.log_dir, args.log_type, args.report_language,
            args.project_name, branch_name=args.branch
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
            print(f"INFO: Ticket {dup_key} already exists (open). Skipping.")
            return

        final_dup_keys = check_finalized_tickets_local_and_ia_summary_desc(
            jira, args.jira_project_key, summary, description, issue_type
        )

        print("Creando ticket de tipo 'Error' en Jira...")
        ticket_key = create_jira_ticket(jira, args.jira_project_key, summary, description, issue_type)
        if ticket_key:
            print(f"Ticket creado: {ticket_key}")
            if final_dup_keys:
                duplicates_str = ", ".join(final_dup_keys)
                comment_body = get_repeated_incident_comment(duplicates_str, args.report_language)
                try:
                    jira.add_comment(ticket_key, comment_body)
                except:
                    pass
                for old_key in final_dup_keys:
                    try:
                        jira.create_issue_link(
                            type="Relates",
                            inwardIssue=ticket_key,
                            outwardIssue=old_key
                        )
                    except:
                        pass
        else:
            print("No se pudo crear el ticket vía librería. Intentando REST API...")
            fallback_key = create_jira_ticket_via_requests(
                args.jira_url, jira_user_email, jira_api_token,
                args.jira_project_key, summary, description, issue_type
            )
            if fallback_key:
                print(f"Ticket creado via REST: {fallback_key}")
                if final_dup_keys:
                    duplicates_str = ", ".join(final_dup_keys)
                    comment_body = get_repeated_incident_comment(duplicates_str, args.report_language)
                    comment_url = f"{args.jira_url}/rest/api/2/issue/{fallback_key}/comment"
                    comment_data = {"body": comment_body}
                    requests.post(comment_url, json=comment_data, auth=(jira_user_email, jira_api_token))
                    for old_key in final_dup_keys:
                        link_url = f"{args.jira_url}/rest/api/2/issueLink"
                        link_payload = {
                            "type": {"name": "Relates"},
                            "inwardIssue": {"key": fallback_key},
                            "outwardIssue": {"key": old_key}
                        }
                        requests.post(link_url, json=link_payload, auth=(jira_user_email, jira_api_token))
            else:
                print("ERROR: Creación de ticket fallida.")

    else:
        recommendations = analyze_logs_for_recommendations(
            args.log_dir, args.report_language, args.project_name
        )
        if not recommendations:
            print("INFO: No hay recomendaciones generadas por la IA.")
            return

        issue_type = "Tarea"
        for i, rec in enumerate(recommendations, start=1):
            r_summary = rec["summary"]
            r_desc = rec["description"]
            if not r_desc.strip():
                continue
            if should_skip_recommendation(r_summary, r_desc):
                continue

            dup_key = check_existing_tickets_local_and_ia_summary_desc(
                jira, args.jira_project_key, r_summary, r_desc, issue_type
            )
            if dup_key:
                continue

            discard_key = check_discarded_tickets_local_and_ia_summary_desc(
                jira, args.jira_project_key, r_summary, r_desc, issue_type
            )
            if discard_key:
                continue

            final_title, wiki_desc = format_ticket_content(
                args.project_name, r_summary, r_desc, "Improvement"
            )
            if not wiki_desc.strip():
                continue

            new_key = create_jira_ticket(jira, args.jira_project_key, final_title, wiki_desc, issue_type)
            if new_key:
                print(f"Recomendación #{i} => Creado ticket: {new_key}")
            else:
                print(f"Intentando creación fallback para la recomendación #{i}...")
                fallback_key = create_jira_ticket_via_requests(
                    args.jira_url, jira_user_email, jira_api_token,
                    args.jira_project_key, final_title, wiki_desc, issue_type
                )
                if fallback_key:
                    print(f"Recomendación #{i} => Creado ticket via REST: {fallback_key}")

if __name__ == "__main__":
    main()
