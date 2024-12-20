import openai
import os
import subprocess
import argparse
from openai import OpenAI
from datetime import datetime

# Verificar si la clave de API está configurada
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

# Inicializar la API de OpenAI
client = OpenAI(api_key=api_key)

def validate_logs_directory(log_dir):
    """
    Valida si el directorio de logs existe y contiene archivos .txt.
    """
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

def clean_log_content(content):
    """
    Elimina líneas vacías y contenido redundante del log.
    """
    lines = content.splitlines()
    cleaned_lines = [line for line in lines if line.strip()]
    return "\n".join(cleaned_lines)

def generate_prompt(log_type, language):
    """
    Genera el prompt según el tipo de log y el idioma.
    """
    if log_type == "failure":
        prompt = (
            "You are an expert log analysis assistant. The provided log type is 'failure'. "
            "Your primary goal is to analyze the logs to identify issues, explain their root causes, and recommend actionable fixes. "
            "Ensure the analysis includes:\n"
            "1. **🔍 Root Cause Analysis:** Identify specific reasons for the failure, with supporting evidence from the logs.\n"
            "2. **🔧 Actionable Fixes:** Provide step-by-step recommendations, including file names, line numbers, and code examples.\n"
            "3. **⚡ Preventive Measures:** Suggest workflow or dependency changes to avoid similar issues.\n"
            "4. **🔥 Critical Issues:** Highlight urgent or blocking issues clearly.\n"
            "5. **🚫 Impact Analysis:** Describe potential consequences of unresolved issues.\n"
            "6. **🎯 Next Steps:** Summarize key actions to resolve issues and improve system reliability.\n"
        )
    else:
        prompt = (
            "You are an expert log analysis assistant. The provided log type is 'success'. "
            "Your primary goal is to confirm the success of the process and suggest optimizations for future scalability. "
            "Ensure the analysis includes:\n"
            "1. **✅ Confirmation of Success:** Clearly state that the process completed successfully.\n"
            "2. **🚀 Opportunities for Optimization:** Suggest areas for performance improvement or simplification.\n"
            "3. **📈 Scalability Recommendations:** Provide advice on extending the success to larger workloads.\n"
            "4. **🌿 Sustainability Suggestions:** Propose best practices to maintain success over time.\n"
            "5. **🎉 Positive Feedback:** Acknowledge outstanding practices or results achieved.\n"
            "6. **✨ Best Practices:** Offer tips to replicate this success.\n"
        )
    if language:
        prompt += f" Generate all response text in {language}."
    return prompt

def create_github_issue(title, body, repo_name):
    """
    Crea un ticket en GitHub utilizando la CLI `gh`.
    """
    try:
        print("DEBUG: Attempting to create a GitHub issue...")
        command = [
            "gh", "issue", "create",
            "--repo", repo_name,
            "--title", title,
            "--body", body
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("GitHub issue created successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to create GitHub issue.")
        print("DEBUG: Command output:", e.stderr)

def summarize_logs_with_openai(log_dir, log_type, language):
    """
    Resume los logs utilizando la API de OpenAI.
    """
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
        role_content = generate_prompt(log_type, language)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": role_content},
                {"role": "user", "content": fragment}
            ],
            max_tokens=2000,
            temperature=0.5
        )
        consolidated_summary += response.choices[0].message.content.strip() + "\n\n"
    return consolidated_summary

def main():
    parser = argparse.ArgumentParser(description="Summarize analysis and create GitHub issue.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the directory with analysis results.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the summary output.")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository (e.g., owner/repo).")
    parser.add_argument("--run-id", type=str, required=True, help="GitHub Actions run ID.")
    parser.add_argument("--log-type", type=str, required=True, choices=["success", "failure"], help="Specify the type of logs to analyze.")
    parser.add_argument("--report-language", type=str, required=False, help="Specify the language for the summary report.")
    parser.add_argument("--create-ticket", action="store_true", help="Flag to create a GitHub issue.")
    args = parser.parse_args()

    summary = summarize_logs_with_openai(args.log_dir, args.log_type, args.report_language)
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(summary)

    if args.create_ticket:
        title = f"{'🔴 Failure' if args.log_type == 'failure' else '🟢 Success'} Report - Project: {args.repo} - Build ID: {args.run_id} - {'Errors Found' if args.log_type == 'failure' else 'All Passed'}"
        create_github_issue(title, summary, args.repo)

if __name__ == "__main__":
    main()
