import openai
import os
import subprocess
import argparse
from openai import OpenAI

# Verificar si la clave de API está configurada
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

# Inicializar la API de OpenAI
client = OpenAI(api_key=api_key)

def create_github_issue(title, body, build_id):
    """
    Crea un ticket en GitHub utilizando la CLI `gh`.
    """
    try:
        print("DEBUG: Attempting to create a GitHub issue...")
        command = [
            "gh", "issue", "create",
            "--title", f"{title} - Build #{build_id}",
            "--body", body
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("GitHub issue created successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to create GitHub issue.")
        print("DEBUG: Command output:", e.stderr)

def summarize_logs_with_openai(log_dir, build_id, language):
    """
    Lee y resume el contenido de los archivos de análisis utilizando la API de OpenAI.
    """
    print(f"DEBUG: Checking log directory: {log_dir}")
    if not os.path.isdir(log_dir):
        print(f"ERROR: Log directory '{log_dir}' does not exist.")
        return None

    all_content = ""
    print(f"DEBUG: Reading log files in directory: {log_dir}")
    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(log_dir, filename)
            print(f"DEBUG: Reading file: {file_path}")
            try:
                with open(file_path, "r") as f:
                    all_content += f"### 📄 {filename}\n{f.read()}\n\n"
            except Exception as e:
                print(f"ERROR: Failed to read file {file_path}: {e}")

    if not all_content:
        print("ERROR: No valid analysis files found in the directory.")
        return None

    # Dividir contenido si es demasiado grande
    print("DEBUG: Splitting content into fragments for OpenAI API...")
    max_chunk_size = 30000
    content_fragments = [all_content[i:i + max_chunk_size] for i in range(0, len(all_content), max_chunk_size)]

    print(f"DEBUG: Total fragments to process: {len(content_fragments)}")
    consolidated_summary = ""
    for idx, fragment in enumerate(content_fragments, 1):
        try:
            print(f"DEBUG: Processing fragment {idx}/{len(content_fragments)} with OpenAI API...")
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant summarizing log analysis results for a GitHub ticket. "
                        "Analyze the provided logs and generate a detailed, structured summary. Ensure the output includes:"
                        "1. **Clear Overview**: Provide a concise overview of the key findings, emphasizing the most relevant issues or successes. Use emojis to reinforce the context: "
                        "   - 🔥 (Critical failure), ❌ (Failed operation), ✅ (Success), 🎉 (Positive result), ⚡ (Unexpected interruption)."
                        "2. **Root Cause Analysis**: Explain the root causes of any identified issues with evidence or patterns observed in the logs. Highlight these points with: "
                        "   - 🔍 (Investigation), 🛠️ (Attention needed)."
                        "3. **Actionable Recommendations**: Offer actionable steps to address each issue, including specific files, line numbers, or configurations. When possible, provide examples of fixes. Use: "
                        "   - 💡 (Idea), 🔧 (Fix), or 🛠️ (Repair)."
                        "4. **Highlight Critical Issues**: Clearly indicate any blocking or urgent issues requiring immediate attention. Use: "
                        "   - 💣 (Critical failure), 📛 (Major warning), 🚫 (Blocked action)."
                        "5. **Affected Areas**: Specify affected files and line numbers, making it easier for developers to locate issues. Reinforce with: "
                        "   - 🗂️ (Files), 📌 (Locations)."
                        "6. **Tooling and Updates**: Suggest updates to tools, libraries, or code versions to enhance robustness and maintainability. Highlight benefits with: "
                        "   - 📈 (Improvement), 🔒 (Security)."
                        "7. **Optimizations and Refactorings**: Recommend improvements to workflow, scalability, or code quality. Include: "
                        "   - ✨ (Refactoring), 🚀 (Optimization), 🌟 (Best practice)."
                        "8. **Pattern Recognition**: Identify recurring issues or anomalies that suggest systemic problems. Use: "
                        "   - 🔄 (Recurring issue), ⚠️ (Potential risk)."
                        "9. **Success Logs**: For success logs, confirm no hidden warnings or issues and suggest ways to sustain or enhance stability. Use: "
                        "   - 🎯 (Precision), 🌈 (Clean result), ✅ (Confirmation)."
                        "10. **Scalability Opportunities**: Highlight opportunities for scalability, performance enhancement, or security improvements. Include: "
                        "   - 📈 (Growth), 🛡️ (Protection)."
                        "11. **Motivational Notes**: Add a motivational comment or thank contributors for their effort. Use: "
                        "   - 🙌 (Appreciation), 👏 (Acknowledgment)."
                        "12. **Structured Report**: Organize the output into clear sections: Findings, Root Causes, Recommendations, and Next Steps. Ensure it is visually appealing and professional, with appropriate emojis for each section."
                        "13. **Log State Context**: Clearly state whether the log indicates success (✅) or failure (❌) at the beginning of the analysis."
                        "14. **Ongoing Monitoring**: Suggest tools or strategies for continuous monitoring if systemic improvements are identified. Use: "
                        "   - ⏱️ (Time tracking), 🔍 (Monitoring)."
                        "15. **Encouragement for Collaboration**: Foster collaboration by emphasizing the shared goal of maintaining a high-quality project. Highlight teamwork with: "
                        "   - 💪 (Teamwork), 🤝 (Collaboration)."
                        f"Generate the report in {language}." if language else "Generate the report in English."
                    )},
                    {"role": "user", "content": fragment}
                ],
                        max_tokens=2000,
                        temperature=0.5
            )
            print(f"DEBUG: OpenAI response received for fragment {idx}")
            consolidated_summary += response.choices[0].message.content.strip() + "\n\n"          
        except openai.OpenAIError as e:
            print(f"ERROR: OpenAI API error while processing fragment {idx}: {e}")
            break
        except Exception as e:
            print(f"ERROR: Unexpected error while processing fragment {idx}: {e}")
            break

    if not consolidated_summary:
        print("ERROR: No summary generated from the logs.")
        return None

    print("DEBUG: Formatting the consolidated summary...")
    formatted_summary = (
        f"## 📊 Consolidated Log Analysis Report - Build #{build_id}\n\n"
        f"{consolidated_summary.strip()}\n\n"
        "---\n"
        f"### 🔗 Context\n"
        f"- **Build ID**: {build_id} 🛠️\n"
        f"- **Logs Directory**: `{log_dir}` 📂\n\n"
        f"### 🚀 Recommendations\n"
        f"- Please address the identified issues promptly. 🕒\n"
        f"- Ensure all fixes are tested thoroughly. ✅\n\n"
        f"### ❤️ Thanks for contributing to the project's quality!"
    )
    return formatted_summary

def main():
    parser = argparse.ArgumentParser(description="Summarize analysis and create GitHub issue.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the directory with analysis results.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the summary output.")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository (e.g., owner/repo).")
    parser.add_argument("--run-id", type=str, required=True, help="GitHub Actions run ID.")
    parser.add_argument("--run-url", type=str, required=True, help="URL to the GitHub Actions run.")
    parser.add_argument("--create-ticket", action="store_true", help="Flag to create a GitHub issue.")
    parser.add_argument("--report-language", type=str, required=False, help="Specify the language for the summary report (e.g., German, Spanish, French). Defaults to English if not provided.")
    args = parser.parse_args()

    print(f"DEBUG: Starting log summarization for Build ID: {args.run_id}")

    build_id = args.run_id

    summary = summarize_logs_with_openai(args.log_dir, build_id, args.report_language)
    if not summary:
        print("ERROR: Could not generate summary. Exiting...")
        exit(1)

    print(f"DEBUG: Saving summary to output file: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(summary)

    print(f"DEBUG: Summary saved to {args.output_file} 🎉")

    if args.create_ticket:
        print("DEBUG: Creating GitHub issue...")
        create_github_issue("Consolidated Log Analysis Report", summary, build_id)

if __name__ == "__main__":
    main()
