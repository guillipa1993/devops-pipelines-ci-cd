import openai
import os
import subprocess
import argparse

# Configurar clave API de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

openai.api_key = api_key

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

def summarize_logs_with_openai(log_dir, build_id):
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
    max_chunk_size = 12000
    content_fragments = [all_content[i:i + max_chunk_size] for i in range(0, len(all_content), max_chunk_size)]

    print(f"DEBUG: Total fragments to process: {len(content_fragments)}")
    consolidated_summary = ""
    for idx, fragment in enumerate(content_fragments, 1):
        try:
            print(f"DEBUG: Processing fragment {idx}/{len(content_fragments)} with OpenAI API...")
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": (
                        "You are an assistant summarizing log analysis results for a GitHub ticket. "
                        "Provide a clear and concise summary with insights, recommendations, and context. "
                        "The output should be structured, friendly, and compatible for a GitHub issue."
                    )},
                    {"role": "user", "content": fragment}
                ]
            )
            print(f"DEBUG: OpenAI response received for fragment {idx}")
            consolidated_summary += response.choices[0].message['content'].strip() + "\n\n"
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
    args = parser.parse_args()

    print(f"DEBUG: Starting log summarization for Build ID: {args.run_id}")

    build_id = args.run_id

    summary = summarize_logs_with_openai(args.log_dir, build_id)
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
