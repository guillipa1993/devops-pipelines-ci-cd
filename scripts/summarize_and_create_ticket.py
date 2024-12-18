import os
import subprocess
import argparse
import openai

# Configurar clave API de OpenAI
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: 'OPENAI_API_KEY' is not set. Please set it as an environment variable.")
    exit(1)

openai.api_key = api_key

def create_github_issue(title, body):
    """
    Crea un ticket en GitHub utilizando la CLI `gh`.
    """
    try:
        command = [
            "gh", "issue", "create",
            "--title", title,
            "--body", body
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("GitHub issue created successfully:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("ERROR: Failed to create GitHub issue.")
        print(e.stderr)

def summarize_logs_with_openai(log_dir):
    """
    Lee y resume el contenido de los archivos de análisis utilizando la API de OpenAI.
    """
    all_content = ""
    # Leer todos los archivos de log
    for filename in os.listdir(log_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(log_dir, filename)
            with open(file_path, "r") as f:
                all_content += f"### {filename}\n{f.read()}\n\n"

    if not all_content:
        print("ERROR: No valid analysis files found in the directory.")
        return None

    # Dividir contenido si es demasiado grande
    max_chunk_size = 12000
    content_fragments = [all_content[i:i + max_chunk_size] for i in range(0, len(all_content), max_chunk_size)]

    # Consultar a la API de OpenAI para resumir
    print("Summarizing log analysis with OpenAI...")
    summary = ""
    for idx, fragment in enumerate(content_fragments, 1):
        try:
            print(f"Processing fragment {idx}/{len(content_fragments)}...")
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an assistant summarizing log analysis results for a GitHub ticket. Provide clear and concise points."},
                    {"role": "user", "content": fragment}
                ],
                max_tokens=1500,
                temperature=0.5
            )
            summary += response['choices'][0]['message']['content'].strip() + "\n\n"
        except Exception as e:
            print(f"ERROR: Failed to process fragment {idx}: {e}")
            break

    return summary.strip()

def main():
    parser = argparse.ArgumentParser(description="Summarize analysis and create GitHub issue.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the directory with analysis results.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the summary output.")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository (e.g., owner/repo).")
    parser.add_argument("--run-id", type=str, required=True, help="GitHub Actions run ID.")
    parser.add_argument("--run-url", type=str, required=True, help="URL to the GitHub Actions run.")
    parser.add_argument("--create-ticket", action="store_true", help="Flag to create a GitHub issue.")
    args = parser.parse_args()

    # Resumir los logs usando OpenAI
    summary = summarize_logs_with_openai(args.log_dir)
    if not summary:
        print("ERROR: Could not generate summary. Exiting...")
        exit(1)

    # Añadir contexto adicional al resumen
    summary_with_context = (
        f"## Consolidated Log Analysis Report\n\n{summary}\n\n"
        f"---\n"
        f"### Context\n"
        f"- **Repository**: {args.repo}\n"
        f"- **Run ID**: {args.run_id}\n"
        f"- **Run URL**: [GitHub Actions Run]({args.run_url})"
    )

    # Guardar el resumen en un archivo
    os.makedirs(os.path.dirname(args.output-file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(summary_with_context)

    print(f"Summary saved to {args.output_file}")

    # Crear el ticket de GitHub si se pasa el flag `--create-ticket`
    if args.create_ticket:
        create_github_issue("Consolidated Log Analysis Report", summary_with_context)

if __name__ == "__main__":
    main()
