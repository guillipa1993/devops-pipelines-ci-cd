import os
import subprocess
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="Summarize analysis and create GitHub issue.")
    parser.add_argument("--log-dir", type=str, required=True, help="Path to the directory with analysis results.")
    parser.add_argument("--output-file", type=str, required=True, help="Path to save the summary output.")
    parser.add_argument("--create-ticket", action="store_true", help="Flag to create a GitHub issue.")
    args = parser.parse_args()

    # Simulación de la generación del resumen de análisis
    summary_content = "Consolidated log analysis report:\n\n- Example item 1\n- Example item 2"
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        f.write(summary_content)

    print(f"Summary saved to {args.output_file}")

    # Crear el ticket de GitHub si se pasa el flag `--create-ticket`
    if args.create_ticket:
        with open(args.output_file, "r") as f:
            issue_body = f.read()
        create_github_issue("Consolidated Log Analysis Report", issue_body)

if __name__ == "__main__":
    main()
