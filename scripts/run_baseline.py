import argparse
import json
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

from app.agents.baseline_oneshot import run_oneshot_baseline_sync


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict single-agent one-shot baseline.")
    parser.add_argument("--task", required=True, help="Natural-language analytics task.")
    parser.add_argument("--json", action="store_true", help="Print full JSON summary.")
    args = parser.parse_args()

    result = run_oneshot_baseline_sync(args.task)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print("\nBaseline run complete.\n")
    print(f"Run dir: {result['run_dir']}")
    print(f"Model: {result['model']}")
    print(f"Task: {result['task']}")
    print(f"Code file: {result['code_path']}")
    print(f"Stdout log: {result['stdout_path']}")
    print(f"Stderr log: {result['stderr_path']}")
    print(f"Images generated: {len(result['image_files'])} -> {result['image_files']}")
    print("")
    print("Evaluation summary:")
    print(f"- execution_success: {result['execution_success']}")
    print(f"- required_steps_completed: {result['required_steps_completed']}")
    print(f"- llm_latency_sec: {result['llm_latency_sec']}")
    print(f"- execution_latency_sec: {result['execution_latency_sec']}")
    print(f"- total_latency_sec: {result['total_latency_sec']}")
    print("")
    print("Step checks:")
    for key, value in result["required_steps"].items():
        print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
