import sys
import os
import subprocess
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")

with open(CONFIG_PATH, "r", encoding="utf8") as f:
    config = yaml.safe_load(f)

base_dir = os.path.dirname(os.path.abspath(__file__))
steps = config.get("pipeline", [])

print(f"Running pipeline with {len(steps)} steps\n")

for i, step in enumerate(steps, 1):
    name = step["name"]
    script = os.path.join(base_dir, step["script"])

    print(f"[{i}/{len(steps)}] {name}")
    print(f"  -> {script}")

    result = subprocess.run([sys.executable, script], cwd=base_dir)

    if result.returncode != 0:
        print(f"  FAILED (exit code {result.returncode})")
        sys.exit(1)

    print(f"  Done\n")

print("Pipeline completed successfully.")
