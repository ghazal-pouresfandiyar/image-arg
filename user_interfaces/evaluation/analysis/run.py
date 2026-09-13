import subprocess
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

scripts = [
    ("descriptive_stats.py", "Descriptive Statistics"),
    ("visualizations.py", "Visualizations"),
    ("comprehensive_report.py", "Comprehensive Report"),
    ("profiling_report.py", "Profiling Report"),
]

print("=" * 60)
print("  RUNNING ALL ANALYSIS SCRIPTS")
print("=" * 60)
print(f"Output directory: {OUTPUT_DIR}\n")

for script_name, label in scripts:
    script_path = os.path.join(SCRIPT_DIR, script_name)
    print(f"\n{'─' * 40}")
    print(f"  Running: {label} ({script_name})")
    print(f"{'─' * 40}")
    result = subprocess.run([sys.executable, script_path], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"❌ {label} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"✅ {label} completed successfully")

print(f"\n{'=' * 60}")
print("  ALL ANALYSIS COMPLETE")
print(f"{'=' * 60}")
print(f"All results are in: {OUTPUT_DIR}")
print(f"\nContents:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    filepath = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(filepath)
    if size > 1024 * 1024:
        size_str = f"{size / (1024*1024):.1f} MB"
    elif size > 1024:
        size_str = f"{size / 1024:.1f} KB"
    else:
        size_str = f"{size} B"
    print(f"  📄 {f} ({size_str})")
