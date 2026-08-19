"""
Blanks out the employee_full_name column in 01_Employee_Details_clean.csv
while keeping the column itself (so any code expecting it doesn't break).

Usage:
    python anonymize_names.py path/to/01_Employee_Details_clean.csv
"""

import sys
import csv

def anonymize(input_path: str) -> None:
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if "employee_full_name" not in fieldnames:
        print(f"Column 'employee_full_name' not found. Columns present: {fieldnames}")
        sys.exit(1)

    for row in rows:
        row["employee_full_name"] = ""  # or "NA" if you'd rather have an explicit marker

    with open(input_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Blanked 'employee_full_name' for {len(rows)} rows in {input_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python anonymize_names.py path/to/01_Employee_Details_clean.csv")
        sys.exit(1)
    anonymize(sys.argv[1])