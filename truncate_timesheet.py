"""
Truncates 04_Timesheet_Details_clean.csv down to a target row count (or a
recent date window) so it fits under GitHub's 100MB file size limit.

Usage examples:
    # Keep only the most recent N rows (fastest, no date parsing needed)
    python truncate_timesheet.py data/Transformed/04_Timesheet_Details_clean.csv --tail-rows 200000

    # Keep only rows on/after a given date (requires a 'date' column)
    python truncate_timesheet.py data/Transformed/04_Timesheet_Details_clean.csv --since 2025-01-01
"""

import argparse
import csv
import sys


def truncate_by_tail_rows(input_path: str, output_path: str, tail_rows: int) -> None:
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    kept = rows[-tail_rows:] if tail_rows < len(rows) else rows

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(kept)

    print(f"Kept last {len(kept)} of {len(rows)} rows -> {output_path}")


def truncate_by_date(input_path: str, output_path: str, since: str, date_col: str = "date") -> None:
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if date_col not in fieldnames:
            print(f"Column '{date_col}' not found. Columns present: {fieldnames}")
            sys.exit(1)
        kept = [row for row in reader if row[date_col] >= since]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)

    print(f"Kept {len(kept)} rows with {date_col} >= {since} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--output", default=None, help="Output path (defaults to overwriting input)")
    parser.add_argument("--tail-rows", type=int, help="Keep only the last N rows")
    parser.add_argument("--since", help="Keep only rows with date >= this value (YYYY-MM-DD)")
    parser.add_argument("--date-col", default="date", help="Name of the date column (default: 'date')")
    args = parser.parse_args()

    output_path = args.output or args.input_path

    if args.tail_rows:
        truncate_by_tail_rows(args.input_path, output_path, args.tail_rows)
    elif args.since:
        truncate_by_date(args.input_path, output_path, args.since, args.date_col)
    else:
        print("Specify either --tail-rows N or --since YYYY-MM-DD")
        sys.exit(1)