import argparse
from pathlib import Path


def remove_nan_rows(input_path: Path, output_path: Path) -> tuple[int, int]:
    total_data_rows = 0
    removed_rows = 0

    with input_path.open("r", encoding="utf-8", newline="") as src, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        for line in src:
            stripped = line.strip()

            if not stripped:
                dst.write(line)
                continue

            if stripped.startswith("#"):
                dst.write(line)
                continue

            # Keep header as-is.
            if stripped.startswith("time_ms,"):
                dst.write(line)
                continue

            total_data_rows += 1
            cells = [cell.strip().lower() for cell in stripped.split(",")]
            if "nan" in cells:
                removed_rows += 1
                continue

            dst.write(line)

    return total_data_rows, removed_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove CSV rows containing NaN values.")
    parser.add_argument("input_csv", help="Path to source CSV file")
    parser.add_argument(
        "-o",
        "--output",
        help="Path to output CSV file (default: <input_stem>_clean.csv)",
    )
    args = parser.parse_args()

    input_path = Path(args.input_csv)
    output_path = (
        Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_clean.csv")
    )

    total_data_rows, removed_rows = remove_nan_rows(input_path, output_path)

    print(f"Saved: {output_path}")
    print(f"Data rows: {total_data_rows}")
    print(f"Removed rows with NaN: {removed_rows}")


if __name__ == "__main__":
    main()
