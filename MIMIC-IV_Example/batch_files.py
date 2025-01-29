import argparse

import polars as pl


def batch_labevents(input_file: str, output_dir: str, rows_per_parquet: int = 10_000_000):
    reader = pl.read_csv_batched(input_file, batch_size=rows_per_parquet)
    file_counter = 1
    batches = reader.next_batches(1)
    while batches:
        for df_batch in batches:
            out_file = f"{output_dir}/labevents-{file_counter:02}.parquet"
            df_batch.write_parquet(out_file)
            print(f"Saved {out_file}")
            file_counter += 1
        batches = reader.next_batches(1)


def batch_chartevents(input_file: str, output_dir: str, rows_per_parquet: int = 10_000_000):
    reader = pl.read_csv_batched(input_file, batch_size=rows_per_parquet)
    file_counter = 1
    batches = reader.next_batches(1)
    while batches:
        for df_batch in batches:
            out_file = f"{output_dir}/chartevents-{file_counter:02}.parquet"
            df_batch.write_parquet(out_file)
            print(f"Saved {out_file}")
            file_counter += 1
        batches = reader.next_batches(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type", choices=["labevents", "chartevents", "both"], default="both", help="Data type to batch."
    )
    parser.add_argument(
        "--lab_input_file",
        default="/mnt/data/meds_transforms/raw_data/labevents_all.csv",
        help="Path to the CSV file including all labevents.",
    )
    parser.add_argument(
        "--lab_output_dir",
        default="/mnt/data/meds_transforms/raw_data/hosp",
        help="Directory to save output Parquet files.",
    )
    parser.add_argument(
        "--chart_input_file",
        default="/mnt/data/meds_transforms/raw_data/chartevents_all.csv",
        help="Path to the the CSV file including all chartevents.",
    )
    parser.add_argument(
        "--chart_output_dir",
        default="/mnt/data/meds_transforms/raw_data/icu",
        help="Directory to save output Parquet files.",
    )
    parser.add_argument(
        "--rows_per_parquet", type=int, default=10_000_000, help="Number of rows per Parquet file."
    )
    args = parser.parse_args()

    if args.type == "labevents":
        batch_labevents(args.lab_input_file, args.lab_output_dir, args.rows_per_parquet)
    elif args.type == "chartevents":
        batch_chartevents(args.chart_input_file, args.chart_output_dir, args.rows_per_parquet)
    else:
        batch_labevents(args.lab_input_file, args.lab_output_dir, args.rows_per_parquet)
        batch_chartevents(args.chart_input_file, args.chart_output_dir, args.rows_per_parquet)


if __name__ == "__main__":
    main()
