"""Download Olist's public dataset and build a local, indexed SQLite warehouse."""
from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import kagglehub
import pandas as pd

DATASET_HANDLE = "olistbr/brazilian-ecommerce"
DATABASE_PATH = Path(__file__).resolve().with_name("olist_analytics.db")
SOURCE_FILES: dict[str, str] = {
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
}
INDEX_STATEMENTS = (
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_customers_customer_id ON customers(customer_id)",
    "CREATE INDEX IF NOT EXISTS ix_customers_unique_id ON customers(customer_unique_id)",
    "CREATE UNIQUE INDEX IF NOT EXISTS ux_orders_order_id ON orders(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_orders_customer_id ON orders(customer_id)",
    "CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(order_status)",
    "CREATE INDEX IF NOT EXISTS ix_orders_purchase_date ON orders(order_purchase_timestamp)",
    "CREATE INDEX IF NOT EXISTS ix_items_order_id ON order_items(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_payments_order_id ON order_payments(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_reviews_order_id ON order_reviews(order_id)",
    "CREATE INDEX IF NOT EXISTS ix_geolocation_zip ON geolocation(geolocation_zip_code_prefix)",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_source_data() -> dict[str, pd.DataFrame]:
    """Download through KaggleHub and load all eight required CSVs."""
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    frames: dict[str, pd.DataFrame] = {}
    for table_name, filename in SOURCE_FILES.items():
        csv_path = dataset_path / filename
        if not csv_path.is_file():
            raise FileNotFoundError(f"Kaggle dataset is missing required file: {csv_path}")
        frame = pd.read_csv(csv_path, low_memory=False)
        if frame.empty:
            raise ValueError(f"Required source file is empty: {filename}")
        frames[table_name] = frame
        LOGGER.info("Loaded %-15s %8s rows", table_name, f"{len(frame):,}")
    return frames


def build_sqlite(frames: dict[str, pd.DataFrame], db_path: Path = DATABASE_PATH) -> None:
    """Atomically replace the local SQLite warehouse and recreate its indexes."""
    db_path = db_path.expanduser().resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = db_path.with_suffix(db_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        with sqlite3.connect(temporary_path) as connection:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute("PRAGMA foreign_keys=ON")
            for table_name, frame in frames.items():
                frame.to_sql(table_name, connection, if_exists="replace", index=False, chunksize=10_000)
            for statement in INDEX_STATEMENTS:
                connection.execute(statement)
            connection.execute("ANALYZE")
            connection.commit()
        temporary_path.replace(db_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    LOGGER.info("SQLite warehouse built at %s", db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and build the Olist SQLite analytics warehouse.")
    parser.add_argument("--db", type=Path, default=DATABASE_PATH, help="Output SQLite database path.")
    args = parser.parse_args()
    frames = load_source_data()
    build_sqlite(frames, args.db)


if __name__ == "__main__":
    main()

