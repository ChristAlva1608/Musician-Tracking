#!/usr/bin/env python3
"""
CSV to Database Loader Utility
Load CSV files into Supabase or PostgreSQL database tables with configurable table names
"""

import os
import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from supabase import create_client, Client
import yaml
from typing import Optional, List, Dict, Any, Union
import argparse
from pathlib import Path
import json
from datetime import datetime

# Load config from YAML
def load_config():
    """Load configuration from config_v1.yaml"""
    # Try to find config relative to this file
    config_path = Path(__file__).parent.parent / 'config' / 'config_v1.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()


class CSVToDatabaseLoader:
    """Load CSV files into Supabase or PostgreSQL database tables"""

    def __init__(self, table_name: str = 'csv_data', db_type: str = 'supabase'):
        """
        Initialize database connection

        Args:
            table_name: Name of the table to load CSV data into
            db_type: Database type - 'supabase' or 'postgres'
        """
        self.table_name = table_name
        self.db_type = db_type.lower()
        self.connection = None
        self.cursor = None
        self.supabase = None

        if self.db_type == 'supabase':
            self._init_supabase()
        elif self.db_type == 'postgres':
            self._init_postgres()
        else:
            raise ValueError(f"Unsupported database type: {db_type}. Use 'supabase' or 'postgres'")

        print(f"✅ Connected to {self.db_type.upper()} database")
        print(f"📋 Target table: {self.table_name}")

    def _init_supabase(self):
        """Initialize Supabase connection"""
        supabase_config = CONFIG.get('database', {}).get('supabase', {})
        self.supabase_url = supabase_config.get('url')
        self.supabase_key = supabase_config.get('anon_key')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Please set database.supabase.url and database.supabase.anon_key in config_v1.yaml"
            )

        self.supabase = create_client(self.supabase_url, self.supabase_key)
        print(f"🔗 Supabase URL: {self.supabase_url}")

    def _init_postgres(self):
        """Initialize PostgreSQL connection"""
        # Get PostgreSQL connection parameters from config or use defaults
        local_config = CONFIG.get('database', {}).get('local', {})
        self.pg_host = local_config.get('host', 'localhost')
        self.pg_port = local_config.get('port', 5432)
        self.pg_database = local_config.get('name', 'postgres')
        self.pg_user = local_config.get('user', 'postgres')
        self.pg_password = local_config.get('password', '')

        try:
            self.connection = psycopg2.connect(
                host=self.pg_host,
                port=self.pg_port,
                database=self.pg_database,
                user=self.pg_user,
                password=self.pg_password
            )
            self.cursor = self.connection.cursor()
            print(f"🔗 PostgreSQL: {self.pg_user}@{self.pg_host}:{self.pg_port}/{self.pg_database}")
        except Exception as e:
            raise ValueError(f"Failed to connect to PostgreSQL: {e}")

    def create_table_from_csv(self, csv_file_path: str, primary_key: str = 'id') -> bool:
        """
        Create table based on CSV structure (PostgreSQL only)

        Args:
            csv_file_path: Path to the CSV file
            primary_key: Name of primary key column (will be added if not in CSV)

        Returns:
            True if successful, False otherwise
        """
        if self.db_type != 'postgres':
            print("⚠️ Table creation only supported for PostgreSQL. Use Supabase UI for Supabase tables.")
            return False

        try:
            # Read CSV to infer schema
            df = pd.read_csv(csv_file_path, nrows=100)  # Sample for type inference

            # Clean column names
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]

            # Build CREATE TABLE statement
            create_sql = f"CREATE TABLE IF NOT EXISTS {self.table_name} ("

            # Add primary key if not in columns
            if primary_key not in df.columns:
                create_sql += f"{primary_key} SERIAL PRIMARY KEY, "

            # Map pandas dtypes to PostgreSQL types
            type_mapping = {
                'int64': 'BIGINT',
                'int32': 'INTEGER',
                'int16': 'SMALLINT',
                'float64': 'DOUBLE PRECISION',
                'float32': 'REAL',
                'object': 'TEXT',
                'bool': 'BOOLEAN',
                'datetime64[ns]': 'TIMESTAMP',
                'timedelta64[ns]': 'INTERVAL'
            }

            for col in df.columns:
                dtype = str(df[col].dtype)
                pg_type = type_mapping.get(dtype, 'TEXT')

                # Check if column should be primary key
                if col == primary_key and primary_key in df.columns:
                    create_sql += f"{col} {pg_type} PRIMARY KEY, "
                else:
                    # Check for nulls to determine if NULL constraint needed
                    if df[col].isnull().any():
                        create_sql += f"{col} {pg_type}, "
                    else:
                        create_sql += f"{col} {pg_type} NOT NULL, "

            # Remove trailing comma and close
            create_sql = create_sql.rstrip(', ') + ");"

            print(f"📝 Creating table with SQL:\n{create_sql}")

            self.cursor.execute(create_sql)
            self.connection.commit()

            print(f"✅ Table '{self.table_name}' created successfully")
            return True

        except Exception as e:
            print(f"❌ Error creating table: {e}")
            if self.connection:
                self.connection.rollback()
            return False

    def load_csv(self, csv_file_path: str,
                 batch_size: int = 100,
                 mode: str = 'append',
                 create_table: bool = False) -> bool:
        """
        Load CSV file into database table

        Args:
            csv_file_path: Path to the CSV file
            batch_size: Number of rows to insert per batch
            mode: 'append' to add to existing data, 'replace' to clear table first
            create_table: Whether to create table if it doesn't exist (PostgreSQL only)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if file exists
            if not Path(csv_file_path).exists():
                print(f"❌ CSV file not found: {csv_file_path}")
                return False

            # Read CSV file
            print(f"📖 Reading CSV file: {csv_file_path}")
            df = pd.read_csv(csv_file_path)
            print(f"📊 Found {len(df)} rows and {len(df.columns)} columns")
            print(f"📋 Columns: {', '.join(df.columns)}")

            # Clean column names
            df.columns = [col.lower().replace(' ', '_').replace('-', '_') for col in df.columns]

            # Create table if requested (PostgreSQL only)
            if create_table and self.db_type == 'postgres':
                self.create_table_from_csv(csv_file_path)

            # Replace NaN values with None for proper serialization
            df = df.where(pd.notnull(df), None)

            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')

            # Clear existing data if mode is 'replace'
            if mode == 'replace':
                self._clear_table()

            # Insert data based on database type
            if self.db_type == 'supabase':
                return self._load_to_supabase(records, batch_size)
            else:
                return self._load_to_postgres(records, df.columns.tolist(), batch_size)

        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False

    def _clear_table(self):
        """Clear all data from the table"""
        print(f"🗑️ Clearing existing data from table: {self.table_name}")

        if self.db_type == 'supabase':
            try:
                self.supabase.table(self.table_name).delete().neq('id', -1).execute()
            except Exception as e:
                print(f"⚠️ Could not clear table (may be empty): {e}")
        else:
            try:
                self.cursor.execute(f"TRUNCATE TABLE {self.table_name} RESTART IDENTITY CASCADE;")
                self.connection.commit()
                print("✅ Table cleared")
            except Exception as e:
                print(f"⚠️ Could not clear table: {e}")
                self.connection.rollback()

    def _load_to_supabase(self, records: List[Dict], batch_size: int) -> bool:
        """Load data to Supabase"""
        total_inserted = 0

        for i in range(0, len(records), batch_size):
            batch = records[i:i+batch_size]

            try:
                result = self.supabase.table(self.table_name).insert(batch).execute()
                total_inserted += len(batch)
                print(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} rows (Total: {total_inserted}/{len(records)})")
            except Exception as batch_error:
                print(f"❌ Error inserting batch {i//batch_size + 1}: {batch_error}")
                continue

        print(f"✅ Successfully loaded {total_inserted} rows into Supabase table '{self.table_name}'")
        return total_inserted > 0

    def _load_to_postgres(self, records: List[Dict], columns: List[str], batch_size: int) -> bool:
        """Load data to PostgreSQL"""
        total_inserted = 0

        # Prepare INSERT statement
        placeholders = ', '.join(['%s'] * len(columns))
        columns_str = ', '.join(columns)
        insert_sql = f"INSERT INTO {self.table_name} ({columns_str}) VALUES ({placeholders})"

        try:
            for i in range(0, len(records), batch_size):
                batch = records[i:i+batch_size]

                # Convert records to tuples for psycopg2
                batch_tuples = [tuple(record.get(col) for col in columns) for record in batch]

                # Use execute_batch for better performance
                execute_batch(self.cursor, insert_sql, batch_tuples, page_size=batch_size)
                total_inserted += len(batch)

                print(f"✅ Inserted batch {i//batch_size + 1}: {len(batch)} rows (Total: {total_inserted}/{len(records)})")

                # Commit periodically
                if (i + batch_size) % (batch_size * 10) == 0:
                    self.connection.commit()

            # Final commit
            self.connection.commit()
            print(f"✅ Successfully loaded {total_inserted} rows into PostgreSQL table '{self.table_name}'")
            return True

        except Exception as e:
            print(f"❌ Error during PostgreSQL insert: {e}")
            self.connection.rollback()
            return False

    def preview_csv(self, csv_file_path: str, rows: int = 5) -> pd.DataFrame:
        """
        Preview CSV file contents

        Args:
            csv_file_path: Path to the CSV file
            rows: Number of rows to preview

        Returns:
            DataFrame with preview data
        """
        try:
            df = pd.read_csv(csv_file_path, nrows=rows)
            print(f"\n📊 Preview of first {rows} rows:")
            print(df)
            print(f"\n📋 Data types:")
            print(df.dtypes)
            return df
        except Exception as e:
            print(f"❌ Error previewing CSV: {e}")
            return None

    def get_csv_info(self, csv_file_path: str) -> Dict[str, Any]:
        """
        Get information about CSV file

        Args:
            csv_file_path: Path to the CSV file

        Returns:
            Dictionary with CSV file information
        """
        try:
            df = pd.read_csv(csv_file_path)

            info = {
                'file_path': csv_file_path,
                'file_size_mb': Path(csv_file_path).stat().st_size / (1024 * 1024),
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns),
                'column_types': {k: str(v) for k, v in df.dtypes.to_dict().items()},
                'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
                'has_null_values': df.isnull().any().any()
            }

            print("\n📊 CSV File Information:")
            print(f"📁 File: {info['file_path']}")
            print(f"💾 Size: {info['file_size_mb']:.2f} MB")
            print(f"📈 Rows: {info['rows']:,}")
            print(f"📊 Columns: {info['columns']}")
            print(f"🔤 Column Names: {', '.join(info['column_names'])}")
            print(f"💭 Memory Usage: {info['memory_usage_mb']:.2f} MB")
            print(f"❓ Has Null Values: {info['has_null_values']}")

            return info
        except Exception as e:
            print(f"❌ Error getting CSV info: {e}")
            return None

    def close(self):
        """Close database connections"""
        if self.db_type == 'postgres' and self.connection:
            self.cursor.close()
            self.connection.close()
            print("🔒 PostgreSQL connection closed")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description='Load CSV files into Supabase or PostgreSQL database tables',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  For Supabase:
    SUPABASE_URL         - Your Supabase project URL
    SUPABASE_ANON_KEY    - Your Supabase anon/public key

  For PostgreSQL:
    POSTGRES_HOST        - PostgreSQL host (default: localhost)
    POSTGRES_PORT        - PostgreSQL port (default: 5432)
    POSTGRES_DATABASE    - Database name (default: postgres)
    POSTGRES_USER        - PostgreSQL username (default: postgres)
    POSTGRES_PASSWORD    - PostgreSQL password

Examples:
  # Load to Supabase
  python csv_to_database.py data.csv --table my_table --db supabase

  # Load to local PostgreSQL
  python csv_to_database.py data.csv --table my_table --db postgres

  # Create table and load (PostgreSQL only)
  python csv_to_database.py data.csv --table my_table --db postgres --create-table

  # Replace existing data
  python csv_to_database.py data.csv --table my_table --mode replace

  # Preview before loading
  python csv_to_database.py data.csv --table my_table --preview
        """
    )

    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--table', '-t', default='csv_data',
                       help='Target table name (default: csv_data)')
    parser.add_argument('--db', '-d', choices=['supabase', 'postgres'], default='supabase',
                       help='Database type: supabase or postgres (default: supabase)')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                       help='Batch size for inserts (default: 100)')
    parser.add_argument('--mode', '-m', choices=['append', 'replace'], default='append',
                       help='Insert mode: append or replace existing data (default: append)')
    parser.add_argument('--create-table', '-c', action='store_true',
                       help='Create table if it doesn\'t exist (PostgreSQL only)')
    parser.add_argument('--preview', '-p', action='store_true',
                       help='Preview CSV before loading')
    parser.add_argument('--info', '-i', action='store_true',
                       help='Show CSV file information')

    args = parser.parse_args()

    loader = None
    try:
        # Initialize loader with specified table name and database type
        loader = CSVToDatabaseLoader(table_name=args.table, db_type=args.db)

        # Show CSV info if requested
        if args.info:
            loader.get_csv_info(args.csv_file)
            if not args.preview:
                return

        # Preview CSV if requested
        if args.preview:
            loader.preview_csv(args.csv_file, rows=10)
            response = input("\n📤 Continue with loading? (y/n): ")
            if response.lower() != 'y':
                print("❌ Loading cancelled")
                return

        # Load CSV into database
        success = loader.load_csv(
            csv_file_path=args.csv_file,
            batch_size=args.batch_size,
            mode=args.mode,
            create_table=args.create_table
        )

        if success:
            print(f"\n✅ CSV loading to {args.db.upper()} completed successfully!")
        else:
            print(f"\n❌ CSV loading to {args.db.upper()} failed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)
    finally:
        if loader:
            loader.close()


if __name__ == "__main__":
    main()