#!/usr/bin/env python3
"""
Script to inspect the actual schema of musician_frame_analysis table in Supabase
"""

import os
from supabase import create_client, Client
import yaml
from pathlib import Path

# Load config from YAML
config_path = Path(__file__).parent.parent / 'config' / 'config_v1.yaml'
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

def inspect_table_schema():
    """
    Connect to Supabase and inspect the musician_frame_analysis table schema
    """
    try:
        # Get Supabase credentials from config
        supabase_config = CONFIG.get('database', {}).get('supabase', {})
        supabase_url = supabase_config.get('url')
        supabase_key = supabase_config.get('anon_key')

        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase credentials. Please set database.supabase.url and database.supabase.anon_key in config_v1.yaml")
            return None
        
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        print(f"✅ Connected to Supabase: {supabase_url}")
        
        # Method 1: Try to get table structure by querying information schema
        try:
            # Query the PostgreSQL information_schema to get column details
            query = """
            SELECT 
                column_name, 
                data_type, 
                is_nullable, 
                column_default,
                ordinal_position
            FROM information_schema.columns 
            WHERE table_name = 'musician_frame_analysis'
            ORDER BY ordinal_position;
            """
            
            # Use RPC to execute the query
            result = supabase.rpc('exec_sql', {'sql': query}).execute()
            
            if result.data:
                print("\n📋 Table Schema (musician_frame_analysis):")
                print("=" * 80)
                print(f"{'Column Name':<25} {'Data Type':<20} {'Nullable':<10} {'Default'}")
                print("-" * 80)
                
                columns = []
                for row in result.data:
                    col_name = row['column_name']
                    data_type = row['data_type']
                    nullable = row['is_nullable']
                    default = row['column_default'] if row['column_default'] else 'NULL'
                    
                    print(f"{col_name:<25} {data_type:<20} {nullable:<10} {default}")
                    columns.append({
                        'name': col_name,
                        'type': data_type,
                        'nullable': nullable == 'YES',
                        'default': default
                    })
                
                return columns
                
        except Exception as e:
            print(f"⚠️ Method 1 failed: {e}")
            
        # Method 2: Try to get one record and inspect its structure
        try:
            print("\n🔍 Trying to inspect table by querying one record...")
            result = supabase.table('musician_frame_analysis').select('*').limit(1).execute()
            
            if result.data and len(result.data) > 0:
                record = result.data[0]
                print(f"\n📋 Found {len(record.keys())} columns in musician_frame_analysis:")
                print("=" * 50)
                
                columns = []
                for i, (key, value) in enumerate(record.items(), 1):
                    value_type = type(value).__name__ if value is not None else 'None'
                    print(f"{i:2d}. {key:<25} (sample: {value_type})")
                    columns.append({
                        'name': key,
                        'type': value_type,
                        'nullable': value is None,
                        'sample_value': value
                    })
                
                return columns
            else:
                print("⚠️ No records found in table")
                
        except Exception as e:
            print(f"⚠️ Method 2 failed: {e}")
            
        # Method 3: Try a simple select to see if table exists
        try:
            print("\n🔍 Checking if table exists...")
            result = supabase.table('musician_frame_analysis').select('count', count='exact').execute()
            print(f"✅ Table exists with {result.count} records")
            
            # Try to get column names by selecting with limit 0
            result = supabase.table('musician_frame_analysis').select('*').limit(0).execute()
            print("✅ Table is accessible but we need records to see column structure")
            
        except Exception as e:
            print(f"❌ Method 3 failed: {e}")
            
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return None
    
    return None

def main():
    print("🔧 Inspecting Supabase table schema...")
    columns = inspect_table_schema()
    
    if columns:
        print(f"\n✅ Successfully retrieved {len(columns)} columns")
        print("\n📝 Column summary for detect.py integration:")
        print("=" * 60)
        
        required_fields = []
        optional_fields = []
        
        for col in columns:
            col_name = col['name']
            if col_name in ['id', 'created_at']:
                continue  # Skip auto-generated fields
                
            if col.get('nullable', True) or col.get('default') not in [None, 'NULL']:
                optional_fields.append(col_name)
            else:
                required_fields.append(col_name)
        
        print("Required fields:")
        for field in required_fields:
            print(f"  - {field}")
            
        print("\nOptional fields:")
        for field in optional_fields:
            print(f"  - {field}")
            
    else:
        print("❌ Could not retrieve table schema")

if __name__ == "__main__":
    main()