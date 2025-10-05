#!/usr/bin/env python3
"""
Alternative script to check table structure by attempting insertions with different field combinations
"""

import os
from supabase import create_client, Client
import yaml
from pathlib import Path
import json

# Load config from YAML
config_path = Path(__file__).parent.parent / 'config' / 'config_v1.yaml'
with open(config_path, 'r') as f:
    CONFIG = yaml.safe_load(f)

def test_table_structure():
    """
    Test what fields exist by trying minimal inserts and seeing what's required
    """
    try:
        # Get Supabase credentials from config
        supabase_config = CONFIG.get('database', {}).get('supabase', {})
        supabase_url = supabase_config.get('url')
        supabase_key = supabase_config.get('anon_key')

        if not supabase_url or not supabase_key:
            print("❌ Missing Supabase credentials in config_v1.yaml")
            return
        
        # Create Supabase client
        supabase: Client = create_client(supabase_url, supabase_key)
        print(f"✅ Connected to Supabase: {supabase_url}")
        
        # Test 1: Try with minimal data to see what's required
        print("\n🔍 Testing minimal required fields...")
        
        minimal_data = {
            'session_id': 'test_schema_check',
            'frame_number': 0
        }
        
        try:
            result = supabase.table('musician_frame_analysis').insert(minimal_data).execute()
            print("✅ Minimal insert successful - these are the minimum required fields")
            print("Required fields:", list(minimal_data.keys()))
            
            # Clean up test record
            supabase.table('musician_frame_analysis').delete().eq('session_id', 'test_schema_check').execute()
            
        except Exception as e:
            print(f"❌ Minimal insert failed: {e}")
            error_msg = str(e)
            
            # Parse error message to find missing columns
            if "null value in column" in error_msg:
                print("🔍 Found required field in error message")
            elif "does not exist" in error_msg:
                print("🔍 Found non-existent field in error message")
        
        # Test 2: Try with the fields we know from database_setup.py
        print("\n🔍 Testing with expected fields from database_setup.py...")
        
        expected_data = {
            'session_id': 'test_schema_check_2',
            'frame_number': 0,
            'video_file': 'test.mp4',
            'fps': 30.0,
            'original_time': 0.0,
            'synced_time': 0.0,
            'left_hand_landmarks': None,
            'right_hand_landmarks': None,
            'pose_landmarks': None,
            'emotion_angry': 0.0,
            'emotion_disgust': 0.0,
            'emotion_fear': 0.0,
            'emotion_happy': 0.0,
            'emotion_sad': 0.0,
            'emotion_surprise': 0.0,
            'emotion_neutral': 0.0,
            'flag_low_wrists': False,
            'flag_turtle_neck': False,
            'flag_hunched_back': False,
            'flag_fingers_pointing_up': False,
            'processing_time_ms': 50
            # Deliberately omitting model_version to test if it exists
        }
        
        try:
            result = supabase.table('musician_frame_analysis').insert(expected_data).execute()
            print("✅ Full insert successful without model_version field")
            print("✅ Confirmed: model_version field has been removed from table")
            
            # Show the inserted record to see actual structure
            if result.data:
                record = result.data[0]
                print(f"\n📋 Actual table structure ({len(record.keys())} fields):")
                print("=" * 60)
                
                for i, (key, value) in enumerate(sorted(record.items()), 1):
                    value_type = type(value).__name__ if value is not None else 'NULL'
                    print(f"{i:2d}. {key:<25} = {value} ({value_type})")
            
            # Clean up test record
            supabase.table('musician_frame_analysis').delete().eq('session_id', 'test_schema_check_2').execute()
            print("\n🧹 Test record cleaned up")
            
            return list(record.keys()) if result.data else None
            
        except Exception as e:
            print(f"❌ Full insert failed: {e}")
            error_msg = str(e)
            
            # Try to extract field information from error
            if "does not exist" in error_msg:
                print("🔍 Some expected fields don't exist in the table")
            elif "null value in column" in error_msg:
                print("🔍 Some fields are required (NOT NULL)")
        
        # Test 3: Try with model_version to see if it still exists
        print("\n🔍 Testing if model_version field still exists...")
        
        data_with_model_version = expected_data.copy()
        data_with_model_version['session_id'] = 'test_schema_check_3'
        data_with_model_version['model_version'] = 'test_v1.0'
        
        try:
            result = supabase.table('musician_frame_analysis').insert(data_with_model_version).execute()
            print("✅ model_version field still exists in table")
            
            # Clean up
            supabase.table('musician_frame_analysis').delete().eq('session_id', 'test_schema_check_3').execute()
            
        except Exception as e:
            if "does not exist" in str(e) and "model_version" in str(e):
                print("✅ Confirmed: model_version field has been removed from table")
            else:
                print(f"❌ Error testing model_version: {e}")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")

def main():
    print("🔧 Testing musician_frame_analysis table structure...")
    test_table_structure()

if __name__ == "__main__":
    main()