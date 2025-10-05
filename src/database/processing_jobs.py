import os
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
import yaml
from pathlib import Path

# Load config from YAML
def load_config():
    """Load configuration from config_v1.yaml"""
    config_path = Path(__file__).parent.parent / 'config' / 'config_v1.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

class ProcessingJobsDatabase:
    """
    Database handler for processing job tracking and management
    """

    def __init__(self, config: Optional[dict] = None):
        """Initialize Supabase connection for processing jobs

        Args:
            config: Configuration dictionary. If None, will load from config_v1.yaml
        """
        # Load config if not provided
        if config is None:
            config = CONFIG

        supabase_config = config.get('database', {}).get('supabase', {})
        self.supabase_url = supabase_config.get('url')
        self.supabase_key = supabase_config.get('anon_key')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing Supabase credentials. Please set database.supabase.url and database.supabase.anon_key in config_v1.yaml"
            )

        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.table_name = 'processing_jobs'

        print(f" ProcessingJobsDatabase connected to Supabase")

        # Note: Table creation SQL is in src/database/sql_code.sql
        # Run that SQL in Supabase dashboard to create the processing_jobs table

    def create_job(self, job_id: str, job_type: str, request_data: Dict[str, Any]) -> bool:
        """
        Create a new processing job record

        Args:
            job_id: Unique job identifier
            job_type: Type of processing job (e.g., 'single_video', 'folder_processing')
            request_data: Original request parameters

        Returns:
            True if successful, False otherwise
        """
        try:
            job_data = {
                'job_id': job_id,
                'type': job_type,
                'status': 'queued',
                'progress': 0,
                'message': 'Job created',
                'request_data': json.dumps(request_data) if request_data else None,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }

            result = self.supabase.table(self.table_name).insert(job_data).execute()
            print(f" Created job {job_id} of type {job_type}")
            return True

        except Exception as e:
            print(f"L Error creating job {job_id}: {e}")
            return False

    def update_job(self, job_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a processing job record

        Args:
            job_id: Job identifier
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        try:
            # Add updated_at timestamp
            updates['updated_at'] = datetime.now().isoformat()

            # If status is being set to completed/failed, add completed_at
            if updates.get('status') in ['completed', 'failed', 'cancelled']:
                updates['completed_at'] = datetime.now().isoformat()

            # Convert output_files list to JSON if present
            if 'output_files' in updates and isinstance(updates['output_files'], list):
                updates['output_files'] = json.dumps(updates['output_files'])

            result = self.supabase.table(self.table_name).update(updates).eq('job_id', job_id).execute()
            print(f" Updated job {job_id}")
            return True

        except Exception as e:
            print(f"L Error updating job {job_id}: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific job by ID

        Args:
            job_id: Job identifier

        Returns:
            Job data dictionary or None if not found
        """
        try:
            result = self.supabase.table(self.table_name).select('*').eq('job_id', job_id).execute()

            if result.data and len(result.data) > 0:
                job = result.data[0]

                # Parse JSON fields
                if job.get('request_data') and isinstance(job['request_data'], str):
                    try:
                        job['request_data'] = json.loads(job['request_data'])
                    except:
                        pass

                if job.get('output_files') and isinstance(job['output_files'], str):
                    try:
                        job['output_files'] = json.loads(job['output_files'])
                    except:
                        job['output_files'] = []
                elif not job.get('output_files'):
                    job['output_files'] = []

                # Add logs field if not present (for compatibility)
                if 'logs' not in job:
                    job['logs'] = []
                    if job.get('message'):
                        job['logs'].append({
                            'timestamp': job.get('updated_at', ''),
                            'message': job['message']
                        })

                return job

            return None

        except Exception as e:
            print(f"L Error fetching job {job_id}: {e}")
            return None

    def get_all_jobs(self, limit: int = 100, offset: int = 0,
                     status: Optional[str] = None,
                     job_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all processing jobs with optional filtering

        Args:
            limit: Maximum number of jobs to return
            offset: Number of jobs to skip
            status: Filter by status (optional)
            job_type: Filter by job type (optional)

        Returns:
            List of job dictionaries
        """
        try:
            query = self.supabase.table(self.table_name).select('*')

            if status:
                query = query.eq('status', status)

            if job_type:
                query = query.eq('type', job_type)

            result = query.order('created_at', desc=True).limit(limit).offset(offset).execute()

            jobs = result.data if result.data else []

            # Parse JSON fields for each job
            for job in jobs:
                if job.get('request_data') and isinstance(job['request_data'], str):
                    try:
                        job['request_data'] = json.loads(job['request_data'])
                    except:
                        pass

                if job.get('output_files') and isinstance(job['output_files'], str):
                    try:
                        job['output_files'] = json.loads(job['output_files'])
                    except:
                        job['output_files'] = []
                elif not job.get('output_files'):
                    job['output_files'] = []

                # Add logs field for compatibility
                if 'logs' not in job:
                    job['logs'] = []
                    if job.get('message'):
                        job['logs'].append({
                            'timestamp': job.get('updated_at', ''),
                            'message': job['message']
                        })

            return jobs

        except Exception as e:
            print(f"L Error fetching jobs: {e}")
            return []

    def get_job_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processing jobs

        Returns:
            Dictionary with job statistics
        """
        try:
            all_jobs = self.get_all_jobs(limit=1000)  # Get more jobs for statistics

            stats = {
                'total': len(all_jobs),
                'by_status': {},
                'by_type': {},
                'recent_24h': 0,
                'average_processing_time': 0
            }

            # Count by status and type
            for job in all_jobs:
                status = job.get('status', 'unknown')
                job_type = job.get('type', 'unknown')

                stats['by_status'][status] = stats['by_status'].get(status, 0) + 1
                stats['by_type'][job_type] = stats['by_type'].get(job_type, 0) + 1

                # Count recent jobs (last 24 hours)
                if job.get('created_at'):
                    try:
                        created = datetime.fromisoformat(job['created_at'].replace('Z', '+00:00'))
                        if created > datetime.now(created.tzinfo) - timedelta(days=1):
                            stats['recent_24h'] += 1
                    except:
                        pass

            return stats

        except Exception as e:
            print(f"L Error calculating statistics: {e}")
            return {
                'total': 0,
                'by_status': {},
                'by_type': {},
                'recent_24h': 0
            }

    def cleanup_old_jobs(self, days: int = 30) -> int:
        """
        Clean up old completed/failed jobs

        Args:
            days: Number of days to keep jobs

        Returns:
            Number of deleted jobs
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

            # Delete old completed and failed jobs
            result = self.supabase.table(self.table_name).delete().in_(
                'status', ['completed', 'failed', 'cancelled']
            ).lt('created_at', cutoff_date).execute()

            deleted_count = len(result.data) if result.data else 0
            print(f" Cleaned up {deleted_count} old jobs")

            return deleted_count

        except Exception as e:
            print(f"L Error cleaning up old jobs: {e}")
            return 0

    def get_active_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all currently active (queued or running) jobs

        Returns:
            List of active job dictionaries
        """
        try:
            result = self.supabase.table(self.table_name).select('*').in_(
                'status', ['queued', 'running']
            ).order('created_at').execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"L Error fetching active jobs: {e}")
            return []
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a specific job from the database

        Args:
            job_id: Job identifier to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete the job from database
            result = self.supabase.table(self.table_name).delete().eq("job_id", job_id).execute()

            # Check if any rows were deleted
            if result.data and len(result.data) > 0:
                print(f"✅ Deleted job {job_id} from database")
                return True
            else:
                print(f"⚠️ Job {job_id} not found in database")
                return False

        except Exception as e:
            print(f"❌ Error deleting job {job_id}: {e}")
            return False
