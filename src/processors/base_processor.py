"""
Base Processor Class
Common functionality and error handling for all processors
"""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Dict, Any, List


class ProcessorError(Exception):
    """Base exception for processor errors"""
    pass


class DependencyError(ProcessorError):
    """Raised when required dependencies are missing"""
    pass


class ValidationError(ProcessorError):
    """Raised when validation fails"""
    pass


class BaseProcessor(ABC):
    """Base class for all processor modules"""

    def __init__(self, config: Dict[str, Any], session_id: str):
        """
        Initialize base processor

        Args:
            config: Configuration dictionary
            session_id: Unique session identifier
        """
        self.config = config
        self.session_id = session_id
        self.results = {}

    @contextmanager
    def _timer(self, step_name: str):
        """
        Context manager for timing operations

        Args:
            step_name: Name of the step being timed
        """
        start_time = time.time()
        print(f"⏱️  [{step_name}] Starting...")
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            if minutes > 0:
                print(f"⏱️  [{step_name}] Completed in {minutes}m {seconds:.2f}s")
            else:
                print(f"⏱️  [{step_name}] Completed in {seconds:.2f}s")

    @abstractmethod
    def validate_dependencies(self):
        """
        Validate that all required dependencies are available
        Should raise DependencyError if dependencies are missing
        """
        pass

    @abstractmethod
    def validate_inputs(self):
        """
        Validate input parameters and configuration
        Should raise ValidationError if validation fails
        """
        pass

    @abstractmethod
    def process(self) -> Dict[str, Any]:
        """
        Execute the main processing logic

        Returns:
            Dictionary containing processing results

        Raises:
            ProcessorError: If processing fails
        """
        pass

    def run(self) -> Dict[str, Any]:
        """
        Run the processor with full validation and error handling

        Returns:
            Dictionary containing processing results

        Raises:
            DependencyError: If dependencies are missing
            ValidationError: If validation fails
            ProcessorError: If processing fails
        """
        try:
            # Step 1: Validate dependencies
            print(f"\n{'='*60}")
            print(f"{self.__class__.__name__} - Dependency Validation")
            print(f"{'='*60}")
            self.validate_dependencies()
            print("✅ All dependencies validated\n")

            # Step 2: Validate inputs
            print(f"{'='*60}")
            print(f"{self.__class__.__name__} - Input Validation")
            print(f"{'='*60}")
            self.validate_inputs()
            print("✅ All inputs validated\n")

            # Step 3: Execute processing
            print(f"{'='*60}")
            print(f"{self.__class__.__name__} - Processing")
            print(f"{'='*60}")
            results = self.process()
            print(f"✅ Processing completed successfully\n")

            return results

        except DependencyError as e:
            print(f"\n❌ DEPENDENCY ERROR: {e}")
            print("   Processing stopped. Please install missing dependencies.")
            raise
        except ValidationError as e:
            print(f"\n❌ VALIDATION ERROR: {e}")
            print("   Processing stopped. Please check your configuration.")
            raise
        except ProcessorError as e:
            print(f"\n❌ PROCESSING ERROR: {e}")
            print("   Processing stopped.")
            raise
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR: {e}")
            import traceback
            traceback.print_exc()
            raise ProcessorError(f"Unexpected error in {self.__class__.__name__}: {e}") from e

    @staticmethod
    def check_package(package_name: str, import_name: str = None) -> bool:
        """
        Check if a package is installed

        Args:
            package_name: Name of the package (for error messages)
            import_name: Name to use for import (if different from package_name)

        Returns:
            True if package is available, False otherwise
        """
        if import_name is None:
            import_name = package_name

        try:
            __import__(import_name)
            return True
        except ImportError:
            return False

    @staticmethod
    def require_packages(packages: List[Dict[str, str]]):
        """
        Require that packages are installed, raise DependencyError if not

        Args:
            packages: List of package dicts with 'name' and optional 'import_name'

        Raises:
            DependencyError: If any required package is missing
        """
        missing = []
        for pkg in packages:
            name = pkg['name']
            import_name = pkg.get('import_name', name)
            if not BaseProcessor.check_package(name, import_name):
                install_name = pkg.get('install_name', name)
                missing.append(install_name)

        if missing:
            raise DependencyError(
                f"Missing required packages: {', '.join(missing)}\n"
                f"Install with: pip install {' '.join(missing)}"
            )
