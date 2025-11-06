"""Health check script for deployed services."""

import sys
import argparse
import requests
import time
from typing import Dict, List, Tuple


class HealthChecker:
    """Health checker for Plant Disease Detector services."""
    
    def __init__(self, timeout: int = 5, retries: int = 3):
        self.timeout = timeout
        self.retries = retries
    
    def check_endpoint(self, url: str, name: str) -> Tuple[bool, str]:
        """
        Check if an endpoint is healthy.
        
        Args:
            url: Endpoint URL
            name: Service name
            
        Returns:
            Tuple of (success, message)
        """
        for attempt in range(self.retries):
            try:
                response = requests.get(url, timeout=self.timeout)
                
                if response.status_code == 200:
                    return True, f"✓ {name} is healthy"
                else:
                    return False, f"✗ {name} returned status {response.status_code}"
                    
            except requests.exceptions.ConnectionError:
                if attempt < self.retries - 1:
                    time.sleep(2)
                    continue
                return False, f"✗ {name} is not reachable (connection error)"
                
            except requests.exceptions.Timeout:
                if attempt < self.retries - 1:
                    time.sleep(2)
                    continue
                return False, f"✗ {name} timed out"
                
            except Exception as e:
                return False, f"✗ {name} health check failed: {e}"
        
        return False, f"✗ {name} health check failed after {self.retries} retries"
    
    def check_api_health(self, base_url: str) -> Dict[str, Tuple[bool, str]]:
        """
        Check API service health.
        
        Args:
            base_url: Base URL of API service
            
        Returns:
            Dictionary of check results
        """
        results = {}
        
        # Check health endpoint
        results['health'] = self.check_endpoint(f"{base_url}/health", "API Health")
        
        # Check info endpoint
        results['info'] = self.check_endpoint(f"{base_url}/info", "API Info")
        
        # Check docs endpoint
        results['docs'] = self.check_endpoint(f"{base_url}/docs", "API Docs")
        
        return results
    
    def check_web_ui(self, base_url: str) -> Tuple[bool, str]:
        """
        Check Streamlit web UI health.
        
        Args:
            base_url: Base URL of web UI
            
        Returns:
            Tuple of (success, message)
        """
        # Streamlit health check endpoint
        health_url = f"{base_url}/_stcore/health"
        return self.check_endpoint(health_url, "Web UI")
    
    def run_all_checks(self, api_url: str = None, web_url: str = None) -> bool:
        """
        Run all health checks.
        
        Args:
            api_url: API service URL
            web_url: Web UI URL
            
        Returns:
            True if all checks pass, False otherwise
        """
        all_passed = True
        
        print("=" * 60)
        print("Plant Disease Detector - Health Check")
        print("=" * 60)
        print()
        
        # Check API
        if api_url:
            print("Checking API service...")
            api_results = self.check_api_health(api_url)
            
            for check, (success, message) in api_results.items():
                print(f"  {message}")
                if not success:
                    all_passed = False
            print()
        
        # Check Web UI
        if web_url:
            print("Checking Web UI service...")
            success, message = self.check_web_ui(web_url)
            print(f"  {message}")
            if not success:
                all_passed = False
            print()
        
        # Summary
        print("=" * 60)
        if all_passed:
            print("✓ All health checks passed")
        else:
            print("✗ Some health checks failed")
        print("=" * 60)
        
        return all_passed


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Health check for Plant Disease Detector services"
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="API service URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--web-url",
        default="http://localhost:8501",
        help="Web UI URL (default: http://localhost:8501)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=5,
        help="Request timeout in seconds (default: 5)"
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries (default: 3)"
    )
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Check API only"
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Check Web UI only"
    )
    
    args = parser.parse_args()
    
    # Create health checker
    checker = HealthChecker(timeout=args.timeout, retries=args.retries)
    
    # Determine which services to check
    check_api = not args.web_only
    check_web = not args.api_only
    
    # Run checks
    all_passed = checker.run_all_checks(
        api_url=args.api_url if check_api else None,
        web_url=args.web_url if check_web else None
    )
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
