"""
health.py - Health Check Utilities

Provides health check and diagnostic functions for production monitoring.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, List
import importlib

from logging_config import get_logger

logger = get_logger("health")


def check_dependencies() -> Dict[str, Any]:
    """
    Check if all required dependencies are installed and importable.
    
    Returns
    -------
    dict
        Status of each dependency
    """
    required_packages = [
        ("numpy", "np"),
        ("scipy", None),
        ("pandas", "pd"),
        ("streamlit", "st"),
        ("plotly", None),
        ("reportlab", None),
    ]
    
    results = {}
    all_ok = True
    
    for package, alias in required_packages:
        try:
            mod = importlib.import_module(package)
            version = getattr(mod, "__version__", "unknown")
            results[package] = {
                "status": "ok",
                "version": version
            }
        except ImportError as e:
            results[package] = {
                "status": "error",
                "error": str(e)
            }
            all_ok = False
    
    return {
        "all_ok": all_ok,
        "packages": results
    }


def check_numerical_backend() -> Dict[str, Any]:
    """
    Verify numerical backend is working correctly.
    
    Returns
    -------
    dict
        Test results
    """
    import numpy as np
    from scipy.linalg import eigh
    
    results = {}
    
    try:
        # Test basic numpy operations
        A = np.random.randn(100, 100)
        B = A @ A.T  # Positive definite
        results["numpy_matmul"] = "ok"
        
        # Test scipy eigenvalue decomposition
        eig_vals, eig_vecs = eigh(B)
        results["scipy_eigh"] = "ok"
        
        # Check for NaN/Inf
        if np.any(np.isnan(eig_vals)) or np.any(np.isinf(eig_vals)):
            results["numerical_stability"] = "warning"
        else:
            results["numerical_stability"] = "ok"
        
        results["all_ok"] = True
        
    except Exception as e:
        results["error"] = str(e)
        results["all_ok"] = False
    
    return results


def get_system_info() -> Dict[str, Any]:
    """
    Get system information for diagnostics.
    
    Returns
    -------
    dict
        System information
    """
    import platform
    import numpy as np
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy_config": {
            "version": np.__version__,
            "blas_info": str(np.__config__.show() if hasattr(np.__config__, 'show') else "N/A")
        },
        "cwd": os.getcwd(),
        "pid": os.getpid()
    }


def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check for production monitoring.
    
    Returns
    -------
    dict
        Complete health status
    """
    start_time = datetime.now()
    
    health = {
        "status": "healthy",
        "timestamp": start_time.isoformat(),
        "checks": {}
    }
    
    # Check dependencies
    dep_check = check_dependencies()
    health["checks"]["dependencies"] = dep_check
    if not dep_check["all_ok"]:
        health["status"] = "unhealthy"
    
    # Check numerical backend
    num_check = check_numerical_backend()
    health["checks"]["numerical"] = num_check
    if not num_check.get("all_ok", False):
        health["status"] = "unhealthy"
    
    # System info
    health["system"] = get_system_info()
    
    # Response time
    health["response_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
    
    return health


def run_self_test() -> Dict[str, Any]:
    """
    Run a quick self-test of core functionality.
    
    Returns
    -------
    dict
        Test results
    """
    results = {
        "tests_passed": 0,
        "tests_failed": 0,
        "details": []
    }
    
    # Test 1: Matrix assembly
    try:
        from models import StructuralProperties, ControlDevices
        from structural import build_structural_matrices
        
        props = StructuralProperties(n_floors=5)
        controls = ControlDevices()
        M, K, C = build_structural_matrices(props, controls)
        
        assert M.shape == (15, 15), "Wrong matrix shape"
        assert K.shape == (15, 15), "Wrong matrix shape"
        
        results["tests_passed"] += 1
        results["details"].append({"test": "matrix_assembly", "status": "passed"})
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append({"test": "matrix_assembly", "status": "failed", "error": str(e)})
    
    # Test 2: Ground motion generation
    try:
        from models import SeismicEvent
        from ground_motion import generate_ground_motion
        
        t, ag = generate_ground_motion(SeismicEvent.KANAI_TAJIMI, 0.3, 5.0, 0.01, "medium")
        
        assert len(t) > 0, "Empty time array"
        assert len(ag) == len(t), "Length mismatch"
        
        results["tests_passed"] += 1
        results["details"].append({"test": "ground_motion", "status": "passed"})
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append({"test": "ground_motion", "status": "failed", "error": str(e)})
    
    # Test 3: Modal analysis
    try:
        from structural import perform_modal_analysis
        
        periods, modes = perform_modal_analysis(M, K)
        
        assert len(periods) > 0, "No periods returned"
        assert periods[0] > 0, "Invalid period"
        
        results["tests_passed"] += 1
        results["details"].append({"test": "modal_analysis", "status": "passed"})
    except Exception as e:
        results["tests_failed"] += 1
        results["details"].append({"test": "modal_analysis", "status": "failed", "error": str(e)})
    
    results["all_passed"] = results["tests_failed"] == 0
    
    return results


if __name__ == "__main__":
    # Run health check when executed directly
    import json
    
    print("Running health check...")
    health = health_check()
    print(json.dumps(health, indent=2, default=str))
    
    print("\nRunning self-test...")
    test_results = run_self_test()
    print(json.dumps(test_results, indent=2))
