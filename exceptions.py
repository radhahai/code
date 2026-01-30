"""
exceptions.py - Custom Exception Classes

This module defines custom exceptions for the seismic analysis application.
Provides specific error types for better error handling and debugging.
"""

from typing import Optional, Any


class SeismicAppError(Exception):
    """Base exception for all seismic application errors."""
    
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ValidationError(SeismicAppError):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, value: Any, constraint: str):
        self.field = field
        self.value = value
        self.constraint = constraint
        message = f"Validation failed for '{field}': value {value} does not satisfy {constraint}"
        super().__init__(message)


class MatrixError(SeismicAppError):
    """Raised when matrix operations fail (singular, ill-conditioned, etc.)."""
    
    def __init__(self, operation: str, matrix_name: str, details: Optional[str] = None):
        self.operation = operation
        self.matrix_name = matrix_name
        message = f"Matrix error during {operation} on {matrix_name}"
        super().__init__(message, details)


class SolverError(SeismicAppError):
    """Raised when the time-stepping solver encounters an error."""
    
    def __init__(self, solver_name: str, step: int, details: Optional[str] = None):
        self.solver_name = solver_name
        self.step = step
        message = f"Solver '{solver_name}' failed at step {step}"
        super().__init__(message, details)


class ConvergenceError(SolverError):
    """Raised when iterative solver fails to converge."""
    
    def __init__(self, solver_name: str, step: int, iterations: int, residual: float):
        self.iterations = iterations
        self.residual = residual
        details = f"Max iterations ({iterations}) reached. Residual: {residual:.2e}"
        super().__init__(solver_name, step, details)


class GroundMotionError(SeismicAppError):
    """Raised when ground motion generation fails."""
    
    def __init__(self, event_type: str, details: Optional[str] = None):
        self.event_type = event_type
        message = f"Failed to generate ground motion for event type '{event_type}'"
        super().__init__(message, details)


class ConfigurationError(SeismicAppError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, details: Optional[str] = None):
        self.config_key = config_key
        message = f"Invalid or missing configuration: '{config_key}'"
        super().__init__(message, details)


class ReportGenerationError(SeismicAppError):
    """Raised when PDF/report generation fails."""
    
    def __init__(self, report_type: str, details: Optional[str] = None):
        self.report_type = report_type
        message = f"Failed to generate {report_type} report"
        super().__init__(message, details)


class NumericalInstabilityError(SeismicAppError):
    """Raised when numerical instability is detected (NaN, Inf values)."""
    
    def __init__(self, location: str, variable: str):
        self.location = location
        self.variable = variable
        message = f"Numerical instability detected in {location}: {variable} contains NaN/Inf"
        super().__init__(message)
