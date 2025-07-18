"""
Sacred Geometry Calculator for MUSE Platform

This module implements the mathematical foundations of sacred geometry
used in Computational Platonism creative discovery. It provides core
calculations for golden ratio, fibonacci sequences, pi relationships,
and other sacred mathematical constants.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class SacredConstant(Enum):
    """Sacred mathematical constants used in MUSE calculations"""
    PHI = "phi"  # Golden ratio
    PI = "pi"    # Circle constant
    E = "e"      # Euler's number
    SQRT_2 = "sqrt_2"  # Square root of 2
    SQRT_3 = "sqrt_3"  # Square root of 3
    SQRT_5 = "sqrt_5"  # Square root of 5


@dataclass
class SacredGeometryResult:
    """Result container for sacred geometry calculations"""
    value: float
    constant_used: SacredConstant
    calculation_type: str
    metadata: Dict[str, Any]


class SacredGeometryCalculator:
    """
    Core sacred geometry calculator for MUSE platform
    
    Implements mathematical constants and relationships fundamental
    to Computational Platonism creative discovery.
    """
    
    # Sacred mathematical constants
    PHI = (1 + math.sqrt(5)) / 2  # Golden ratio: 1.618033988749...
    PI = math.pi
    E = math.e
    SQRT_2 = math.sqrt(2)
    SQRT_3 = math.sqrt(3)
    SQRT_5 = math.sqrt(5)
    
    def __init__(self):
        """Initialize the sacred geometry calculator"""
        self.precision = 15  # Default precision for calculations
        self._fibonacci_cache = {0: 0, 1: 1}  # Cache for fibonacci calculations
        
    def golden_ratio_sequence(self, n: int) -> List[float]:
        """
        Generate sequence based on golden ratio powers
        
        Args:
            n: Number of terms to generate
            
        Returns:
            List of golden ratio powers
        """
        if n <= 0:
            return []
            
        sequence = []
        for i in range(n):
            value = self.PHI ** i
            sequence.append(value)
            
        return sequence
    
    def fibonacci_sequence(self, n: int) -> List[int]:
        """
        Generate Fibonacci sequence up to n terms
        
        Args:
            n: Number of Fibonacci terms to generate
            
        Returns:
            List of Fibonacci numbers
        """
        if n <= 0:
            return []
        elif n == 1:
            return [0]
        elif n == 2:
            return [0, 1]
            
        fib = [0, 1]
        for i in range(2, n):
            fib.append(fib[i-1] + fib[i-2])
            
        return fib
    
    def fibonacci_ratio_convergence(self, n: int) -> List[float]:
        """
        Calculate convergence of Fibonacci ratios to golden ratio
        
        Args:
            n: Number of ratios to calculate
            
        Returns:
            List of ratios converging to phi
        """
        if n < 2:
            return []
            
        fib = self.fibonacci_sequence(n + 1)
        ratios = []
        
        for i in range(1, len(fib) - 1):
            if fib[i] != 0:
                ratio = fib[i + 1] / fib[i]
                ratios.append(ratio)
                
        return ratios
    
    def pentagonal_numbers(self, n: int) -> List[int]:
        """
        Generate pentagonal numbers (related to golden ratio)
        
        Args:
            n: Number of pentagonal numbers to generate
            
        Returns:
            List of pentagonal numbers
        """
        return [i * (3 * i - 1) // 2 for i in range(1, n + 1)]
    
    def sacred_spiral_points(self, n: int, scale: float = 1.0) -> List[Tuple[float, float]]:
        """
        Generate points on golden spiral (sacred spiral)
        
        Args:
            n: Number of points to generate
            scale: Scale factor for the spiral
            
        Returns:
            List of (x, y) coordinates on golden spiral
        """
        points = []
        angle_step = 2 * self.PI / self.PHI
        
        for i in range(n):
            angle = i * angle_step
            radius = scale * (self.PHI ** (i / 10))  # Spiral grows by phi
            
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            points.append((x, y))
            
        return points
    
    def vesica_piscis_ratio(self) -> float:
        """
        Calculate the ratio in Vesica Piscis (sacred geometry shape)
        
        Returns:
            The mathematical ratio of Vesica Piscis
        """
        return self.SQRT_3
    
    def platonic_solid_angles(self, solid_type: str) -> Dict[str, float]:
        """
        Calculate angles for Platonic solids
        
        Args:
            solid_type: Type of Platonic solid ('tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron')
            
        Returns:
            Dictionary of angles in the solid
        """
        angles = {
            'tetrahedron': {
                'dihedral_angle': math.acos(1/3),
                'face_angle': math.pi/3
            },
            'cube': {
                'dihedral_angle': math.pi/2,
                'face_angle': math.pi/2
            },
            'octahedron': {
                'dihedral_angle': math.acos(-1/3),
                'face_angle': math.pi/3
            },
            'dodecahedron': {
                'dihedral_angle': math.acos(-self.SQRT_5/3),
                'face_angle': math.pi - math.pi/5
            },
            'icosahedron': {
                'dihedral_angle': math.acos(-self.SQRT_5/3),
                'face_angle': math.pi/3
            }
        }
        
        return angles.get(solid_type, {})
    
    def sacred_triangle_ratios(self) -> Dict[str, float]:
        """
        Calculate ratios for sacred triangles
        
        Returns:
            Dictionary of sacred triangle ratios
        """
        return {
            'golden_gnomon': self.PHI,
            'golden_gnomons_reciprocal': 1 / self.PHI,
            'egyptian_triangle': 5 / 4,  # 3-4-5 triangle
            'kepler_triangle': self.PHI / self.SQRT_5,
            'pentagonal_triangle': (self.PHI + 1) / 2
        }
    
    def calculate_sacred_frequency(self, base_frequency: float, harmonic: int) -> float:
        """
        Calculate sacred frequency based on golden ratio harmonics
        
        Args:
            base_frequency: Base frequency in Hz
            harmonic: Harmonic number
            
        Returns:
            Sacred frequency using golden ratio
        """
        return base_frequency * (self.PHI ** (harmonic - 1))
    
    def mandala_geometry(self, layers: int, points_per_layer: int) -> List[List[Tuple[float, float]]]:
        """
        Generate mandala geometry points
        
        Args:
            layers: Number of concentric layers
            points_per_layer: Number of points per layer
            
        Returns:
            List of layers, each containing (x, y) coordinates
        """
        mandala_points = []
        
        for layer in range(1, layers + 1):
            layer_points = []
            radius = layer * self.PHI  # Each layer scaled by golden ratio
            
            for point in range(points_per_layer):
                angle = (2 * self.PI * point) / points_per_layer
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                layer_points.append((x, y))
                
            mandala_points.append(layer_points)
            
        return mandala_points
    
    def sacred_rectangle_dimensions(self, width: float) -> Tuple[float, float]:
        """
        Calculate dimensions for sacred rectangle (golden rectangle)
        
        Args:
            width: Width of the rectangle
            
        Returns:
            Tuple of (width, height) for golden rectangle
        """
        height = width / self.PHI
        return (width, height)
    
    def calculate_sacred_ratio(self, numerator: float, denominator: float) -> SacredGeometryResult:
        """
        Analyze if a ratio relates to sacred geometry constants
        
        Args:
            numerator: Numerator of the ratio
            denominator: Denominator of the ratio
            
        Returns:
            SacredGeometryResult with analysis
        """
        if denominator == 0:
            raise ValueError("Denominator cannot be zero")
            
        ratio = numerator / denominator
        tolerance = 1e-10
        
        # Check against sacred constants
        sacred_constants = {
            SacredConstant.PHI: self.PHI,
            SacredConstant.PI: self.PI,
            SacredConstant.E: self.E,
            SacredConstant.SQRT_2: self.SQRT_2,
            SacredConstant.SQRT_3: self.SQRT_3,
            SacredConstant.SQRT_5: self.SQRT_5
        }
        
        for constant, value in sacred_constants.items():
            if abs(ratio - value) < tolerance:
                return SacredGeometryResult(
                    value=ratio,
                    constant_used=constant,
                    calculation_type="ratio_analysis",
                    metadata={
                        "numerator": numerator,
                        "denominator": denominator,
                        "match_precision": abs(ratio - value)
                    }
                )
        
        # Check against reciprocals
        for constant, value in sacred_constants.items():
            if abs(ratio - (1/value)) < tolerance:
                return SacredGeometryResult(
                    value=ratio,
                    constant_used=constant,
                    calculation_type="reciprocal_ratio_analysis",
                    metadata={
                        "numerator": numerator,
                        "denominator": denominator,
                        "reciprocal_of": constant.value,
                        "match_precision": abs(ratio - (1/value))
                    }
                )
        
        # No sacred constant match found
        return SacredGeometryResult(
            value=ratio,
            constant_used=None,
            calculation_type="ordinary_ratio",
            metadata={
                "numerator": numerator,
                "denominator": denominator,
                "no_sacred_match": True
            }
        )
    
    def generate_sacred_matrix(self, size: int) -> np.ndarray:
        """
        Generate matrix with sacred geometry properties
        
        Args:
            size: Size of the square matrix
            
        Returns:
            NumPy array with sacred geometry values
        """
        matrix = np.zeros((size, size))
        
        for i in range(size):
            for j in range(size):
                # Use golden ratio in matrix generation
                value = (self.PHI ** i) * math.cos(2 * self.PI * j / size)
                matrix[i, j] = value
                
        return matrix
    
    def sacred_geometry_validation(self, value: float) -> Dict[str, Any]:
        """
        Validate if a value has sacred geometry significance
        
        Args:
            value: Value to validate
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            "input_value": value,
            "sacred_matches": [],
            "fibonacci_relation": None,
            "geometric_significance": []
        }
        
        tolerance = 1e-10
        
        # Check against sacred constants
        sacred_constants = {
            "phi": self.PHI,
            "pi": self.PI,
            "e": self.E,
            "sqrt_2": self.SQRT_2,
            "sqrt_3": self.SQRT_3,
            "sqrt_5": self.SQRT_5
        }
        
        for name, constant in sacred_constants.items():
            if abs(value - constant) < tolerance:
                validation_results["sacred_matches"].append(name)
        
        # Check Fibonacci relation
        fib_sequence = self.fibonacci_sequence(20)
        for i, fib_num in enumerate(fib_sequence):
            if abs(value - fib_num) < tolerance:
                validation_results["fibonacci_relation"] = {
                    "index": i,
                    "fibonacci_number": fib_num
                }
                break
        
        # Check geometric significance
        if abs(value - 60) < tolerance:  # 60 degrees
            validation_results["geometric_significance"].append("equilateral_triangle_angle")
        if abs(value - 90) < tolerance:  # 90 degrees
            validation_results["geometric_significance"].append("right_angle")
        if abs(value - 108) < tolerance:  # Pentagon interior angle
            validation_results["geometric_significance"].append("pentagon_interior_angle")
        
        return validation_results