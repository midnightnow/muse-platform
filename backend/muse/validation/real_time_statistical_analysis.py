"""
Real-time Statistical Analysis Module for MUSE Validation

This module provides comprehensive real-time statistical analysis capabilities
for ongoing validation experiments, including live monitoring, adaptive sampling,
and dynamic statistical inference.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from enum import Enum
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, mannwhitneyu, chi2_contingency, pearsonr
import pandas as pd
from collections import deque, defaultdict


class AnalysisType(Enum):
    """Types of real-time statistical analysis"""
    DESCRIPTIVE = "descriptive"
    INFERENTIAL = "inferential"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"


class StatisticalTest(Enum):
    """Statistical tests for real-time analysis"""
    T_TEST_INDEPENDENT = "t_test_independent"
    T_TEST_PAIRED = "t_test_paired"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON_SIGNED_RANK = "wilcoxon_signed_rank"
    CHI_SQUARE = "chi_square"
    CORRELATION = "correlation"
    REGRESSION = "regression"
    ANOVA = "anova"


@dataclass
class RealTimeDataPoint:
    """Single data point for real-time analysis"""
    timestamp: datetime
    experiment_id: str
    participant_id: str
    condition: str
    metric_name: str
    value: float
    metadata: Dict[str, Any]


@dataclass
class StatisticalResult:
    """Result of statistical analysis"""
    test_type: StatisticalTest
    statistic: float
    p_value: float
    effect_size: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    degrees_of_freedom: Optional[int]
    interpretation: str
    significant: bool
    timestamp: datetime
    sample_size: int
    metadata: Dict[str, Any]


@dataclass
class RealTimeAnalysisConfig:
    """Configuration for real-time analysis"""
    experiment_id: str
    analysis_types: List[AnalysisType]
    update_interval: float  # seconds
    minimum_sample_size: int
    significance_level: float
    effect_size_threshold: float
    
    # Adaptive sampling parameters
    adaptive_sampling: bool = False
    stopping_criteria: Dict[str, Any] = None
    interim_analysis_intervals: List[int] = None
    
    # Quality control
    outlier_detection: bool = True
    outlier_threshold: float = 3.0
    data_quality_threshold: float = 0.7


class RealTimeStatisticalAnalyzer:
    """
    Real-time statistical analyzer for ongoing validation experiments
    
    Provides continuous statistical analysis with live monitoring,
    adaptive sampling, and dynamic inference capabilities.
    """
    
    def __init__(self, config: RealTimeAnalysisConfig):
        """
        Initialize the real-time analyzer
        
        Args:
            config: Configuration for real-time analysis
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.data_buffer: deque = deque(maxlen=10000)  # Store recent data points
        self.data_by_condition: Dict[str, List[RealTimeDataPoint]] = defaultdict(list)
        self.data_by_metric: Dict[str, List[RealTimeDataPoint]] = defaultdict(list)
        
        # Analysis state
        self.analysis_history: List[StatisticalResult] = []
        self.current_statistics: Dict[str, Any] = {}
        self.is_running = False
        self.analysis_task: Optional[asyncio.Task] = None
        
        # Quality control
        self.outliers_detected: List[RealTimeDataPoint] = []
        self.data_quality_scores: List[float] = []
        
        # Adaptive sampling
        self.stopping_criteria_met = False
        self.next_interim_analysis = 0
        
        # Performance monitoring
        self.analysis_times: List[float] = []
        self.update_count = 0
        
        self.logger.info(f"Real-time analyzer initialized for experiment {config.experiment_id}")
    
    async def start_analysis(self):
        """Start real-time analysis"""
        if self.is_running:
            self.logger.warning("Analysis already running")
            return
        
        self.is_running = True
        self.analysis_task = asyncio.create_task(self._analysis_loop())
        
        self.logger.info(f"Started real-time analysis for experiment {self.config.experiment_id}")
    
    async def stop_analysis(self):
        """Stop real-time analysis"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"Stopped real-time analysis for experiment {self.config.experiment_id}")
    
    async def add_data_point(self, data_point: RealTimeDataPoint):
        """Add a new data point for real-time analysis"""
        # Quality control
        if self.config.outlier_detection and self._is_outlier(data_point):
            self.outliers_detected.append(data_point)
            self.logger.warning(f"Outlier detected: {data_point.value} for metric {data_point.metric_name}")
            return
        
        # Add to buffers
        self.data_buffer.append(data_point)
        self.data_by_condition[data_point.condition].append(data_point)
        self.data_by_metric[data_point.metric_name].append(data_point)
        
        # Update quality score
        self._update_data_quality_score(data_point)
        
        self.logger.debug(f"Added data point: {data_point.metric_name} = {data_point.value}")
    
    async def add_batch_data(self, data_points: List[RealTimeDataPoint]):
        """Add multiple data points at once"""
        for data_point in data_points:
            await self.add_data_point(data_point)
    
    async def _analysis_loop(self):
        """Main analysis loop"""
        while self.is_running:
            try:
                start_time = time.time()
                
                # Check if we have enough data
                if len(self.data_buffer) >= self.config.minimum_sample_size:
                    # Perform analyses
                    await self._perform_analyses()
                    
                    # Check stopping criteria
                    if self._check_stopping_criteria():
                        self.logger.info("Stopping criteria met - halting analysis")
                        break
                    
                    # Check interim analysis
                    if self._check_interim_analysis():
                        await self._perform_interim_analysis()
                
                # Record analysis time
                analysis_time = time.time() - start_time
                self.analysis_times.append(analysis_time)
                self.update_count += 1
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                await asyncio.sleep(self.config.update_interval)
    
    async def _perform_analyses(self):
        """Perform all configured analyses"""
        for analysis_type in self.config.analysis_types:
            if analysis_type == AnalysisType.DESCRIPTIVE:
                await self._perform_descriptive_analysis()
            elif analysis_type == AnalysisType.INFERENTIAL:
                await self._perform_inferential_analysis()
            elif analysis_type == AnalysisType.COMPARATIVE:
                await self._perform_comparative_analysis()
            elif analysis_type == AnalysisType.PREDICTIVE:
                await self._perform_predictive_analysis()
            elif analysis_type == AnalysisType.ADAPTIVE:
                await self._perform_adaptive_analysis()
    
    async def _perform_descriptive_analysis(self):
        """Perform descriptive statistical analysis"""
        descriptive_stats = {}
        
        # Overall statistics
        all_values = [dp.value for dp in self.data_buffer]
        descriptive_stats['overall'] = {
            'count': len(all_values),
            'mean': np.mean(all_values),
            'median': np.median(all_values),
            'std': np.std(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'skewness': stats.skew(all_values),
            'kurtosis': stats.kurtosis(all_values)
        }
        
        # By condition
        descriptive_stats['by_condition'] = {}
        for condition, data_points in self.data_by_condition.items():
            values = [dp.value for dp in data_points]
            if values:
                descriptive_stats['by_condition'][condition] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values)
                }
        
        # By metric
        descriptive_stats['by_metric'] = {}
        for metric, data_points in self.data_by_metric.items():
            values = [dp.value for dp in data_points]
            if values:
                descriptive_stats['by_metric'][metric] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'skewness': stats.skew(values),
                    'kurtosis': stats.kurtosis(values)
                }
        
        # Time series analysis
        descriptive_stats['time_series'] = self._analyze_time_series()
        
        self.current_statistics['descriptive'] = descriptive_stats
    
    async def _perform_inferential_analysis(self):
        """Perform inferential statistical analysis"""
        inferential_results = {}
        
        # Get conditions
        conditions = list(self.data_by_condition.keys())
        
        if len(conditions) >= 2:
            # Compare conditions
            for i, condition1 in enumerate(conditions):
                for condition2 in conditions[i+1:]:
                    comparison_key = f"{condition1}_vs_{condition2}"
                    
                    # Get data for both conditions
                    data1 = [dp.value for dp in self.data_by_condition[condition1]]
                    data2 = [dp.value for dp in self.data_by_condition[condition2]]
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        # Perform statistical tests
                        comparison_results = await self._compare_groups(data1, data2, condition1, condition2)
                        inferential_results[comparison_key] = comparison_results
        
        # Single group tests
        for condition, data_points in self.data_by_condition.items():
            values = [dp.value for dp in data_points]
            if len(values) >= 5:
                # One-sample t-test against theoretical mean
                theoretical_mean = 0.5  # Assuming 0-1 scale
                t_stat, p_value = stats.ttest_1samp(values, theoretical_mean)
                
                inferential_results[f"{condition}_vs_theoretical"] = {
                    'test_type': 'one_sample_t_test',
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level,
                    'sample_size': len(values),
                    'theoretical_mean': theoretical_mean,
                    'sample_mean': np.mean(values)
                }
        
        self.current_statistics['inferential'] = inferential_results
    
    async def _perform_comparative_analysis(self):
        """Perform comparative analysis across metrics"""
        comparative_results = {}
        
        # Get metrics
        metrics = list(self.data_by_metric.keys())
        
        if len(metrics) >= 2:
            # Compare metrics
            for i, metric1 in enumerate(metrics):
                for metric2 in metrics[i+1:]:
                    comparison_key = f"{metric1}_vs_{metric2}"
                    
                    # Get data for both metrics
                    data1 = [dp.value for dp in self.data_by_metric[metric1]]
                    data2 = [dp.value for dp in self.data_by_metric[metric2]]
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        # Correlation analysis
                        if len(data1) == len(data2):
                            correlation, p_value = pearsonr(data1, data2)
                            comparative_results[comparison_key] = {
                                'test_type': 'correlation',
                                'correlation': correlation,
                                'p_value': p_value,
                                'significant': p_value < self.config.significance_level,
                                'sample_size': len(data1),
                                'interpretation': self._interpret_correlation(correlation)
                            }
                        
                        # Compare distributions
                        comparison_results = await self._compare_groups(data1, data2, metric1, metric2)
                        comparative_results[f"{comparison_key}_distribution"] = comparison_results
        
        # Effect size analysis
        comparative_results['effect_sizes'] = self._calculate_effect_sizes()
        
        self.current_statistics['comparative'] = comparative_results
    
    async def _perform_predictive_analysis(self):
        """Perform predictive analysis"""
        predictive_results = {}
        
        # Trend analysis
        predictive_results['trends'] = self._analyze_trends()
        
        # Prediction intervals
        predictive_results['predictions'] = self._generate_predictions()
        
        # Sample size projections
        predictive_results['sample_size_projections'] = self._project_sample_sizes()
        
        self.current_statistics['predictive'] = predictive_results
    
    async def _perform_adaptive_analysis(self):
        """Perform adaptive analysis for sample size adjustments"""
        adaptive_results = {}
        
        # Power analysis
        adaptive_results['power_analysis'] = self._perform_power_analysis()
        
        # Sequential analysis
        adaptive_results['sequential_analysis'] = self._perform_sequential_analysis()
        
        # Futility analysis
        adaptive_results['futility_analysis'] = self._perform_futility_analysis()
        
        self.current_statistics['adaptive'] = adaptive_results
    
    async def _compare_groups(self, data1: List[float], data2: List[float], 
                            group1_name: str, group2_name: str) -> Dict[str, Any]:
        """Compare two groups using multiple statistical tests"""
        results = {}
        
        # Independent t-test
        t_stat, p_value = ttest_ind(data1, data2)
        results['t_test'] = {
            'statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.config.significance_level,
            'effect_size': self._calculate_cohens_d(data1, data2),
            'interpretation': self._interpret_t_test(t_stat, p_value)
        }
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_u = mannwhitneyu(data1, data2, alternative='two-sided')
        results['mann_whitney'] = {
            'statistic': u_stat,
            'p_value': p_value_u,
            'significant': p_value_u < self.config.significance_level,
            'interpretation': self._interpret_mann_whitney(u_stat, p_value_u)
        }
        
        # Welch's t-test (unequal variances)
        t_stat_welch, p_value_welch = ttest_ind(data1, data2, equal_var=False)
        results['welch_t_test'] = {
            'statistic': t_stat_welch,
            'p_value': p_value_welch,
            'significant': p_value_welch < self.config.significance_level,
            'interpretation': self._interpret_t_test(t_stat_welch, p_value_welch)
        }
        
        # Confidence interval for difference
        diff_mean = np.mean(data1) - np.mean(data2)
        se_diff = np.sqrt(np.var(data1)/len(data1) + np.var(data2)/len(data2))
        ci_lower = diff_mean - 1.96 * se_diff
        ci_upper = diff_mean + 1.96 * se_diff
        
        results['confidence_interval'] = {
            'difference_mean': diff_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': 0.95
        }
        
        return results
    
    def _analyze_time_series(self) -> Dict[str, Any]:
        """Analyze time series patterns in the data"""
        # Sort data by timestamp
        sorted_data = sorted(self.data_buffer, key=lambda x: x.timestamp)
        
        if len(sorted_data) < 5:
            return {'insufficient_data': True}
        
        # Extract time series
        timestamps = [dp.timestamp for dp in sorted_data]
        values = [dp.value for dp in sorted_data]
        
        # Calculate time differences
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)]
        
        # Basic time series statistics
        time_series_stats = {
            'data_points': len(values),
            'time_span': (timestamps[-1] - timestamps[0]).total_seconds(),
            'mean_interval': np.mean(time_diffs),
            'median_interval': np.median(time_diffs),
            'interval_variance': np.var(time_diffs),
            'trend': self._calculate_trend(values),
            'autocorrelation': self._calculate_autocorrelation(values),
            'stationarity': self._test_stationarity(values)
        }
        
        return time_series_stats
    
    def _calculate_trend(self, values: List[float]) -> Dict[str, Any]:
        """Calculate trend in time series"""
        if len(values) < 3:
            return {'trend': 'insufficient_data'}
        
        # Linear regression to find trend
        x = np.arange(len(values))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
        
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value**2,
            'p_value': p_value,
            'direction': trend_direction,
            'significant': p_value < 0.05
        }
    
    def _calculate_autocorrelation(self, values: List[float]) -> Dict[str, Any]:
        """Calculate autocorrelation in time series"""
        if len(values) < 10:
            return {'autocorrelation': 'insufficient_data'}
        
        # Calculate autocorrelation at different lags
        autocorr_results = {}
        for lag in [1, 2, 3, 5]:
            if len(values) > lag:
                autocorr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                autocorr_results[f'lag_{lag}'] = autocorr
        
        return autocorr_results
    
    def _test_stationarity(self, values: List[float]) -> Dict[str, Any]:
        """Test for stationarity in time series"""
        if len(values) < 20:
            return {'stationarity': 'insufficient_data'}
        
        # Simple stationarity test - compare first and second half
        mid_point = len(values) // 2
        first_half = values[:mid_point]
        second_half = values[mid_point:]
        
        # Compare means
        t_stat, p_value = ttest_ind(first_half, second_half)
        
        return {
            'mean_stationarity': {
                'statistic': t_stat,
                'p_value': p_value,
                'stationary': p_value > 0.05  # Null hypothesis: means are equal
            },
            'variance_stationarity': {
                'first_half_var': np.var(first_half),
                'second_half_var': np.var(second_half),
                'variance_ratio': np.var(first_half) / np.var(second_half)
            }
        }
    
    def _calculate_effect_sizes(self) -> Dict[str, Any]:
        """Calculate effect sizes for all comparisons"""
        effect_sizes = {}
        
        conditions = list(self.data_by_condition.keys())
        if len(conditions) >= 2:
            for i, condition1 in enumerate(conditions):
                for condition2 in conditions[i+1:]:
                    data1 = [dp.value for dp in self.data_by_condition[condition1]]
                    data2 = [dp.value for dp in self.data_by_condition[condition2]]
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        cohens_d = self._calculate_cohens_d(data1, data2)
                        effect_sizes[f"{condition1}_vs_{condition2}"] = {
                            'cohens_d': cohens_d,
                            'interpretation': self._interpret_effect_size(cohens_d),
                            'sample_size_1': len(data1),
                            'sample_size_2': len(data2)
                        }
        
        return effect_sizes
    
    def _calculate_cohens_d(self, data1: List[float], data2: List[float]) -> float:
        """Calculate Cohen's d effect size"""
        n1, n2 = len(data1), len(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
        
        return cohens_d
    
    def _interpret_effect_size(self, cohens_d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)
        if abs_corr < 0.1:
            return 'negligible'
        elif abs_corr < 0.3:
            return 'small'
        elif abs_corr < 0.5:
            return 'medium'
        elif abs_corr < 0.7:
            return 'large'
        else:
            return 'very large'
    
    def _interpret_t_test(self, t_stat: float, p_value: float) -> str:
        """Interpret t-test results"""
        if p_value < 0.001:
            return 'highly significant'
        elif p_value < 0.01:
            return 'very significant'
        elif p_value < 0.05:
            return 'significant'
        else:
            return 'not significant'
    
    def _interpret_mann_whitney(self, u_stat: float, p_value: float) -> str:
        """Interpret Mann-Whitney U test results"""
        if p_value < 0.001:
            return 'highly significant'
        elif p_value < 0.01:
            return 'very significant'
        elif p_value < 0.05:
            return 'significant'
        else:
            return 'not significant'
    
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in the data"""
        trends = {}
        
        # Overall trend
        all_values = [dp.value for dp in sorted(self.data_buffer, key=lambda x: x.timestamp)]
        if len(all_values) >= 5:
            trends['overall'] = self._calculate_trend(all_values)
        
        # Trends by condition
        for condition, data_points in self.data_by_condition.items():
            values = [dp.value for dp in sorted(data_points, key=lambda x: x.timestamp)]
            if len(values) >= 5:
                trends[f'condition_{condition}'] = self._calculate_trend(values)
        
        return trends
    
    def _generate_predictions(self) -> Dict[str, Any]:
        """Generate predictions for future data points"""
        predictions = {}
        
        # Simple linear extrapolation
        all_values = [dp.value for dp in sorted(self.data_buffer, key=lambda x: x.timestamp)]
        if len(all_values) >= 10:
            # Fit linear model
            x = np.arange(len(all_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, all_values)
            
            # Predict next 5 points
            future_points = []
            for i in range(1, 6):
                future_x = len(all_values) + i
                predicted_value = slope * future_x + intercept
                
                # Prediction interval
                se_pred = std_err * np.sqrt(1 + 1/len(all_values) + 
                                           (future_x - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
                
                future_points.append({
                    'step': i,
                    'predicted_value': predicted_value,
                    'confidence_interval': [
                        predicted_value - 1.96 * se_pred,
                        predicted_value + 1.96 * se_pred
                    ]
                })
            
            predictions['linear_extrapolation'] = {
                'model_fit': {
                    'slope': slope,
                    'intercept': intercept,
                    'r_squared': r_value**2,
                    'p_value': p_value
                },
                'future_points': future_points
            }
        
        return predictions
    
    def _project_sample_sizes(self) -> Dict[str, Any]:
        """Project required sample sizes for significance"""
        projections = {}
        
        # Calculate current effect sizes
        conditions = list(self.data_by_condition.keys())
        if len(conditions) >= 2:
            for i, condition1 in enumerate(conditions):
                for condition2 in conditions[i+1:]:
                    data1 = [dp.value for dp in self.data_by_condition[condition1]]
                    data2 = [dp.value for dp in self.data_by_condition[condition2]]
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        # Current effect size
                        current_effect_size = self._calculate_cohens_d(data1, data2)
                        
                        # Project sample size needed for 80% power
                        if abs(current_effect_size) > 0.1:
                            projected_n = self._calculate_sample_size_needed(
                                current_effect_size, 
                                power=0.8, 
                                alpha=self.config.significance_level
                            )
                            
                            projections[f"{condition1}_vs_{condition2}"] = {
                                'current_n1': len(data1),
                                'current_n2': len(data2),
                                'current_effect_size': current_effect_size,
                                'projected_n_per_group': projected_n,
                                'additional_needed': max(0, projected_n - min(len(data1), len(data2)))
                            }
        
        return projections
    
    def _calculate_sample_size_needed(self, effect_size: float, power: float, alpha: float) -> int:
        """Calculate sample size needed for given effect size and power"""
        # Simplified power calculation for independent t-test
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / effect_size)**2
        
        return int(np.ceil(n))
    
    def _perform_power_analysis(self) -> Dict[str, Any]:
        """Perform power analysis for current data"""
        power_analysis = {}
        
        conditions = list(self.data_by_condition.keys())
        if len(conditions) >= 2:
            for i, condition1 in enumerate(conditions):
                for condition2 in conditions[i+1:]:
                    data1 = [dp.value for dp in self.data_by_condition[condition1]]
                    data2 = [dp.value for dp in self.data_by_condition[condition2]]
                    
                    if len(data1) >= 3 and len(data2) >= 3:
                        # Calculate observed power
                        effect_size = self._calculate_cohens_d(data1, data2)
                        n1, n2 = len(data1), len(data2)
                        
                        # Statistical power calculation
                        observed_power = self._calculate_statistical_power(
                            effect_size, n1, n2, self.config.significance_level
                        )
                        
                        power_analysis[f"{condition1}_vs_{condition2}"] = {
                            'observed_power': observed_power,
                            'effect_size': effect_size,
                            'sample_size_1': n1,
                            'sample_size_2': n2,
                            'adequate_power': observed_power >= 0.8
                        }
        
        return power_analysis
    
    def _calculate_statistical_power(self, effect_size: float, n1: int, n2: int, alpha: float) -> float:
        """Calculate statistical power for independent t-test"""
        # Standard error for difference between means
        se_diff = np.sqrt(1/n1 + 1/n2)
        
        # Non-centrality parameter
        ncp = effect_size / se_diff
        
        # Critical t-value
        df = n1 + n2 - 2
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # Power calculation
        power = 1 - stats.t.cdf(t_critical, df, ncp) + stats.t.cdf(-t_critical, df, ncp)
        
        return power
    
    def _perform_sequential_analysis(self) -> Dict[str, Any]:
        """Perform sequential analysis for early stopping"""
        sequential_results = {}
        
        # Calculate sequential probability ratio
        conditions = list(self.data_by_condition.keys())
        if len(conditions) >= 2:
            condition1, condition2 = conditions[0], conditions[1]
            data1 = [dp.value for dp in self.data_by_condition[condition1]]
            data2 = [dp.value for dp in self.data_by_condition[condition2]]
            
            if len(data1) >= 5 and len(data2) >= 5:
                # Calculate cumulative test statistics
                cumulative_stats = []
                min_n = min(len(data1), len(data2))
                
                for n in range(5, min_n + 1):
                    sub_data1 = data1[:n]
                    sub_data2 = data2[:n]
                    
                    t_stat, p_value = ttest_ind(sub_data1, sub_data2)
                    cumulative_stats.append({
                        'n': n,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < self.config.significance_level
                    })
                
                sequential_results['cumulative_tests'] = cumulative_stats
                
                # Check for early stopping
                if len(cumulative_stats) >= 3:
                    recent_significant = [s['significant'] for s in cumulative_stats[-3:]]
                    if all(recent_significant):
                        sequential_results['early_stopping_recommended'] = True
                        sequential_results['early_stopping_reason'] = 'consistent_significance'
        
        return sequential_results
    
    def _perform_futility_analysis(self) -> Dict[str, Any]:
        """Perform futility analysis"""
        futility_results = {}
        
        conditions = list(self.data_by_condition.keys())
        if len(conditions) >= 2:
            condition1, condition2 = conditions[0], conditions[1]
            data1 = [dp.value for dp in self.data_by_condition[condition1]]
            data2 = [dp.value for dp in self.data_by_condition[condition2]]
            
            if len(data1) >= 10 and len(data2) >= 10:
                # Calculate conditional power
                current_effect_size = self._calculate_cohens_d(data1, data2)
                
                # Project power if current trend continues
                projected_power = self._calculate_statistical_power(
                    current_effect_size, 
                    len(data1) * 2,  # Project doubling sample size
                    len(data2) * 2,
                    self.config.significance_level
                )
                
                futility_results['conditional_power'] = projected_power
                futility_results['current_effect_size'] = current_effect_size
                
                # Futility threshold
                if projected_power < 0.2:
                    futility_results['futility_indicated'] = True
                    futility_results['futility_reason'] = 'low_conditional_power'
                elif abs(current_effect_size) < 0.1:
                    futility_results['futility_indicated'] = True
                    futility_results['futility_reason'] = 'negligible_effect_size'
                else:
                    futility_results['futility_indicated'] = False
        
        return futility_results
    
    def _is_outlier(self, data_point: RealTimeDataPoint) -> bool:
        """Check if data point is an outlier"""
        # Get recent values for the same metric
        recent_values = [dp.value for dp in self.data_by_metric[data_point.metric_name][-50:]]
        
        if len(recent_values) < 5:
            return False
        
        # Use IQR method for outlier detection
        q1 = np.percentile(recent_values, 25)
        q3 = np.percentile(recent_values, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return data_point.value < lower_bound or data_point.value > upper_bound
    
    def _update_data_quality_score(self, data_point: RealTimeDataPoint):
        """Update data quality score"""
        quality_score = 1.0
        
        # Check for missing metadata
        if not data_point.metadata:
            quality_score -= 0.1
        
        # Check for reasonable response time
        response_time = data_point.metadata.get('response_time', 0)
        if response_time < 1.0 or response_time > 60.0:  # Unreasonable response times
            quality_score -= 0.2
        
        # Check for attention check
        if 'attention_check_passed' in data_point.metadata:
            if not data_point.metadata['attention_check_passed']:
                quality_score -= 0.3
        
        # Check for engagement
        engagement = data_point.metadata.get('engagement_level', 1.0)
        if engagement < 0.5:
            quality_score -= 0.2
        
        quality_score = max(0.0, quality_score)
        self.data_quality_scores.append(quality_score)
        
        # Keep only recent scores
        if len(self.data_quality_scores) > 100:
            self.data_quality_scores = self.data_quality_scores[-100:]
    
    def _check_stopping_criteria(self) -> bool:
        """Check if stopping criteria are met"""
        if not self.config.stopping_criteria:
            return False
        
        criteria = self.config.stopping_criteria
        
        # Check significance criteria
        if 'significance_threshold' in criteria:
            # Check if recent analyses show consistent significance
            recent_results = [r for r in self.analysis_history[-5:] if r.p_value < criteria['significance_threshold']]
            if len(recent_results) >= 3:
                return True
        
        # Check effect size criteria
        if 'effect_size_threshold' in criteria:
            # Check if effect size is consistently above threshold
            recent_effect_sizes = []
            for result in self.analysis_history[-5:]:
                if result.effect_size and abs(result.effect_size) >= criteria['effect_size_threshold']:
                    recent_effect_sizes.append(result.effect_size)
            
            if len(recent_effect_sizes) >= 3:
                return True
        
        # Check maximum sample size
        if 'max_sample_size' in criteria:
            if len(self.data_buffer) >= criteria['max_sample_size']:
                return True
        
        # Check futility criteria
        if 'futility_threshold' in criteria:
            adaptive_stats = self.current_statistics.get('adaptive', {})
            futility_analysis = adaptive_stats.get('futility_analysis', {})
            if futility_analysis.get('futility_indicated', False):
                return True
        
        return False
    
    def _check_interim_analysis(self) -> bool:
        """Check if interim analysis should be performed"""
        if not self.config.interim_analysis_intervals:
            return False
        
        current_sample_size = len(self.data_buffer)
        
        # Check if we've reached the next interim analysis point
        if self.next_interim_analysis < len(self.config.interim_analysis_intervals):
            next_n = self.config.interim_analysis_intervals[self.next_interim_analysis]
            if current_sample_size >= next_n:
                self.next_interim_analysis += 1
                return True
        
        return False
    
    async def _perform_interim_analysis(self):
        """Perform interim analysis"""
        self.logger.info(f"Performing interim analysis at n={len(self.data_buffer)}")
        
        # Perform comprehensive analysis
        await self._perform_analyses()
        
        # Log interim results
        self.logger.info(f"Interim analysis completed: {len(self.analysis_history)} results")
    
    def get_current_results(self) -> Dict[str, Any]:
        """Get current analysis results"""
        return {
            'experiment_id': self.config.experiment_id,
            'timestamp': datetime.now().isoformat(),
            'is_running': self.is_running,
            'sample_size': len(self.data_buffer),
            'update_count': self.update_count,
            'statistics': self.current_statistics,
            'data_quality': {
                'mean_quality_score': np.mean(self.data_quality_scores) if self.data_quality_scores else 0.0,
                'outliers_detected': len(self.outliers_detected),
                'data_points_by_condition': {k: len(v) for k, v in self.data_by_condition.items()},
                'data_points_by_metric': {k: len(v) for k, v in self.data_by_metric.items()}
            },
            'performance': {
                'mean_analysis_time': np.mean(self.analysis_times) if self.analysis_times else 0.0,
                'max_analysis_time': np.max(self.analysis_times) if self.analysis_times else 0.0,
                'update_interval': self.config.update_interval
            },
            'stopping_criteria_met': self.stopping_criteria_met
        }
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get history of statistical analyses"""
        return [asdict(result) for result in self.analysis_history]
    
    def export_data(self) -> Dict[str, Any]:
        """Export all collected data"""
        return {
            'experiment_id': self.config.experiment_id,
            'config': asdict(self.config),
            'data_points': [asdict(dp) for dp in self.data_buffer],
            'analysis_history': self.get_analysis_history(),
            'current_statistics': self.current_statistics,
            'outliers': [asdict(outlier) for outlier in self.outliers_detected],
            'data_quality_scores': self.data_quality_scores,
            'export_timestamp': datetime.now().isoformat()
        }