"""
Tests for MUSE Platform validation framework and statistical analysis
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import List, Dict, Any

from muse.validation.mathematical_validation_framework import MathematicalValidationFramework
from muse.validation.real_time_statistical_analysis import RealTimeStatisticalAnalysis
from muse.validation.metrics_calculator import MetricsCalculator
from muse.validation.automated_data_collection_pipeline import AutomatedDataCollectionPipeline
from muse.validation.participant_recruitment_system import ParticipantRecruitmentSystem


class TestMathematicalValidationFramework:
    """Test suite for MathematicalValidationFramework."""
    
    def test_initialization(self, validation_framework):
        """Test validation framework initialization."""
        framework = validation_framework
        assert hasattr(framework, 'statistical_threshold')
        assert hasattr(framework, 'confidence_level')
        assert framework.statistical_threshold > 0
        assert 0 < framework.confidence_level < 1
    
    @pytest.mark.asyncio
    async def test_validate_mathematical_coherence(self, validation_framework, sample_creative_session_data):
        """Test mathematical coherence validation."""
        framework = validation_framework
        
        session_data = sample_creative_session_data
        coherence_result = await framework.validate_mathematical_coherence(session_data)
        
        assert isinstance(coherence_result, dict)
        assert "coherence_score" in coherence_result
        assert "mathematical_relationships" in coherence_result
        assert "consistency_metrics" in coherence_result
        assert "validation_details" in coherence_result
        
        # Test coherence score properties
        coherence_score = coherence_result["coherence_score"]
        assert isinstance(coherence_score, float)
        assert 0.0 <= coherence_score <= 1.0
        
        # Test mathematical relationships
        relationships = coherence_result["mathematical_relationships"]
        assert isinstance(relationships, dict)
        assert "golden_ratio_presence" in relationships
        assert "fibonacci_patterns" in relationships
        assert "frequency_harmony" in relationships
    
    @pytest.mark.asyncio
    async def test_analyze_creativity_metrics(self, validation_framework, sample_creative_session_data):
        """Test creativity metrics analysis."""
        framework = validation_framework
        
        session_data = sample_creative_session_data
        creativity_result = await framework.analyze_creativity_metrics(session_data)
        
        assert isinstance(creativity_result, dict)
        assert "creativity_score" in creativity_result
        assert "originality_score" in creativity_result
        assert "complexity_score" in creativity_result
        assert "fluency_score" in creativity_result
        assert "elaboration_score" in creativity_result
        
        # Test score properties
        for score_key in ["creativity_score", "originality_score", "complexity_score", "fluency_score", "elaboration_score"]:
            score = creativity_result[score_key]
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_validate_user_engagement(self, validation_framework, sample_user_data):
        """Test user engagement validation."""
        framework = validation_framework
        
        user_data = sample_user_data
        engagement_result = await framework.validate_user_engagement(user_data)
        
        assert isinstance(engagement_result, dict)
        assert "engagement_score" in engagement_result
        assert "interaction_quality" in engagement_result
        assert "session_depth" in engagement_result
        assert "retention_indicators" in engagement_result
        
        # Test engagement score
        engagement_score = engagement_result["engagement_score"]
        assert isinstance(engagement_score, float)
        assert 0.0 <= engagement_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_cross_modal_validation(self, validation_framework, sample_creative_session_data):
        """Test cross-modal validation between different analysis types."""
        framework = validation_framework
        
        session_data = sample_creative_session_data
        cross_modal_result = await framework.cross_modal_validation(session_data)
        
        assert isinstance(cross_modal_result, dict)
        assert "modal_consistency" in cross_modal_result
        assert "geometry_frequency_alignment" in cross_modal_result
        assert "semantic_frequency_alignment" in cross_modal_result
        assert "geometry_semantic_alignment" in cross_modal_result
        assert "overall_coherence" in cross_modal_result
        
        # Test alignment scores
        for alignment_key in ["geometry_frequency_alignment", "semantic_frequency_alignment", "geometry_semantic_alignment"]:
            alignment_score = cross_modal_result[alignment_key]
            assert isinstance(alignment_score, float)
            assert 0.0 <= alignment_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_longitudinal_analysis(self, validation_framework):
        """Test longitudinal analysis across multiple sessions."""
        framework = validation_framework
        
        # Create multiple session data points
        session_data = []
        for i in range(5):
            session = {
                "session_id": f"session_{i}",
                "timestamp": datetime.now() - timedelta(days=i),
                "creativity_score": 0.7 + 0.05 * i,
                "coherence_score": 0.8 + 0.02 * i,
                "user_satisfaction": 0.75 + 0.03 * i
            }
            session_data.append(session)
        
        longitudinal_result = await framework.longitudinal_analysis(session_data)
        
        assert isinstance(longitudinal_result, dict)
        assert "trend_analysis" in longitudinal_result
        assert "improvement_rate" in longitudinal_result
        assert "stability_metrics" in longitudinal_result
        assert "prediction_confidence" in longitudinal_result
        
        # Test trend analysis
        trend_analysis = longitudinal_result["trend_analysis"]
        assert isinstance(trend_analysis, dict)
        assert "creativity_trend" in trend_analysis
        assert "coherence_trend" in trend_analysis
        assert "satisfaction_trend" in trend_analysis
    
    @pytest.mark.asyncio
    async def test_statistical_significance_testing(self, validation_framework):
        """Test statistical significance testing."""
        framework = validation_framework
        
        # Create control and treatment groups
        control_group = [0.5, 0.6, 0.7, 0.8, 0.9, 0.6, 0.7, 0.8, 0.5, 0.6]
        treatment_group = [0.7, 0.8, 0.9, 1.0, 1.1, 0.8, 0.9, 1.0, 0.7, 0.8]
        
        significance_result = await framework.statistical_significance_testing(
            control_group, treatment_group, "creativity_score"
        )
        
        assert isinstance(significance_result, dict)
        assert "p_value" in significance_result
        assert "effect_size" in significance_result
        assert "confidence_interval" in significance_result
        assert "is_significant" in significance_result
        assert "statistical_power" in significance_result
        
        # Test p-value properties
        p_value = significance_result["p_value"]
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0
        
        # Test effect size
        effect_size = significance_result["effect_size"]
        assert isinstance(effect_size, float)
        
        # Test confidence interval
        confidence_interval = significance_result["confidence_interval"]
        assert isinstance(confidence_interval, tuple)
        assert len(confidence_interval) == 2
    
    @pytest.mark.asyncio
    async def test_validation_with_empty_data(self, validation_framework):
        """Test validation with empty or invalid data."""
        framework = validation_framework
        
        # Test with empty session data
        empty_session = {}
        result = await framework.validate_mathematical_coherence(empty_session)
        assert isinstance(result, dict)
        assert "error" in result or "coherence_score" in result
        
        # Test with invalid creativity data
        invalid_session = {"invalid_key": "invalid_value"}
        result = await framework.analyze_creativity_metrics(invalid_session)
        assert isinstance(result, dict)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_performance(self, validation_framework, sample_creative_session_data, benchmark_data):
        """Test validation framework performance."""
        framework = validation_framework
        
        import time
        
        # Test coherence validation performance
        start_time = time.time()
        await framework.validate_mathematical_coherence(sample_creative_session_data)
        duration = time.time() - start_time
        
        # Should complete within reasonable time
        assert duration < 1.0  # 1 second threshold
        
        # Test creativity metrics performance
        start_time = time.time()
        await framework.analyze_creativity_metrics(sample_creative_session_data)
        duration = time.time() - start_time
        
        assert duration < 1.0


class TestRealTimeStatisticalAnalysis:
    """Test suite for RealTimeStatisticalAnalysis."""
    
    def test_initialization(self, statistical_analysis):
        """Test statistical analysis initialization."""
        analysis = statistical_analysis
        assert hasattr(analysis, 'window_size')
        assert hasattr(analysis, 'update_frequency')
        assert analysis.window_size > 0
        assert analysis.update_frequency > 0
    
    @pytest.mark.asyncio
    async def test_add_data_point(self, statistical_analysis):
        """Test adding data points to real-time analysis."""
        analysis = statistical_analysis
        
        # Add data points
        await analysis.add_data_point("creativity_score", 0.8, datetime.now())
        await analysis.add_data_point("creativity_score", 0.9, datetime.now())
        await analysis.add_data_point("coherence_score", 0.7, datetime.now())
        
        # Verify data was added
        current_stats = await analysis.get_current_statistics()
        assert isinstance(current_stats, dict)
        assert "creativity_score" in current_stats
        assert "coherence_score" in current_stats
    
    @pytest.mark.asyncio
    async def test_calculate_running_statistics(self, statistical_analysis):
        """Test running statistics calculation."""
        analysis = statistical_analysis
        
        # Add multiple data points
        data_points = [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        for i, value in enumerate(data_points):
            timestamp = datetime.now() - timedelta(minutes=i)
            await analysis.add_data_point("test_metric", value, timestamp)
        
        # Calculate running statistics
        running_stats = await analysis.calculate_running_statistics("test_metric")
        
        assert isinstance(running_stats, dict)
        assert "mean" in running_stats
        assert "std" in running_stats
        assert "min" in running_stats
        assert "max" in running_stats
        assert "trend" in running_stats
        assert "variance" in running_stats
        
        # Test statistical properties
        assert running_stats["mean"] == np.mean(data_points)
        assert running_stats["std"] == np.std(data_points)
        assert running_stats["min"] == np.min(data_points)
        assert running_stats["max"] == np.max(data_points)
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, statistical_analysis):
        """Test anomaly detection in real-time data."""
        analysis = statistical_analysis
        
        # Add normal data points
        normal_data = [0.8, 0.82, 0.79, 0.81, 0.83, 0.78, 0.80, 0.82, 0.79, 0.81]
        for i, value in enumerate(normal_data):
            timestamp = datetime.now() - timedelta(minutes=i)
            await analysis.add_data_point("normal_metric", value, timestamp)
        
        # Add anomalous data point
        anomaly_timestamp = datetime.now()
        await analysis.add_data_point("normal_metric", 0.3, anomaly_timestamp)  # Clear anomaly
        
        # Detect anomalies
        anomalies = await analysis.detect_anomalies("normal_metric")
        
        assert isinstance(anomalies, list)
        assert len(anomalies) > 0
        
        # Check anomaly properties
        anomaly = anomalies[0]
        assert "timestamp" in anomaly
        assert "value" in anomaly
        assert "anomaly_score" in anomaly
        assert "threshold" in anomaly
        
        # The anomalous value should be detected
        assert anomaly["value"] == 0.3
    
    @pytest.mark.asyncio
    async def test_trend_analysis(self, statistical_analysis):
        """Test trend analysis in real-time data."""
        analysis = statistical_analysis
        
        # Add trending data (increasing trend)
        trending_data = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        for i, value in enumerate(trending_data):
            timestamp = datetime.now() - timedelta(minutes=len(trending_data) - i)
            await analysis.add_data_point("trending_metric", value, timestamp)
        
        # Analyze trend
        trend_analysis = await analysis.analyze_trend("trending_metric")
        
        assert isinstance(trend_analysis, dict)
        assert "trend_direction" in trend_analysis
        assert "trend_strength" in trend_analysis
        assert "trend_significance" in trend_analysis
        assert "slope" in trend_analysis
        assert "correlation" in trend_analysis
        
        # Should detect increasing trend
        assert trend_analysis["trend_direction"] == "increasing"
        assert trend_analysis["trend_strength"] > 0.8
        assert trend_analysis["slope"] > 0
    
    @pytest.mark.asyncio
    async def test_correlation_analysis(self, statistical_analysis):
        """Test correlation analysis between metrics."""
        analysis = statistical_analysis
        
        # Add correlated data
        base_data = [0.5, 0.6, 0.7, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
        correlated_data = [x + 0.1 + np.random.normal(0, 0.05) for x in base_data]
        
        for i, (val1, val2) in enumerate(zip(base_data, correlated_data)):
            timestamp = datetime.now() - timedelta(minutes=i)
            await analysis.add_data_point("metric_1", val1, timestamp)
            await analysis.add_data_point("metric_2", val2, timestamp)
        
        # Calculate correlation
        correlation_result = await analysis.calculate_correlation("metric_1", "metric_2")
        
        assert isinstance(correlation_result, dict)
        assert "correlation_coefficient" in correlation_result
        assert "p_value" in correlation_result
        assert "is_significant" in correlation_result
        assert "strength" in correlation_result
        
        # Should show strong positive correlation
        correlation_coef = correlation_result["correlation_coefficient"]
        assert 0.7 <= correlation_coef <= 1.0
    
    @pytest.mark.asyncio
    async def test_window_sliding(self, statistical_analysis):
        """Test sliding window behavior."""
        analysis = statistical_analysis
        
        # Set small window size for testing
        analysis.window_size = 5
        
        # Add more data points than window size
        data_points = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for i, value in enumerate(data_points):
            timestamp = datetime.now() - timedelta(minutes=i)
            await analysis.add_data_point("windowed_metric", value, timestamp)
        
        # Get current statistics
        current_stats = await analysis.get_current_statistics()
        windowed_stats = current_stats.get("windowed_metric", {})
        
        # Should only include last 5 data points
        assert "data_points" in windowed_stats
        data_points_count = len(windowed_stats["data_points"])
        assert data_points_count <= analysis.window_size
    
    @pytest.mark.asyncio
    async def test_real_time_alerts(self, statistical_analysis):
        """Test real-time alerting system."""
        analysis = statistical_analysis
        
        # Set up alert thresholds
        await analysis.set_alert_threshold("alert_metric", "high", 0.9)
        await analysis.set_alert_threshold("alert_metric", "low", 0.1)
        
        # Add data points that should trigger alerts
        await analysis.add_data_point("alert_metric", 0.05, datetime.now())  # Low alert
        await analysis.add_data_point("alert_metric", 0.95, datetime.now())  # High alert
        
        # Check for alerts
        alerts = await analysis.get_active_alerts()
        
        assert isinstance(alerts, list)
        assert len(alerts) >= 2
        
        # Check alert properties
        for alert in alerts:
            assert "metric_name" in alert
            assert "alert_type" in alert
            assert "value" in alert
            assert "threshold" in alert
            assert "timestamp" in alert
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_real_time_performance(self, statistical_analysis):
        """Test real-time analysis performance."""
        analysis = statistical_analysis
        
        import time
        
        # Test performance with many data points
        start_time = time.time()
        
        for i in range(100):
            await analysis.add_data_point("performance_metric", 0.5 + 0.01 * i, datetime.now())
        
        duration = time.time() - start_time
        assert duration < 1.0  # Should handle 100 data points in under 1 second
        
        # Test statistics calculation performance
        start_time = time.time()
        await analysis.calculate_running_statistics("performance_metric")
        duration = time.time() - start_time
        
        assert duration < 0.1  # Statistics calculation should be fast


class TestMetricsCalculator:
    """Test suite for MetricsCalculator."""
    
    def test_initialization(self):
        """Test metrics calculator initialization."""
        calculator = MetricsCalculator()
        assert hasattr(calculator, 'metric_definitions')
        assert hasattr(calculator, 'aggregation_methods')
    
    def test_calculate_creativity_score(self):
        """Test creativity score calculation."""
        calculator = MetricsCalculator()
        
        # Test with sample creative output
        creative_output = {
            "text": "The golden spiral dances through mathematical dimensions of infinite beauty",
            "originality_indicators": ["unique_metaphors", "novel_connections"],
            "complexity_measures": {"sentence_structure": 0.8, "concept_depth": 0.9},
            "fluency_metrics": {"word_count": 12, "idea_density": 0.7}
        }
        
        creativity_score = calculator.calculate_creativity_score(creative_output)
        
        assert isinstance(creativity_score, float)
        assert 0.0 <= creativity_score <= 1.0
    
    def test_calculate_coherence_score(self):
        """Test coherence score calculation."""
        calculator = MetricsCalculator()
        
        # Test with coherent content
        coherent_content = {
            "text": "The fibonacci sequence creates beautiful spiral patterns in nature",
            "mathematical_elements": ["fibonacci", "spiral", "patterns"],
            "logical_flow": 0.9,
            "consistency_score": 0.8
        }
        
        coherence_score = calculator.calculate_coherence_score(coherent_content)
        
        assert isinstance(coherence_score, float)
        assert 0.0 <= coherence_score <= 1.0
        assert coherence_score > 0.5  # Should be high for coherent content
    
    def test_calculate_engagement_metrics(self):
        """Test engagement metrics calculation."""
        calculator = MetricsCalculator()
        
        # Test with engagement data
        engagement_data = {
            "session_duration": 1800,  # 30 minutes
            "interaction_count": 15,
            "user_actions": ["explore", "create", "share", "iterate"],
            "satisfaction_rating": 4.5,
            "return_likelihood": 0.8
        }
        
        engagement_metrics = calculator.calculate_engagement_metrics(engagement_data)
        
        assert isinstance(engagement_metrics, dict)
        assert "engagement_score" in engagement_metrics
        assert "interaction_quality" in engagement_metrics
        assert "session_depth" in engagement_metrics
        
        # Test score properties
        engagement_score = engagement_metrics["engagement_score"]
        assert isinstance(engagement_score, float)
        assert 0.0 <= engagement_score <= 1.0
    
    def test_calculate_learning_progression(self):
        """Test learning progression calculation."""
        calculator = MetricsCalculator()
        
        # Test with learning data
        learning_data = {
            "initial_assessment": 0.3,
            "current_assessment": 0.8,
            "session_count": 10,
            "skill_improvements": ["creativity", "mathematical_thinking", "pattern_recognition"],
            "knowledge_areas": ["sacred_geometry", "frequency_analysis", "semantic_understanding"]
        }
        
        learning_progression = calculator.calculate_learning_progression(learning_data)
        
        assert isinstance(learning_progression, dict)
        assert "progression_rate" in learning_progression
        assert "skill_development" in learning_progression
        assert "knowledge_acquisition" in learning_progression
        
        # Test progression rate
        progression_rate = learning_progression["progression_rate"]
        assert isinstance(progression_rate, float)
        assert progression_rate > 0  # Should show positive progression
    
    def test_calculate_resonance_metrics(self):
        """Test resonance metrics calculation."""
        calculator = MetricsCalculator()
        
        # Test with resonance data
        resonance_data = {
            "frequency_resonance": 0.85,
            "semantic_resonance": 0.78,
            "geometry_resonance": 0.92,
            "user_resonance": 0.88,
            "community_resonance": 0.76
        }
        
        resonance_metrics = calculator.calculate_resonance_metrics(resonance_data)
        
        assert isinstance(resonance_metrics, dict)
        assert "overall_resonance" in resonance_metrics
        assert "resonance_distribution" in resonance_metrics
        assert "resonance_stability" in resonance_metrics
        
        # Test overall resonance
        overall_resonance = resonance_metrics["overall_resonance"]
        assert isinstance(overall_resonance, float)
        assert 0.0 <= overall_resonance <= 1.0
    
    def test_aggregate_metrics(self):
        """Test metrics aggregation."""
        calculator = MetricsCalculator()
        
        # Test with multiple metric sets
        metrics_data = [
            {"creativity": 0.8, "coherence": 0.9, "engagement": 0.7},
            {"creativity": 0.85, "coherence": 0.88, "engagement": 0.75},
            {"creativity": 0.78, "coherence": 0.92, "engagement": 0.72},
            {"creativity": 0.82, "coherence": 0.87, "engagement": 0.78}
        ]
        
        aggregated_metrics = calculator.aggregate_metrics(metrics_data)
        
        assert isinstance(aggregated_metrics, dict)
        assert "mean_values" in aggregated_metrics
        assert "std_values" in aggregated_metrics
        assert "trend_analysis" in aggregated_metrics
        
        # Test mean values
        mean_values = aggregated_metrics["mean_values"]
        assert "creativity" in mean_values
        assert "coherence" in mean_values
        assert "engagement" in mean_values
        
        # Verify mean calculation
        expected_creativity_mean = np.mean([0.8, 0.85, 0.78, 0.82])
        assert abs(mean_values["creativity"] - expected_creativity_mean) < 1e-10
    
    def test_calculate_statistical_significance(self):
        """Test statistical significance calculation."""
        calculator = MetricsCalculator()
        
        # Test with sample data
        group_a = [0.5, 0.6, 0.7, 0.8, 0.9]
        group_b = [0.7, 0.8, 0.9, 1.0, 1.1]
        
        significance_result = calculator.calculate_statistical_significance(group_a, group_b)
        
        assert isinstance(significance_result, dict)
        assert "p_value" in significance_result
        assert "effect_size" in significance_result
        assert "is_significant" in significance_result
        
        # Test p-value
        p_value = significance_result["p_value"]
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0
    
    def test_metric_validation(self):
        """Test metric validation."""
        calculator = MetricsCalculator()
        
        # Test valid metrics
        valid_metrics = {"creativity": 0.8, "coherence": 0.9, "engagement": 0.7}
        validation_result = calculator.validate_metrics(valid_metrics)
        assert validation_result["is_valid"] is True
        assert validation_result["errors"] == []
        
        # Test invalid metrics
        invalid_metrics = {"creativity": 1.5, "coherence": -0.1, "engagement": "invalid"}
        validation_result = calculator.validate_metrics(invalid_metrics)
        assert validation_result["is_valid"] is False
        assert len(validation_result["errors"]) > 0


class TestAutomatedDataCollectionPipeline:
    """Test suite for AutomatedDataCollectionPipeline."""
    
    def test_initialization(self):
        """Test data collection pipeline initialization."""
        pipeline = AutomatedDataCollectionPipeline()
        assert hasattr(pipeline, 'collection_config')
        assert hasattr(pipeline, 'data_sources')
        assert hasattr(pipeline, 'processing_pipeline')
    
    @pytest.mark.asyncio
    async def test_collect_user_interaction_data(self):
        """Test user interaction data collection."""
        pipeline = AutomatedDataCollectionPipeline()
        
        # Mock user interaction
        interaction_data = {
            "user_id": "test_user",
            "session_id": "test_session",
            "action": "creative_exploration",
            "timestamp": datetime.now(),
            "duration": 120,
            "input_data": {"text": "Explore mathematical patterns"},
            "output_data": {"generated_content": "Beautiful spiral patterns"}
        }
        
        collection_result = await pipeline.collect_user_interaction_data(interaction_data)
        
        assert isinstance(collection_result, dict)
        assert "status" in collection_result
        assert "data_id" in collection_result
        assert collection_result["status"] == "collected"
    
    @pytest.mark.asyncio
    async def test_collect_system_performance_data(self):
        """Test system performance data collection."""
        pipeline = AutomatedDataCollectionPipeline()
        
        # Mock performance data
        performance_data = {
            "timestamp": datetime.now(),
            "response_time": 0.05,
            "memory_usage": 150.5,
            "cpu_usage": 23.8,
            "active_sessions": 15,
            "error_rate": 0.001
        }
        
        collection_result = await pipeline.collect_system_performance_data(performance_data)
        
        assert isinstance(collection_result, dict)
        assert "status" in collection_result
        assert collection_result["status"] == "collected"
    
    @pytest.mark.asyncio
    async def test_collect_validation_metrics(self):
        """Test validation metrics collection."""
        pipeline = AutomatedDataCollectionPipeline()
        
        # Mock validation metrics
        validation_data = {
            "session_id": "validation_session",
            "creativity_score": 0.85,
            "coherence_score": 0.92,
            "engagement_score": 0.78,
            "resonance_score": 0.88,
            "user_satisfaction": 4.2,
            "timestamp": datetime.now()
        }
        
        collection_result = await pipeline.collect_validation_metrics(validation_data)
        
        assert isinstance(collection_result, dict)
        assert "status" in collection_result
        assert collection_result["status"] == "collected"
    
    @pytest.mark.asyncio
    async def test_data_preprocessing(self):
        """Test data preprocessing pipeline."""
        pipeline = AutomatedDataCollectionPipeline()
        
        # Mock raw data
        raw_data = [
            {"metric": "creativity", "value": "0.8", "timestamp": "2024-01-01T10:00:00"},
            {"metric": "coherence", "value": "0.9", "timestamp": "2024-01-01T10:01:00"},
            {"metric": "engagement", "value": "0.7", "timestamp": "2024-01-01T10:02:00"}
        ]
        
        processed_data = await pipeline.preprocess_data(raw_data)
        
        assert isinstance(processed_data, list)
        assert len(processed_data) == 3
        
        # Check data type conversions
        for item in processed_data:
            assert isinstance(item["value"], float)
            assert isinstance(item["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_data_quality_validation(self):
        """Test data quality validation."""
        pipeline = AutomatedDataCollectionPipeline()
        
        # Test valid data
        valid_data = {
            "creativity_score": 0.85,
            "coherence_score": 0.92,
            "timestamp": datetime.now(),
            "user_id": "valid_user"
        }
        
        quality_result = await pipeline.validate_data_quality(valid_data)
        assert quality_result["is_valid"] is True
        assert quality_result["quality_score"] > 0.8
        
        # Test invalid data
        invalid_data = {
            "creativity_score": 1.5,  # Out of range
            "coherence_score": -0.1,  # Out of range
            "timestamp": "invalid_date",
            "user_id": ""  # Empty
        }
        
        quality_result = await pipeline.validate_data_quality(invalid_data)
        assert quality_result["is_valid"] is False
        assert len(quality_result["errors"]) > 0
    
    @pytest.mark.asyncio
    async def test_batch_processing(self):
        """Test batch data processing."""
        pipeline = AutomatedDataCollectionPipeline()
        
        # Create batch data
        batch_data = []
        for i in range(100):
            data_point = {
                "user_id": f"user_{i}",
                "session_id": f"session_{i}",
                "creativity_score": 0.5 + 0.005 * i,
                "timestamp": datetime.now() - timedelta(minutes=i)
            }
            batch_data.append(data_point)
        
        batch_result = await pipeline.process_batch(batch_data)
        
        assert isinstance(batch_result, dict)
        assert "processed_count" in batch_result
        assert "failed_count" in batch_result
        assert "processing_time" in batch_result
        assert batch_result["processed_count"] == 100
        assert batch_result["failed_count"] == 0
    
    @pytest.mark.asyncio
    async def test_data_export(self):
        """Test data export functionality."""
        pipeline = AutomatedDataCollectionPipeline()
        
        # Mock collected data
        export_data = {
            "user_interactions": [
                {"user_id": "user1", "action": "explore", "timestamp": datetime.now()},
                {"user_id": "user2", "action": "create", "timestamp": datetime.now()}
            ],
            "validation_metrics": [
                {"session_id": "s1", "creativity_score": 0.8, "timestamp": datetime.now()},
                {"session_id": "s2", "creativity_score": 0.9, "timestamp": datetime.now()}
            ]
        }
        
        export_result = await pipeline.export_data(export_data, format="json")
        
        assert isinstance(export_result, dict)
        assert "status" in export_result
        assert "export_path" in export_result
        assert export_result["status"] == "exported"


class TestParticipantRecruitmentSystem:
    """Test suite for ParticipantRecruitmentSystem."""
    
    def test_initialization(self):
        """Test participant recruitment system initialization."""
        recruitment_system = ParticipantRecruitmentSystem()
        assert hasattr(recruitment_system, 'recruitment_criteria')
        assert hasattr(recruitment_system, 'participant_pool')
        assert hasattr(recruitment_system, 'study_protocols')
    
    @pytest.mark.asyncio
    async def test_recruit_participants(self):
        """Test participant recruitment."""
        recruitment_system = ParticipantRecruitmentSystem()
        
        # Define recruitment criteria
        criteria = {
            "age_range": (18, 65),
            "interests": ["mathematics", "art", "creativity"],
            "experience_level": "beginner_to_intermediate",
            "availability": "flexible",
            "target_count": 50
        }
        
        recruitment_result = await recruitment_system.recruit_participants(criteria)
        
        assert isinstance(recruitment_result, dict)
        assert "recruited_count" in recruitment_result
        assert "participant_profiles" in recruitment_result
        assert "recruitment_success_rate" in recruitment_result
    
    @pytest.mark.asyncio
    async def test_screen_participants(self):
        """Test participant screening."""
        recruitment_system = ParticipantRecruitmentSystem()
        
        # Mock participant data
        participant_data = {
            "participant_id": "p001",
            "age": 25,
            "interests": ["mathematics", "music", "art"],
            "experience": "intermediate",
            "availability": "weekends",
            "screening_responses": {
                "creativity_self_assessment": 7,
                "math_comfort_level": 8,
                "technology_proficiency": 9
            }
        }
        
        screening_result = await recruitment_system.screen_participant(participant_data)
        
        assert isinstance(screening_result, dict)
        assert "screening_score" in screening_result
        assert "is_eligible" in screening_result
        assert "eligibility_factors" in screening_result
        
        # Test screening score
        screening_score = screening_result["screening_score"]
        assert isinstance(screening_score, float)
        assert 0.0 <= screening_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_assign_to_study_groups(self):
        """Test study group assignment."""
        recruitment_system = ParticipantRecruitmentSystem()
        
        # Mock participants
        participants = [
            {"participant_id": f"p{i:03d}", "screening_score": 0.7 + 0.01 * i}
            for i in range(20)
        ]
        
        assignment_result = await recruitment_system.assign_to_study_groups(
            participants, 
            groups=["control", "treatment_a", "treatment_b"],
            balancing_factors=["screening_score"]
        )
        
        assert isinstance(assignment_result, dict)
        assert "group_assignments" in assignment_result
        assert "group_balance" in assignment_result
        assert "assignment_quality" in assignment_result
        
        # Test group assignments
        group_assignments = assignment_result["group_assignments"]
        assert len(group_assignments) == 20
        assert all("group" in assignment for assignment in group_assignments)
    
    @pytest.mark.asyncio
    async def test_track_participant_engagement(self):
        """Test participant engagement tracking."""
        recruitment_system = ParticipantRecruitmentSystem()
        
        # Mock engagement data
        engagement_data = {
            "participant_id": "p001",
            "session_count": 5,
            "total_time": 3600,  # 1 hour
            "completion_rate": 0.8,
            "interaction_quality": 0.85,
            "feedback_scores": [4, 5, 4, 5, 4]
        }
        
        tracking_result = await recruitment_system.track_participant_engagement(engagement_data)
        
        assert isinstance(tracking_result, dict)
        assert "engagement_score" in tracking_result
        assert "retention_prediction" in tracking_result
        assert "intervention_recommendations" in tracking_result
        
        # Test engagement score
        engagement_score = tracking_result["engagement_score"]
        assert isinstance(engagement_score, float)
        assert 0.0 <= engagement_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_generate_study_report(self):
        """Test study report generation."""
        recruitment_system = ParticipantRecruitmentSystem()
        
        # Mock study data
        study_data = {
            "study_id": "muse_validation_study_001",
            "participants": 100,
            "duration_days": 30,
            "completion_rate": 0.85,
            "primary_outcomes": {
                "creativity_improvement": 0.23,
                "engagement_increase": 0.31,
                "satisfaction_score": 4.2
            },
            "statistical_results": {
                "p_values": {"creativity": 0.001, "engagement": 0.005},
                "effect_sizes": {"creativity": 0.8, "engagement": 0.6}
            }
        }
        
        report_result = await recruitment_system.generate_study_report(study_data)
        
        assert isinstance(report_result, dict)
        assert "executive_summary" in report_result
        assert "statistical_analysis" in report_result
        assert "recommendations" in report_result
        assert "appendices" in report_result


# Integration tests for validation framework
class TestValidationFrameworkIntegration:
    """Integration tests for the complete validation framework."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_validation_pipeline(self, validation_framework, statistical_analysis, sample_creative_session_data):
        """Test end-to-end validation pipeline."""
        framework = validation_framework
        analysis = statistical_analysis
        
        # 1. Validate mathematical coherence
        coherence_result = await framework.validate_mathematical_coherence(sample_creative_session_data)
        assert "coherence_score" in coherence_result
        
        # 2. Analyze creativity metrics
        creativity_result = await framework.analyze_creativity_metrics(sample_creative_session_data)
        assert "creativity_score" in creativity_result
        
        # 3. Add results to real-time analysis
        await analysis.add_data_point("coherence", coherence_result["coherence_score"], datetime.now())
        await analysis.add_data_point("creativity", creativity_result["creativity_score"], datetime.now())
        
        # 4. Calculate running statistics
        coherence_stats = await analysis.calculate_running_statistics("coherence")
        creativity_stats = await analysis.calculate_running_statistics("creativity")
        
        assert "mean" in coherence_stats
        assert "mean" in creativity_stats
        
        # 5. Cross-modal validation
        cross_modal_result = await framework.cross_modal_validation(sample_creative_session_data)
        assert "overall_coherence" in cross_modal_result
        
        # The complete pipeline should work seamlessly
        assert coherence_result["coherence_score"] > 0
        assert creativity_result["creativity_score"] > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_validation_with_data_collection(self, validation_framework):
        """Test validation framework with automated data collection."""
        framework = validation_framework
        pipeline = AutomatedDataCollectionPipeline()
        
        # Simulate validation session
        session_data = {
            "session_id": "integration_test_session",
            "user_id": "test_user",
            "generated_content": {
                "text": "The golden ratio creates infinite beauty in spiral galaxies",
                "frequency_signature": [440.0, 554.37, 659.25, 783.99],
                "semantic_vector": [0.1, 0.3, 0.7, 0.2, 0.9]
            },
            "validation_metrics": {
                "creativity_score": 0.85,
                "coherence_score": 0.92
            }
        }
        
        # 1. Validate the session
        validation_result = await framework.validate_mathematical_coherence(session_data)
        
        # 2. Collect validation data
        collection_result = await pipeline.collect_validation_metrics({
            "session_id": session_data["session_id"],
            "creativity_score": validation_result.get("coherence_score", 0.8),
            "coherence_score": validation_result.get("coherence_score", 0.9),
            "timestamp": datetime.now()
        })
        
        assert validation_result["coherence_score"] > 0
        assert collection_result["status"] == "collected"
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_longitudinal_validation_study(self, validation_framework, statistical_analysis):
        """Test longitudinal validation study simulation."""
        framework = validation_framework
        analysis = statistical_analysis
        
        # Simulate 30-day study with daily sessions
        study_data = []
        
        for day in range(30):
            # Simulate daily session with gradual improvement
            base_creativity = 0.5 + (day * 0.01)  # Gradual improvement
            base_coherence = 0.6 + (day * 0.008)
            
            session_data = {
                "session_id": f"study_session_{day}",
                "day": day,
                "user_id": "study_participant",
                "generated_content": {
                    "text": f"Day {day}: Exploring mathematical beauty in nature",
                    "creativity_indicators": ["metaphor", "connection", "insight"]
                },
                "validation_metrics": {
                    "creativity_score": base_creativity + np.random.normal(0, 0.05),
                    "coherence_score": base_coherence + np.random.normal(0, 0.03)
                }
            }
            
            # Validate session
            validation_result = await framework.analyze_creativity_metrics(session_data)
            
            # Add to real-time analysis
            timestamp = datetime.now() - timedelta(days=30-day)
            await analysis.add_data_point("creativity", validation_result["creativity_score"], timestamp)
            await analysis.add_data_point("coherence", validation_result.get("coherence_score", base_coherence), timestamp)
            
            study_data.append({
                "day": day,
                "creativity_score": validation_result["creativity_score"],
                "coherence_score": validation_result.get("coherence_score", base_coherence),
                "timestamp": timestamp
            })
        
        # Analyze longitudinal trends
        longitudinal_result = await framework.longitudinal_analysis(study_data)
        
        assert "trend_analysis" in longitudinal_result
        assert "improvement_rate" in longitudinal_result
        
        # Should show improvement over time
        trend_analysis = longitudinal_result["trend_analysis"]
        assert trend_analysis["creativity_trend"] == "increasing"
        assert longitudinal_result["improvement_rate"] > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_user_validation_comparison(self, validation_framework):
        """Test multi-user validation comparison."""
        framework = validation_framework
        
        # Create multiple user sessions
        user_sessions = []
        for user_id in ["user_a", "user_b", "user_c", "user_d", "user_e"]:
            session_data = {
                "session_id": f"session_{user_id}",
                "user_id": user_id,
                "generated_content": {
                    "text": f"User {user_id} explores sacred geometry patterns",
                    "creativity_level": 0.7 + hash(user_id) % 10 / 20,  # Varied creativity
                    "coherence_level": 0.8 + hash(user_id) % 8 / 25
                }
            }
            user_sessions.append(session_data)
        
        # Validate all sessions
        validation_results = []
        for session in user_sessions:
            creativity_result = await framework.analyze_creativity_metrics(session)
            coherence_result = await framework.validate_mathematical_coherence(session)
            
            validation_results.append({
                "user_id": session["user_id"],
                "creativity_score": creativity_result["creativity_score"],
                "coherence_score": coherence_result["coherence_score"]
            })
        
        # Compare results
        creativity_scores = [r["creativity_score"] for r in validation_results]
        coherence_scores = [r["coherence_score"] for r in validation_results]
        
        # Should have variation between users
        assert len(set(creativity_scores)) > 1  # Different scores
        assert len(set(coherence_scores)) > 1
        
        # All scores should be valid
        assert all(0.0 <= score <= 1.0 for score in creativity_scores)
        assert all(0.0 <= score <= 1.0 for score in coherence_scores)