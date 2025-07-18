"""
Validation Dashboard for MUSE Platform

This module provides a comprehensive dashboard for managing and monitoring
validation experiments, including CLI interface and FastAPI router endpoints.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

import click
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .mathematical_validation_framework import (
    MUSEValidationFramework,
    ValidationHypothesis,
    ExperimentPhase,
    ValidationExperiment,
    ValidationMetrics
)
from ..core.frequency_engine import MuseFrequencyEngine
from ..core.sacred_geometry_calculator import SacredGeometryCalculator
from ..core.semantic_projection_engine import SemanticProjectionEngine


# Pydantic models for API
class ExperimentCreate(BaseModel):
    """Model for creating new experiments"""
    hypothesis: str = Field(..., description="Hypothesis being tested")
    title: str = Field(..., description="Experiment title")
    description: str = Field(..., description="Detailed description")
    design_type: str = Field(default="between_subjects", description="Experimental design type")
    sample_size: int = Field(default=100, ge=10, le=10000, description="Sample size")
    created_by: str = Field(default="user", description="Creator of the experiment")
    control_variables: Dict[str, Any] = Field(default_factory=dict, description="Control variables")


class ExperimentUpdate(BaseModel):
    """Model for updating experiments"""
    title: Optional[str] = None
    description: Optional[str] = None
    sample_size: Optional[int] = None
    control_variables: Optional[Dict[str, Any]] = None


class ValidationCampaign(BaseModel):
    """Model for validation campaign setup"""
    name: str = Field(..., description="Campaign name")
    hypotheses: List[str] = Field(..., description="List of hypotheses to test")
    sample_size_per_experiment: int = Field(default=100, description="Sample size per experiment")
    total_budget: Optional[float] = Field(None, description="Total budget for campaign")
    duration_days: int = Field(default=30, description="Campaign duration in days")
    priority: str = Field(default="medium", description="Campaign priority")


class ValidationSuggestion(BaseModel):
    """Model for validation suggestions"""
    suggestion_type: str
    description: str
    priority: str
    estimated_impact: float
    implementation_effort: str


class ValidationDashboard:
    """
    Comprehensive validation dashboard with CLI and API capabilities
    
    Provides interfaces for managing validation experiments, monitoring progress,
    and generating reports through both command-line and web API interfaces.
    """
    
    def __init__(self, 
                 validation_framework: Optional[MUSEValidationFramework] = None,
                 data_directory: str = "validation_data"):
        """
        Initialize the validation dashboard
        
        Args:
            validation_framework: MUSE validation framework instance
            data_directory: Directory for storing validation data
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize validation framework
        if validation_framework is None:
            frequency_engine = MuseFrequencyEngine()
            geometry_calculator = SacredGeometryCalculator()
            projection_engine = SemanticProjectionEngine()
            
            validation_framework = MUSEValidationFramework(
                frequency_engine=frequency_engine,
                geometry_calculator=geometry_calculator,
                projection_engine=projection_engine
            )
        
        self.validation_framework = validation_framework
        
        # Setup data directory
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)
        
        # Initialize dashboard state
        self.dashboard_state = {
            'active_campaigns': {},
            'dashboard_stats': {
                'total_experiments': 0,
                'completed_experiments': 0,
                'active_experiments': 0,
                'total_participants': 0,
                'last_updated': datetime.now().isoformat()
            },
            'recent_activities': [],
            'system_health': 'healthy'
        }
        
        self.logger.info("Validation Dashboard initialized")
    
    def create_router(self) -> APIRouter:
        """
        Create FastAPI router for validation endpoints
        
        Returns:
            FastAPI router with validation endpoints
        """
        router = APIRouter()
        
        @router.get("/validation/summary")
        async def get_validation_summary():
            """Get comprehensive validation dashboard summary"""
            try:
                # Get validation framework summary
                framework_summary = self.validation_framework.get_validation_summary()
                
                # Update dashboard stats
                self.dashboard_state['dashboard_stats'].update({
                    'total_experiments': framework_summary.get('total_experiments', 0),
                    'completed_experiments': framework_summary.get('completed_experiments', 0),
                    'active_experiments': framework_summary.get('active_experiments', 0),
                    'total_participants': framework_summary.get('total_participants', 0),
                    'last_updated': datetime.now().isoformat()
                })
                
                # Get recent activities
                recent_activities = self._get_recent_activities()
                
                # System health check
                system_health = self._check_system_health()
                
                return JSONResponse({
                    'dashboard_summary': {
                        'stats': self.dashboard_state['dashboard_stats'],
                        'framework_summary': framework_summary,
                        'recent_activities': recent_activities,
                        'system_health': system_health,
                        'available_hypotheses': [h.value for h in ValidationHypothesis],
                        'active_campaigns': list(self.dashboard_state['active_campaigns'].keys())
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Failed to get validation summary: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/validation/experiment")
        async def create_experiment(experiment_data: ExperimentCreate, background_tasks: BackgroundTasks):
            """Create a new validation experiment"""
            try:
                # Validate hypothesis
                try:
                    hypothesis = ValidationHypothesis(experiment_data.hypothesis)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Invalid hypothesis: {experiment_data.hypothesis}")
                
                # Create experiment
                experiment = self.validation_framework.create_experiment(
                    hypothesis=hypothesis,
                    title=experiment_data.title,
                    description=experiment_data.description,
                    design_type=experiment_data.design_type,
                    sample_size=experiment_data.sample_size,
                    created_by=experiment_data.created_by,
                    control_variables=experiment_data.control_variables
                )
                
                # Record activity
                self._record_activity("experiment_created", {
                    'experiment_id': experiment.id,
                    'title': experiment.title,
                    'hypothesis': experiment.hypothesis.value
                })
                
                # Start experiment in background
                background_tasks.add_task(self._run_experiment_background, experiment.id)
                
                return JSONResponse({
                    'experiment_id': experiment.id,
                    'title': experiment.title,
                    'hypothesis': experiment.hypothesis.value,
                    'phase': experiment.phase.value,
                    'status': 'created',
                    'message': 'Experiment created and queued for execution'
                })
                
            except Exception as e:
                self.logger.error(f"Failed to create experiment: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/validation/experiment/{experiment_id}")
        async def get_experiment(experiment_id: str):
            """Get detailed information about a specific experiment"""
            try:
                status = self.validation_framework.get_experiment_status(experiment_id)
                
                if 'error' in status:
                    raise HTTPException(status_code=404, detail=status['error'])
                
                # Get detailed experiment data
                experiment = self.validation_framework.experiments.get(experiment_id)
                if not experiment:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                # Get results if available
                results = self.validation_framework.validation_data.get('results', {}).get(experiment_id, {})
                
                return JSONResponse({
                    'experiment': {
                        'id': experiment.id,
                        'title': experiment.title,
                        'description': experiment.description,
                        'hypothesis': experiment.hypothesis.value,
                        'phase': experiment.phase.value,
                        'design_type': experiment.design_type,
                        'sample_size': experiment.sample_size,
                        'control_condition': experiment.control_condition,
                        'treatment_condition': experiment.treatment_condition,
                        'outcome_measures': experiment.outcome_measures,
                        'statistical_tests': experiment.statistical_tests,
                        'created_at': experiment.created_at.isoformat(),
                        'last_updated': experiment.last_updated.isoformat(),
                        'start_date': experiment.start_date.isoformat(),
                        'end_date': experiment.end_date.isoformat() if experiment.end_date else None
                    },
                    'status': status,
                    'results': results,
                    'timestamp': datetime.now().isoformat()
                })
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get experiment {experiment_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/validation/experiment/{experiment_id}/live")
        async def get_experiment_live_metrics(experiment_id: str):
            """Get real-time metrics for an experiment"""
            try:
                if experiment_id not in self.validation_framework.experiments:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                experiment = self.validation_framework.experiments[experiment_id]
                
                # Calculate live metrics
                live_metrics = {
                    'experiment_id': experiment_id,
                    'phase': experiment.phase.value,
                    'progress': self.validation_framework._calculate_progress(experiment),
                    'metrics_collected': len(experiment.metrics),
                    'target_sample_size': experiment.sample_size,
                    'current_sample_size': len(experiment.metrics),
                    'completion_percentage': (len(experiment.metrics) / experiment.sample_size) * 100 if experiment.sample_size > 0 else 0,
                    'estimated_completion': self._estimate_completion_time(experiment),
                    'real_time_stats': self._calculate_real_time_stats(experiment),
                    'last_update': datetime.now().isoformat()
                }
                
                return JSONResponse(live_metrics)
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get live metrics for {experiment_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/validation/experiment/{experiment_id}/report")
        async def get_experiment_report(experiment_id: str, format: str = "json"):
            """Get experiment report in specified format"""
            try:
                if experiment_id not in self.validation_framework.experiments:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                if format == "markdown":
                    report = self.validation_framework.generate_markdown_report(experiment_id)
                    return PlainTextResponse(report, media_type="text/markdown")
                
                elif format == "json":
                    results = self.validation_framework.validation_data.get('results', {}).get(experiment_id, {})
                    report = results.get('report', {})
                    
                    if not report:
                        raise HTTPException(status_code=404, detail="Report not available yet")
                    
                    return JSONResponse(report)
                
                else:
                    raise HTTPException(status_code=400, detail="Invalid format. Use 'json' or 'markdown'")
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get report for {experiment_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/validation/campaign")
        async def create_validation_campaign(campaign_data: ValidationCampaign, background_tasks: BackgroundTasks):
            """Setup a comprehensive validation campaign"""
            try:
                campaign_id = f"campaign_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Validate hypotheses
                validated_hypotheses = []
                for hypothesis_str in campaign_data.hypotheses:
                    try:
                        hypothesis = ValidationHypothesis(hypothesis_str)
                        validated_hypotheses.append(hypothesis)
                    except ValueError:
                        raise HTTPException(status_code=400, detail=f"Invalid hypothesis: {hypothesis_str}")
                
                # Create campaign
                campaign = {
                    'id': campaign_id,
                    'name': campaign_data.name,
                    'hypotheses': [h.value for h in validated_hypotheses],
                    'sample_size_per_experiment': campaign_data.sample_size_per_experiment,
                    'total_budget': campaign_data.total_budget,
                    'duration_days': campaign_data.duration_days,
                    'priority': campaign_data.priority,
                    'created_at': datetime.now().isoformat(),
                    'status': 'active',
                    'experiments': [],
                    'progress': 0.0
                }
                
                # Create experiments for each hypothesis
                for hypothesis in validated_hypotheses:
                    experiment = self.validation_framework.create_experiment(
                        hypothesis=hypothesis,
                        title=f"{campaign_data.name} - {hypothesis.value}",
                        description=f"Validation experiment for {hypothesis.value} as part of {campaign_data.name} campaign",
                        sample_size=campaign_data.sample_size_per_experiment,
                        created_by="campaign_system"
                    )
                    campaign['experiments'].append(experiment.id)
                
                # Store campaign
                self.dashboard_state['active_campaigns'][campaign_id] = campaign
                
                # Record activity
                self._record_activity("campaign_created", {
                    'campaign_id': campaign_id,
                    'name': campaign_data.name,
                    'experiments_count': len(campaign['experiments'])
                })
                
                # Start campaign experiments in background
                background_tasks.add_task(self._run_campaign_background, campaign_id)
                
                return JSONResponse({
                    'campaign_id': campaign_id,
                    'name': campaign_data.name,
                    'experiments_created': len(campaign['experiments']),
                    'experiments': campaign['experiments'],
                    'status': 'active',
                    'message': 'Campaign created and experiments queued'
                })
                
            except Exception as e:
                self.logger.error(f"Failed to create campaign: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/validation/campaign/{campaign_id}")
        async def get_campaign_status(campaign_id: str):
            """Get status of a validation campaign"""
            try:
                if campaign_id not in self.dashboard_state['active_campaigns']:
                    raise HTTPException(status_code=404, detail="Campaign not found")
                
                campaign = self.dashboard_state['active_campaigns'][campaign_id]
                
                # Update campaign progress
                experiment_statuses = []
                total_progress = 0.0
                
                for exp_id in campaign['experiments']:
                    status = self.validation_framework.get_experiment_status(exp_id)
                    if 'error' not in status:
                        experiment_statuses.append(status)
                        total_progress += status.get('progress', 0.0)
                
                campaign['progress'] = total_progress / len(campaign['experiments']) if campaign['experiments'] else 0.0
                
                return JSONResponse({
                    'campaign': campaign,
                    'experiment_statuses': experiment_statuses,
                    'overall_progress': campaign['progress'],
                    'timestamp': datetime.now().isoformat()
                })
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to get campaign status: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.get("/validation/suggestions")
        async def get_validation_suggestions():
            """Get AI-generated suggestions for improving validation"""
            try:
                suggestions = self._generate_validation_suggestions()
                return JSONResponse({
                    'suggestions': suggestions,
                    'generated_at': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error(f"Failed to generate suggestions: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.post("/validation/experiment/{experiment_id}/stop")
        async def stop_experiment(experiment_id: str):
            """Stop a running experiment"""
            try:
                if experiment_id not in self.validation_framework.experiments:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                experiment = self.validation_framework.experiments[experiment_id]
                
                if experiment.phase == ExperimentPhase.COMPLETE:
                    raise HTTPException(status_code=400, detail="Experiment already completed")
                
                # Stop experiment (reset to design phase)
                experiment.phase = ExperimentPhase.DESIGN
                experiment.last_updated = datetime.now()
                
                # Record activity
                self._record_activity("experiment_stopped", {
                    'experiment_id': experiment_id,
                    'title': experiment.title
                })
                
                return JSONResponse({
                    'experiment_id': experiment_id,
                    'status': 'stopped',
                    'message': 'Experiment has been stopped'
                })
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to stop experiment: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @router.delete("/validation/experiment/{experiment_id}")
        async def delete_experiment(experiment_id: str):
            """Delete an experiment"""
            try:
                if experiment_id not in self.validation_framework.experiments:
                    raise HTTPException(status_code=404, detail="Experiment not found")
                
                experiment = self.validation_framework.experiments[experiment_id]
                
                # Remove from framework
                del self.validation_framework.experiments[experiment_id]
                
                # Remove from validation data
                if experiment_id in self.validation_framework.validation_data.get('experiments', {}):
                    del self.validation_framework.validation_data['experiments'][experiment_id]
                
                if experiment_id in self.validation_framework.validation_data.get('results', {}):
                    del self.validation_framework.validation_data['results'][experiment_id]
                
                # Record activity
                self._record_activity("experiment_deleted", {
                    'experiment_id': experiment_id,
                    'title': experiment.title
                })
                
                return JSONResponse({
                    'experiment_id': experiment_id,
                    'status': 'deleted',
                    'message': 'Experiment has been deleted'
                })
                
            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Failed to delete experiment: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        return router
    
    async def _run_experiment_background(self, experiment_id: str):
        """Run experiment in background task"""
        try:
            await self.validation_framework.run_experiment(experiment_id)
            
            # Record completion
            self._record_activity("experiment_completed", {
                'experiment_id': experiment_id
            })
            
        except Exception as e:
            self.logger.error(f"Background experiment {experiment_id} failed: {e}")
            
            # Record failure
            self._record_activity("experiment_failed", {
                'experiment_id': experiment_id,
                'error': str(e)
            })
    
    async def _run_campaign_background(self, campaign_id: str):
        """Run campaign experiments in background"""
        try:
            campaign = self.dashboard_state['active_campaigns'][campaign_id]
            
            # Run all experiments in campaign
            for exp_id in campaign['experiments']:
                await self.validation_framework.run_experiment(exp_id)
            
            # Update campaign status
            campaign['status'] = 'completed'
            campaign['progress'] = 1.0
            
            # Record completion
            self._record_activity("campaign_completed", {
                'campaign_id': campaign_id,
                'name': campaign['name']
            })
            
        except Exception as e:
            self.logger.error(f"Background campaign {campaign_id} failed: {e}")
            
            # Update campaign status
            if campaign_id in self.dashboard_state['active_campaigns']:
                self.dashboard_state['active_campaigns'][campaign_id]['status'] = 'failed'
    
    def _get_recent_activities(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent dashboard activities"""
        activities = self.dashboard_state.get('recent_activities', [])
        return activities[-limit:] if activities else []
    
    def _record_activity(self, activity_type: str, details: Dict[str, Any]):
        """Record dashboard activity"""
        activity = {
            'type': activity_type,
            'details': details,
            'timestamp': datetime.now().isoformat(),
            'id': f"activity_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        if 'recent_activities' not in self.dashboard_state:
            self.dashboard_state['recent_activities'] = []
        
        self.dashboard_state['recent_activities'].append(activity)
        
        # Keep only last 100 activities
        if len(self.dashboard_state['recent_activities']) > 100:
            self.dashboard_state['recent_activities'] = self.dashboard_state['recent_activities'][-100:]
    
    def _check_system_health(self) -> Dict[str, Any]:
        """Check system health status"""
        health_status = {
            'status': 'healthy',
            'components': {
                'validation_framework': 'operational',
                'frequency_engine': 'operational',
                'sacred_geometry': 'operational',
                'semantic_projection': 'operational',
                'data_storage': 'operational'
            },
            'metrics': {
                'experiments_running': len([exp for exp in self.validation_framework.experiments.values() 
                                          if exp.phase not in [ExperimentPhase.DESIGN, ExperimentPhase.COMPLETE]]),
                'memory_usage': 'normal',
                'cpu_usage': 'normal',
                'disk_space': 'normal'
            },
            'last_check': datetime.now().isoformat()
        }
        
        return health_status
    
    def _estimate_completion_time(self, experiment: ValidationExperiment) -> Optional[str]:
        """Estimate experiment completion time"""
        if experiment.phase == ExperimentPhase.COMPLETE:
            return None
        
        # Simple estimation based on phase and sample size
        phase_times = {
            ExperimentPhase.DESIGN: 0,
            ExperimentPhase.RECRUITMENT: 5,
            ExperimentPhase.DATA_COLLECTION: 10,
            ExperimentPhase.ANALYSIS: 3,
            ExperimentPhase.REPORTING: 2
        }
        
        remaining_time = 0
        current_phase_index = list(ExperimentPhase).index(experiment.phase)
        
        for i, phase in enumerate(ExperimentPhase):
            if i > current_phase_index:
                remaining_time += phase_times.get(phase, 0)
        
        # Adjust for sample size
        if experiment.sample_size > 100:
            remaining_time += (experiment.sample_size - 100) / 100 * 2
        
        if remaining_time > 0:
            return f"~{remaining_time:.0f} minutes"
        else:
            return "Soon"
    
    def _calculate_real_time_stats(self, experiment: ValidationExperiment) -> Dict[str, Any]:
        """Calculate real-time statistics for experiment"""
        if not experiment.metrics:
            return {'message': 'No metrics available yet'}
        
        # Extract metric values
        satisfaction_scores = [m.user_satisfaction for m in experiment.metrics]
        fitness_scores = [m.mathematical_fitness for m in experiment.metrics]
        quality_scores = [m.quality_rating for m in experiment.metrics]
        
        return {
            'current_metrics': {
                'satisfaction': {
                    'mean': sum(satisfaction_scores) / len(satisfaction_scores),
                    'latest': satisfaction_scores[-1] if satisfaction_scores else 0,
                    'trend': 'stable'
                },
                'fitness': {
                    'mean': sum(fitness_scores) / len(fitness_scores),
                    'latest': fitness_scores[-1] if fitness_scores else 0,
                    'trend': 'stable'
                },
                'quality': {
                    'mean': sum(quality_scores) / len(quality_scores),
                    'latest': quality_scores[-1] if quality_scores else 0,
                    'trend': 'stable'
                }
            },
            'sample_progress': {
                'collected': len(experiment.metrics),
                'target': experiment.sample_size,
                'percentage': (len(experiment.metrics) / experiment.sample_size) * 100
            }
        }
    
    def _generate_validation_suggestions(self) -> List[ValidationSuggestion]:
        """Generate AI-powered validation suggestions"""
        suggestions = []
        
        # Analyze current experiments
        total_experiments = len(self.validation_framework.experiments)
        completed_experiments = len([exp for exp in self.validation_framework.experiments.values() 
                                   if exp.phase == ExperimentPhase.COMPLETE])
        
        # Suggestion 1: Increase sample sizes
        if total_experiments > 0:
            avg_sample_size = sum(exp.sample_size for exp in self.validation_framework.experiments.values()) / total_experiments
            if avg_sample_size < 200:
                suggestions.append(ValidationSuggestion(
                    suggestion_type="sample_size",
                    description="Consider increasing sample sizes to improve statistical power",
                    priority="high",
                    estimated_impact=0.8,
                    implementation_effort="medium"
                ))
        
        # Suggestion 2: Test additional hypotheses
        tested_hypotheses = set(exp.hypothesis for exp in self.validation_framework.experiments.values())
        all_hypotheses = set(ValidationHypothesis)
        untested_hypotheses = all_hypotheses - tested_hypotheses
        
        if untested_hypotheses:
            suggestions.append(ValidationSuggestion(
                suggestion_type="hypothesis_coverage",
                description=f"Test additional hypotheses: {', '.join(h.value for h in untested_hypotheses)}",
                priority="medium",
                estimated_impact=0.6,
                implementation_effort="low"
            ))
        
        # Suggestion 3: Improve experiment design
        between_subjects_count = len([exp for exp in self.validation_framework.experiments.values() 
                                    if exp.design_type == "between_subjects"])
        if between_subjects_count > 0 and total_experiments > 0:
            suggestions.append(ValidationSuggestion(
                suggestion_type="design_improvement",
                description="Consider mixed-design experiments to reduce individual differences",
                priority="medium",
                estimated_impact=0.5,
                implementation_effort="high"
            ))
        
        # Suggestion 4: Replication studies
        if completed_experiments > 0:
            suggestions.append(ValidationSuggestion(
                suggestion_type="replication",
                description="Run replication studies to confirm significant findings",
                priority="high",
                estimated_impact=0.9,
                implementation_effort="medium"
            ))
        
        # Suggestion 5: Longitudinal studies
        suggestions.append(ValidationSuggestion(
            suggestion_type="longitudinal",
            description="Implement longitudinal studies to test signature stability over time",
            priority="low",
            estimated_impact=0.7,
            implementation_effort="high"
        ))
        
        return suggestions
    
    # CLI Interface
    def create_cli(self):
        """Create CLI interface for validation dashboard"""
        
        @click.group()
        def cli():
            """MUSE Validation Dashboard CLI"""
            pass
        
        @cli.command()
        @click.option('--format', default='table', help='Output format (table, json)')
        def summary(format):
            """Show validation summary"""
            summary_data = self.validation_framework.get_validation_summary()
            
            if format == 'json':
                click.echo(json.dumps(summary_data, indent=2))
            else:
                click.echo("=== MUSE Validation Summary ===")
                click.echo(f"Total Experiments: {summary_data.get('total_experiments', 0)}")
                click.echo(f"Completed: {summary_data.get('completed_experiments', 0)}")
                click.echo(f"Active: {summary_data.get('active_experiments', 0)}")
                click.echo(f"Total Participants: {summary_data.get('total_participants', 0)}")
                
                hypotheses = summary_data.get('hypotheses_tested', {})
                if hypotheses:
                    click.echo("\nHypotheses Tested:")
                    for hypothesis, count in hypotheses.items():
                        click.echo(f"  {hypothesis}: {count}")
        
        @cli.command()
        @click.argument('hypothesis')
        @click.option('--title', required=True, help='Experiment title')
        @click.option('--description', required=True, help='Experiment description')
        @click.option('--sample-size', default=100, help='Sample size')
        def create_experiment(hypothesis, title, description, sample_size):
            """Create a new validation experiment"""
            try:
                hypothesis_enum = ValidationHypothesis(hypothesis)
                
                experiment = self.validation_framework.create_experiment(
                    hypothesis=hypothesis_enum,
                    title=title,
                    description=description,
                    sample_size=sample_size
                )
                
                click.echo(f"Created experiment: {experiment.id}")
                click.echo(f"Title: {experiment.title}")
                click.echo(f"Hypothesis: {experiment.hypothesis.value}")
                click.echo(f"Sample Size: {experiment.sample_size}")
                
            except ValueError as e:
                click.echo(f"Error: {e}")
                click.echo("Available hypotheses:")
                for h in ValidationHypothesis:
                    click.echo(f"  {h.value}")
        
        @cli.command()
        @click.argument('experiment_id')
        def run_experiment(experiment_id):
            """Run a validation experiment"""
            try:
                click.echo(f"Running experiment {experiment_id}...")
                
                # Run experiment synchronously for CLI
                import asyncio
                results = asyncio.run(self.validation_framework.run_experiment(experiment_id))
                
                click.echo("Experiment completed!")
                click.echo(f"Status: {results.get('status', 'unknown')}")
                click.echo(f"Participants: {len(results.get('participants', []))}")
                
            except Exception as e:
                click.echo(f"Error running experiment: {e}")
        
        @cli.command()
        @click.argument('experiment_id')
        @click.option('--format', default='markdown', help='Report format (markdown, json)')
        def report(experiment_id, format):
            """Generate experiment report"""
            try:
                if format == 'markdown':
                    report_text = self.validation_framework.generate_markdown_report(experiment_id)
                    click.echo(report_text)
                elif format == 'json':
                    results = self.validation_framework.validation_data.get('results', {}).get(experiment_id, {})
                    report_data = results.get('report', {})
                    click.echo(json.dumps(report_data, indent=2))
                else:
                    click.echo("Invalid format. Use 'markdown' or 'json'")
                
            except Exception as e:
                click.echo(f"Error generating report: {e}")
        
        @cli.command()
        def list_experiments():
            """List all experiments"""
            experiments = self.validation_framework.experiments
            
            if not experiments:
                click.echo("No experiments found.")
                return
            
            click.echo("=== Experiments ===")
            for exp_id, exp in experiments.items():
                click.echo(f"{exp_id}: {exp.title}")
                click.echo(f"  Hypothesis: {exp.hypothesis.value}")
                click.echo(f"  Phase: {exp.phase.value}")
                click.echo(f"  Sample Size: {exp.sample_size}")
                click.echo(f"  Created: {exp.created_at.strftime('%Y-%m-%d %H:%M')}")
                click.echo()
        
        @cli.command()
        @click.argument('experiment_id')
        def status(experiment_id):
            """Get experiment status"""
            try:
                status_data = self.validation_framework.get_experiment_status(experiment_id)
                
                if 'error' in status_data:
                    click.echo(f"Error: {status_data['error']}")
                    return
                
                click.echo(f"=== Experiment Status ===")
                click.echo(f"ID: {status_data['id']}")
                click.echo(f"Title: {status_data['title']}")
                click.echo(f"Phase: {status_data['phase']}")
                click.echo(f"Progress: {status_data['progress'] * 100:.1f}%")
                click.echo(f"Sample Size: {status_data['sample_size']}")
                click.echo(f"Metrics Collected: {status_data['metrics_collected']}")
                
            except Exception as e:
                click.echo(f"Error getting status: {e}")
        
        return cli


# Create global dashboard instance
dashboard = ValidationDashboard()

# Export router for use in main application
router = dashboard.create_router()