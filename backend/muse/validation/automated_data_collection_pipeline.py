"""
Automated Data Collection Pipeline for MUSE Validation

This module provides comprehensive automated data collection capabilities for
validation experiments, including realistic simulation, quality control, and
real-time monitoring.
"""

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import numpy as np
from scipy import stats

from .participant_recruitment_system import ParticipantProfile, ParticipantStatus
from .mathematical_validation_framework import ValidationExperiment, ValidationMetrics
from ..core.frequency_engine import MuseFrequencyEngine, FrequencySignature
from ..core.sacred_geometry_calculator import SacredGeometryCalculator
from ..core.semantic_projection_engine import SemanticProjectionEngine


class DataCollectionStatus(Enum):
    """Status of data collection process"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DataQualityFlag(Enum):
    """Quality flags for collected data"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INVALID = "invalid"


@dataclass
class DataCollectionTask:
    """Individual data collection task"""
    id: str
    experiment_id: str
    participant_id: str
    task_type: str
    
    # Task parameters
    instructions: str
    expected_duration: timedelta
    data_schema: Dict[str, Any]
    quality_checks: List[str]
    
    # Execution tracking
    status: DataCollectionStatus
    start_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    
    # Collected data
    raw_data: Dict[str, Any] = None
    processed_data: Dict[str, Any] = None
    quality_metrics: Dict[str, float] = None
    quality_flag: DataQualityFlag = None
    
    # Metadata
    created_at: datetime = None
    last_updated: datetime = None
    collection_notes: List[str] = None


@dataclass
class DataCollectionSession:
    """Complete data collection session for a participant"""
    id: str
    experiment_id: str
    participant_id: str
    
    # Session configuration
    tasks: List[DataCollectionTask]
    session_duration: timedelta
    break_intervals: List[timedelta]
    
    # Progress tracking
    status: DataCollectionStatus
    current_task_index: int = 0
    completion_percentage: float = 0.0
    
    # Quality monitoring
    overall_quality_score: float = 0.0
    attention_check_results: List[bool] = None
    response_time_patterns: List[float] = None
    
    # Session metadata
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_duration: Optional[timedelta] = None
    
    # Environmental factors
    environmental_conditions: Dict[str, Any] = None
    technical_issues: List[str] = None


class AutomatedDataCollectionPipeline:
    """
    Comprehensive automated data collection pipeline
    
    Provides realistic simulation of data collection processes including
    quality control, attention checks, and real-time monitoring.
    """
    
    def __init__(self, 
                 frequency_engine: Optional[MuseFrequencyEngine] = None,
                 geometry_calculator: Optional[SacredGeometryCalculator] = None,
                 projection_engine: Optional[SemanticProjectionEngine] = None):
        """
        Initialize the data collection pipeline
        
        Args:
            frequency_engine: MUSE frequency engine for data generation
            geometry_calculator: Sacred geometry calculator for mathematical data
            projection_engine: Semantic projection engine for meaning analysis
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize core engines
        self.frequency_engine = frequency_engine or MuseFrequencyEngine()
        self.geometry_calculator = geometry_calculator or SacredGeometryCalculator()
        self.projection_engine = projection_engine or SemanticProjectionEngine()
        
        # Active data collection sessions
        self.active_sessions: Dict[str, DataCollectionSession] = {}
        self.completed_sessions: Dict[str, DataCollectionSession] = {}
        
        # Data collection statistics
        self.collection_stats = {
            'total_sessions': 0,
            'completed_sessions': 0,
            'failed_sessions': 0,
            'average_quality_score': 0.0,
            'average_completion_time': 0.0,
            'attention_check_pass_rate': 0.0
        }
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.9,
            'good': 0.8,
            'acceptable': 0.7,
            'poor': 0.5,
            'invalid': 0.0
        }
        
        # Task templates for different experiment types
        self.task_templates = {
            'sacred_geometry_evaluation': {
                'instructions': 'Evaluate the aesthetic appeal and mathematical beauty of various geometric forms',
                'expected_duration': timedelta(minutes=15),
                'data_schema': {
                    'geometry_ratings': {'type': 'list', 'item_type': 'float'},
                    'aesthetic_preferences': {'type': 'dict'},
                    'mathematical_understanding': {'type': 'dict'}
                },
                'quality_checks': ['response_time', 'consistency', 'attention_check']
            },
            'archetypal_preference_assessment': {
                'instructions': 'Rate your preference for different creative styles and themes',
                'expected_duration': timedelta(minutes=20),
                'data_schema': {
                    'style_preferences': {'type': 'dict'},
                    'thematic_resonance': {'type': 'dict'},
                    'emotional_responses': {'type': 'list'}
                },
                'quality_checks': ['response_time', 'consistency', 'attention_check']
            },
            'creative_output_evaluation': {
                'instructions': 'Evaluate and compare different creative outputs',
                'expected_duration': timedelta(minutes=25),
                'data_schema': {
                    'quality_ratings': {'type': 'list', 'item_type': 'float'},
                    'creativity_scores': {'type': 'list', 'item_type': 'float'},
                    'preference_rankings': {'type': 'list'}
                },
                'quality_checks': ['response_time', 'consistency', 'attention_check', 'ranking_validity']
            },
            'frequency_signature_validation': {
                'instructions': 'Complete personality and preference assessments',
                'expected_duration': timedelta(minutes=30),
                'data_schema': {
                    'personality_responses': {'type': 'dict'},
                    'creative_preferences': {'type': 'dict'},
                    'mathematical_affinity': {'type': 'dict'}
                },
                'quality_checks': ['response_time', 'consistency', 'attention_check', 'personality_validity']
            }
        }
        
        self.logger.info("Automated Data Collection Pipeline initialized")
    
    async def collect_experiment_data(self, 
                                    experiment: ValidationExperiment,
                                    participants: List[ParticipantProfile]) -> Dict[str, Any]:
        """
        Collect data for a complete validation experiment
        
        Args:
            experiment: Validation experiment
            participants: List of participant profiles
            
        Returns:
            Complete experiment data collection results
        """
        collection_id = f"collection_{experiment.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Starting data collection for experiment {experiment.id} with {len(participants)} participants")
        
        # Initialize collection results
        collection_results = {
            'collection_id': collection_id,
            'experiment_id': experiment.id,
            'participant_count': len(participants),
            'sessions': {},
            'quality_summary': {},
            'completion_statistics': {},
            'start_time': datetime.now(),
            'end_time': None,
            'status': DataCollectionStatus.ACTIVE
        }
        
        try:
            # Create data collection sessions for each participant
            sessions = []
            for participant in participants:
                session = await self._create_data_collection_session(experiment, participant)
                sessions.append(session)
                self.active_sessions[session.id] = session
            
            # Run data collection sessions in parallel (with some delay to simulate real-world timing)
            session_tasks = []
            for session in sessions:
                task = asyncio.create_task(self._run_data_collection_session(session))
                session_tasks.append(task)
                
                # Add small delay between session starts
                await asyncio.sleep(0.1)
            
            # Wait for all sessions to complete
            session_results = await asyncio.gather(*session_tasks, return_exceptions=True)
            
            # Process session results
            for i, result in enumerate(session_results):
                session = sessions[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"Session {session.id} failed: {result}")
                    session.status = DataCollectionStatus.FAILED
                    collection_results['sessions'][session.id] = {
                        'status': 'failed',
                        'error': str(result)
                    }
                else:
                    # Move to completed sessions
                    self.completed_sessions[session.id] = session
                    if session.id in self.active_sessions:
                        del self.active_sessions[session.id]
                    
                    collection_results['sessions'][session.id] = result
            
            # Calculate quality summary
            collection_results['quality_summary'] = self._calculate_quality_summary(sessions)
            
            # Calculate completion statistics
            collection_results['completion_statistics'] = self._calculate_completion_statistics(sessions)
            
            # Update collection status
            collection_results['status'] = DataCollectionStatus.COMPLETED
            collection_results['end_time'] = datetime.now()
            
            # Update global statistics
            self._update_collection_statistics(sessions)
            
            self.logger.info(f"Data collection completed for experiment {experiment.id}")
            
            return collection_results
            
        except Exception as e:
            collection_results['status'] = DataCollectionStatus.FAILED
            collection_results['error'] = str(e)
            collection_results['end_time'] = datetime.now()
            
            self.logger.error(f"Data collection failed for experiment {experiment.id}: {e}")
            raise
    
    async def _create_data_collection_session(self, 
                                            experiment: ValidationExperiment,
                                            participant: ParticipantProfile) -> DataCollectionSession:
        """Create a data collection session for a participant"""
        session_id = f"session_{experiment.id}_{participant.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine task types based on experiment hypothesis
        task_types = self._get_task_types_for_hypothesis(experiment.hypothesis)
        
        # Create tasks for the session
        tasks = []
        for i, task_type in enumerate(task_types):
            task = await self._create_data_collection_task(
                session_id, experiment.id, participant.id, task_type, i
            )
            tasks.append(task)
        
        # Calculate session duration
        total_duration = sum((task.expected_duration for task in tasks), timedelta())
        break_intervals = [timedelta(minutes=5) for _ in range(len(tasks) - 1)]
        
        # Create session
        session = DataCollectionSession(
            id=session_id,
            experiment_id=experiment.id,
            participant_id=participant.id,
            tasks=tasks,
            session_duration=total_duration,
            break_intervals=break_intervals,
            status=DataCollectionStatus.PENDING,
            attention_check_results=[],
            response_time_patterns=[],
            environmental_conditions=self._simulate_environmental_conditions(),
            technical_issues=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        return session
    
    def _get_task_types_for_hypothesis(self, hypothesis) -> List[str]:
        """Get appropriate task types for a given hypothesis"""
        from .mathematical_validation_framework import ValidationHypothesis
        
        task_mappings = {
            ValidationHypothesis.SACRED_GEOMETRY_EFFECTIVENESS: [
                'sacred_geometry_evaluation',
                'creative_output_evaluation'
            ],
            ValidationHypothesis.HARDWARE_ENTROPY_UNIQUENESS: [
                'creative_output_evaluation',
                'frequency_signature_validation'
            ],
            ValidationHypothesis.ARCHETYPAL_PREDICTION_ACCURACY: [
                'archetypal_preference_assessment',
                'frequency_signature_validation'
            ],
            ValidationHypothesis.DISCOVERY_VS_GENERATION: [
                'creative_output_evaluation',
                'sacred_geometry_evaluation'
            ],
            ValidationHypothesis.FREQUENCY_SIGNATURE_STABILITY: [
                'frequency_signature_validation',
                'archetypal_preference_assessment'
            ],
            ValidationHypothesis.RESONANCE_MATCHING_ACCURACY: [
                'archetypal_preference_assessment',
                'creative_output_evaluation'
            ]
        }
        
        return task_mappings.get(hypothesis, ['creative_output_evaluation'])
    
    async def _create_data_collection_task(self, 
                                         session_id: str,
                                         experiment_id: str,
                                         participant_id: str,
                                         task_type: str,
                                         task_index: int) -> DataCollectionTask:
        """Create an individual data collection task"""
        task_id = f"task_{session_id}_{task_index}_{task_type}"
        
        template = self.task_templates.get(task_type, self.task_templates['creative_output_evaluation'])
        
        task = DataCollectionTask(
            id=task_id,
            experiment_id=experiment_id,
            participant_id=participant_id,
            task_type=task_type,
            instructions=template['instructions'],
            expected_duration=template['expected_duration'],
            data_schema=template['data_schema'],
            quality_checks=template['quality_checks'],
            status=DataCollectionStatus.PENDING,
            collection_notes=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        return task
    
    async def _run_data_collection_session(self, session: DataCollectionSession) -> Dict[str, Any]:
        """Run a complete data collection session"""
        session.status = DataCollectionStatus.ACTIVE
        session.start_time = datetime.now()
        
        session_results = {
            'session_id': session.id,
            'participant_id': session.participant_id,
            'task_results': [],
            'quality_metrics': {},
            'completion_time': None,
            'status': 'active'
        }
        
        try:
            # Run each task in the session
            for i, task in enumerate(session.tasks):
                session.current_task_index = i
                
                # Run the task
                task_result = await self._run_data_collection_task(task, session)
                session_results['task_results'].append(task_result)
                
                # Update session progress
                session.completion_percentage = (i + 1) / len(session.tasks) * 100
                
                # Simulate break between tasks (except after last task)
                if i < len(session.tasks) - 1:
                    break_duration = session.break_intervals[i]
                    await asyncio.sleep(break_duration.total_seconds() / 100)  # Scaled down for simulation
            
            # Calculate session quality metrics
            session_results['quality_metrics'] = self._calculate_session_quality_metrics(session)
            
            # Complete session
            session.status = DataCollectionStatus.COMPLETED
            session.end_time = datetime.now()
            session.total_duration = session.end_time - session.start_time
            session.overall_quality_score = session_results['quality_metrics'].get('overall_score', 0.0)
            
            session_results['completion_time'] = session.total_duration.total_seconds()
            session_results['status'] = 'completed'
            
            return session_results
            
        except Exception as e:
            session.status = DataCollectionStatus.FAILED
            session.end_time = datetime.now()
            session_results['status'] = 'failed'
            session_results['error'] = str(e)
            
            self.logger.error(f"Session {session.id} failed: {e}")
            raise
    
    async def _run_data_collection_task(self, 
                                      task: DataCollectionTask,
                                      session: DataCollectionSession) -> Dict[str, Any]:
        """Run an individual data collection task"""
        task.status = DataCollectionStatus.ACTIVE
        task.start_time = datetime.now()
        
        # Get participant profile
        participant = self._get_participant_profile(task.participant_id)
        
        # Generate task-specific data based on task type
        if task.task_type == 'sacred_geometry_evaluation':
            raw_data = await self._collect_sacred_geometry_data(task, participant)
        elif task.task_type == 'archetypal_preference_assessment':
            raw_data = await self._collect_archetypal_preference_data(task, participant)
        elif task.task_type == 'creative_output_evaluation':
            raw_data = await self._collect_creative_output_data(task, participant)
        elif task.task_type == 'frequency_signature_validation':
            raw_data = await self._collect_frequency_signature_data(task, participant)
        else:
            raw_data = await self._collect_generic_task_data(task, participant)
        
        # Process the collected data
        processed_data = await self._process_task_data(raw_data, task)
        
        # Calculate quality metrics
        quality_metrics = self._calculate_task_quality_metrics(raw_data, processed_data, task)
        
        # Assign quality flag
        quality_flag = self._assign_quality_flag(quality_metrics)
        
        # Update task
        task.raw_data = raw_data
        task.processed_data = processed_data
        task.quality_metrics = quality_metrics
        task.quality_flag = quality_flag
        task.status = DataCollectionStatus.COMPLETED
        task.completion_time = datetime.now()
        
        return {
            'task_id': task.id,
            'task_type': task.task_type,
            'raw_data': raw_data,
            'processed_data': processed_data,
            'quality_metrics': quality_metrics,
            'quality_flag': quality_flag.value,
            'completion_time': (task.completion_time - task.start_time).total_seconds(),
            'status': 'completed'
        }
    
    async def _collect_sacred_geometry_data(self, 
                                          task: DataCollectionTask, 
                                          participant: Optional[ParticipantProfile]) -> Dict[str, Any]:
        """Collect sacred geometry evaluation data"""
        # Simulate participant evaluating various geometric forms
        geometric_forms = [
            'golden_rectangle', 'fibonacci_spiral', 'pentagram', 'hexagon',
            'vesica_piscis', 'flower_of_life', 'metatron_cube', 'sri_yantra'
        ]
        
        # Generate ratings based on participant's mathematical affinity
        math_affinity = 0.5  # Default
        if participant:
            math_affinity = participant.mathematical_background.get('mathematical_affinity', 0.5)
        
        geometry_ratings = {}
        for form in geometric_forms:
            # Base rating influenced by mathematical affinity
            base_rating = math_affinity * 0.6 + random.uniform(0.2, 0.8)
            
            # Add noise based on participant reliability
            reliability = participant.reliability_score if participant else 0.8
            noise = random.uniform(-0.2, 0.2) * (1 - reliability)
            
            rating = max(0.0, min(1.0, base_rating + noise))
            geometry_ratings[form] = rating
        
        # Aesthetic preferences
        aesthetic_preferences = {
            'symmetry_importance': random.uniform(0.3, 1.0),
            'complexity_preference': random.uniform(0.2, 0.9),
            'color_influence': random.uniform(0.1, 0.7),
            'size_preference': random.choice(['small', 'medium', 'large'])
        }
        
        # Mathematical understanding assessment
        mathematical_understanding = {
            'golden_ratio_familiarity': random.uniform(0.0, 1.0),
            'fibonacci_knowledge': random.uniform(0.0, 1.0),
            'geometric_intuition': random.uniform(0.2, 1.0),
            'pattern_recognition': random.uniform(0.3, 1.0)
        }
        
        # Add attention check
        attention_check_passed = random.random() < 0.85  # 85% pass rate
        
        # Simulate task completion time
        await asyncio.sleep(0.05)  # Scaled simulation time
        
        return {
            'geometry_ratings': geometry_ratings,
            'aesthetic_preferences': aesthetic_preferences,
            'mathematical_understanding': mathematical_understanding,
            'attention_check_passed': attention_check_passed,
            'response_times': [random.uniform(2.0, 15.0) for _ in geometric_forms],
            'task_engagement': random.uniform(0.6, 1.0)
        }
    
    async def _collect_archetypal_preference_data(self, 
                                                task: DataCollectionTask,
                                                participant: Optional[ParticipantProfile]) -> Dict[str, Any]:
        """Collect archetypal preference assessment data"""
        from ..core.frequency_engine import MuseArchetype
        
        # Generate preferences based on participant's personality
        archetypes = [archetype.value for archetype in MuseArchetype]
        
        style_preferences = {}
        for archetype in archetypes:
            # Base preference influenced by personality traits
            base_preference = random.uniform(0.2, 0.8)
            
            if participant:
                # Adjust based on personality traits
                if archetype == 'SOPHIA' and participant.personality_traits.get('openness', 0.5) > 0.7:
                    base_preference += 0.2
                elif archetype == 'THALIA' and participant.personality_traits.get('extraversion', 0.5) > 0.7:
                    base_preference += 0.2
                elif archetype == 'PSYCHE' and participant.personality_traits.get('neuroticism', 0.5) > 0.7:
                    base_preference += 0.2
            
            style_preferences[archetype] = max(0.0, min(1.0, base_preference))
        
        # Thematic resonance
        themes = [
            'heroic_journey', 'romantic_love', 'cosmic_wonder', 'inner_reflection',
            'joyful_celebration', 'tragic_beauty', 'sacred_mystery', 'technical_mastery'
        ]
        
        thematic_resonance = {}
        for theme in themes:
            resonance = random.uniform(0.1, 0.9)
            thematic_resonance[theme] = resonance
        
        # Emotional responses
        emotional_responses = [
            random.uniform(0.0, 1.0) for _ in range(20)  # 20 emotional stimuli
        ]
        
        # Attention check
        attention_check_passed = random.random() < 0.88  # 88% pass rate
        
        await asyncio.sleep(0.07)  # Scaled simulation time
        
        return {
            'style_preferences': style_preferences,
            'thematic_resonance': thematic_resonance,
            'emotional_responses': emotional_responses,
            'attention_check_passed': attention_check_passed,
            'response_times': [random.uniform(3.0, 20.0) for _ in range(len(archetypes) + len(themes))],
            'task_engagement': random.uniform(0.7, 1.0)
        }
    
    async def _collect_creative_output_data(self, 
                                          task: DataCollectionTask,
                                          participant: Optional[ParticipantProfile]) -> Dict[str, Any]:
        """Collect creative output evaluation data"""
        # Simulate evaluating multiple creative outputs
        num_outputs = random.randint(8, 15)
        
        quality_ratings = []
        creativity_scores = []
        
        for i in range(num_outputs):
            # Generate ratings with some consistency
            base_quality = random.uniform(0.3, 0.9)
            base_creativity = random.uniform(0.2, 0.8)
            
            # Add participant-specific bias
            if participant:
                creative_experience = participant.creative_preferences.get('creative_experience', 'none')
                if creative_experience == 'experienced':
                    base_quality += 0.1
                    base_creativity += 0.1
                elif creative_experience == 'none':
                    base_quality -= 0.1
                    base_creativity -= 0.1
            
            quality_ratings.append(max(0.0, min(1.0, base_quality + random.uniform(-0.1, 0.1))))
            creativity_scores.append(max(0.0, min(1.0, base_creativity + random.uniform(-0.1, 0.1))))
        
        # Preference rankings (consistent with ratings)
        output_indices = list(range(num_outputs))
        preference_rankings = sorted(output_indices, key=lambda i: quality_ratings[i], reverse=True)
        
        # Add some noise to rankings
        for i in range(len(preference_rankings) - 1):
            if random.random() < 0.15:  # 15% chance of ranking inconsistency
                preference_rankings[i], preference_rankings[i + 1] = preference_rankings[i + 1], preference_rankings[i]
        
        # Attention check
        attention_check_passed = random.random() < 0.82  # 82% pass rate
        
        await asyncio.sleep(0.08)  # Scaled simulation time
        
        return {
            'quality_ratings': quality_ratings,
            'creativity_scores': creativity_scores,
            'preference_rankings': preference_rankings,
            'attention_check_passed': attention_check_passed,
            'response_times': [random.uniform(5.0, 25.0) for _ in range(num_outputs)],
            'task_engagement': random.uniform(0.6, 1.0)
        }
    
    async def _collect_frequency_signature_data(self, 
                                              task: DataCollectionTask,
                                              participant: Optional[ParticipantProfile]) -> Dict[str, Any]:
        """Collect frequency signature validation data"""
        # Personality assessment responses
        personality_questions = [
            'openness_1', 'openness_2', 'openness_3',
            'conscientiousness_1', 'conscientiousness_2', 'conscientiousness_3',
            'extraversion_1', 'extraversion_2', 'extraversion_3',
            'agreeableness_1', 'agreeableness_2', 'agreeableness_3',
            'neuroticism_1', 'neuroticism_2', 'neuroticism_3'
        ]
        
        personality_responses = {}
        for question in personality_questions:
            # Generate response based on participant's actual traits if available
            base_response = random.uniform(1.0, 7.0)  # 7-point scale
            
            if participant:
                trait_name = question.split('_')[0]
                if trait_name in participant.personality_traits:
                    trait_value = participant.personality_traits[trait_name]
                    base_response = 1.0 + trait_value * 6.0  # Convert to 1-7 scale
                    
                    # Add some noise for test-retest reliability
                    noise = random.uniform(-0.5, 0.5)
                    base_response += noise
            
            personality_responses[question] = max(1.0, min(7.0, base_response))
        
        # Creative preferences
        creative_preferences = {
            'poetry_style_preference': random.choice(['epic', 'lyric', 'narrative', 'free_verse']),
            'emotional_range_preference': random.choice(['tragic', 'comedic', 'sacred', 'balanced']),
            'collaboration_preference': random.choice(['individual', 'small_group', 'large_group']),
            'inspiration_sources': random.sample([
                'nature', 'music', 'literature', 'art', 'mathematics', 'spirituality', 'technology'
            ], random.randint(2, 5))
        }
        
        # Mathematical affinity assessment
        mathematical_affinity = {
            'geometry_interest': random.uniform(0.0, 1.0),
            'pattern_recognition': random.uniform(0.2, 1.0),
            'mathematical_beauty': random.uniform(0.0, 1.0),
            'logical_reasoning': random.uniform(0.3, 1.0),
            'abstract_thinking': random.uniform(0.2, 1.0)
        }
        
        # Attention checks
        attention_check_passed = random.random() < 0.90  # 90% pass rate for personality assessment
        
        await asyncio.sleep(0.10)  # Scaled simulation time
        
        return {
            'personality_responses': personality_responses,
            'creative_preferences': creative_preferences,
            'mathematical_affinity': mathematical_affinity,
            'attention_check_passed': attention_check_passed,
            'response_times': [random.uniform(2.0, 12.0) for _ in personality_questions],
            'task_engagement': random.uniform(0.8, 1.0)
        }
    
    async def _collect_generic_task_data(self, 
                                       task: DataCollectionTask,
                                       participant: Optional[ParticipantProfile]) -> Dict[str, Any]:
        """Collect generic task data"""
        # Generate basic task data
        responses = [random.uniform(0.0, 1.0) for _ in range(10)]
        
        # Attention check
        attention_check_passed = random.random() < 0.85
        
        await asyncio.sleep(0.05)  # Scaled simulation time
        
        return {
            'responses': responses,
            'attention_check_passed': attention_check_passed,
            'response_times': [random.uniform(1.0, 10.0) for _ in range(10)],
            'task_engagement': random.uniform(0.5, 1.0)
        }
    
    async def _process_task_data(self, raw_data: Dict[str, Any], task: DataCollectionTask) -> Dict[str, Any]:
        """Process raw task data into analysis-ready format"""
        processed_data = {}
        
        # Common processing
        processed_data['task_id'] = task.id
        processed_data['task_type'] = task.task_type
        processed_data['completion_time'] = (task.completion_time - task.start_time).total_seconds() if task.completion_time and task.start_time else 0
        processed_data['attention_check_passed'] = raw_data.get('attention_check_passed', False)
        
        # Task-specific processing
        if task.task_type == 'sacred_geometry_evaluation':
            processed_data['mean_geometry_rating'] = np.mean(list(raw_data.get('geometry_ratings', {}).values()))
            processed_data['geometry_preference_variance'] = np.var(list(raw_data.get('geometry_ratings', {}).values()))
            processed_data['mathematical_understanding_score'] = np.mean(list(raw_data.get('mathematical_understanding', {}).values()))
            
        elif task.task_type == 'archetypal_preference_assessment':
            processed_data['archetypal_diversity'] = np.var(list(raw_data.get('style_preferences', {}).values()))
            processed_data['dominant_archetype'] = max(raw_data.get('style_preferences', {}).items(), key=lambda x: x[1])[0]
            processed_data['thematic_coherence'] = np.mean(list(raw_data.get('thematic_resonance', {}).values()))
            
        elif task.task_type == 'creative_output_evaluation':
            processed_data['mean_quality_rating'] = np.mean(raw_data.get('quality_ratings', []))
            processed_data['mean_creativity_score'] = np.mean(raw_data.get('creativity_scores', []))
            processed_data['rating_consistency'] = 1.0 - np.std(raw_data.get('quality_ratings', []))
            processed_data['ranking_validity'] = self._calculate_ranking_validity(raw_data)
            
        elif task.task_type == 'frequency_signature_validation':
            processed_data['personality_consistency'] = self._calculate_personality_consistency(raw_data)
            processed_data['creative_preference_clarity'] = self._calculate_preference_clarity(raw_data)
            processed_data['mathematical_affinity_score'] = np.mean(list(raw_data.get('mathematical_affinity', {}).values()))
        
        # Response time analysis
        response_times = raw_data.get('response_times', [])
        if response_times:
            processed_data['mean_response_time'] = np.mean(response_times)
            processed_data['response_time_variance'] = np.var(response_times)
            processed_data['response_time_outliers'] = len([t for t in response_times if abs(t - np.mean(response_times)) > 2 * np.std(response_times)])
        
        # Engagement metrics
        processed_data['task_engagement'] = raw_data.get('task_engagement', 0.5)
        
        return processed_data
    
    def _calculate_ranking_validity(self, raw_data: Dict[str, Any]) -> float:
        """Calculate validity of preference rankings"""
        quality_ratings = raw_data.get('quality_ratings', [])
        preference_rankings = raw_data.get('preference_rankings', [])
        
        if not quality_ratings or not preference_rankings:
            return 0.0
        
        # Calculate correlation between ratings and rankings
        rating_order = sorted(range(len(quality_ratings)), key=lambda i: quality_ratings[i], reverse=True)
        
        # Calculate Spearman rank correlation
        rank_differences = [abs(rating_order.index(i) - preference_rankings.index(i)) for i in range(len(quality_ratings))]
        validity = 1.0 - (np.mean(rank_differences) / len(quality_ratings))
        
        return max(0.0, min(1.0, validity))
    
    def _calculate_personality_consistency(self, raw_data: Dict[str, Any]) -> float:
        """Calculate consistency of personality responses"""
        personality_responses = raw_data.get('personality_responses', {})
        
        if not personality_responses:
            return 0.0
        
        # Group responses by trait
        trait_groups = {}
        for question, response in personality_responses.items():
            trait = question.split('_')[0]
            if trait not in trait_groups:
                trait_groups[trait] = []
            trait_groups[trait].append(response)
        
        # Calculate consistency within each trait
        consistencies = []
        for trait, responses in trait_groups.items():
            if len(responses) > 1:
                consistency = 1.0 - (np.std(responses) / 3.0)  # Normalize by expected std
                consistencies.append(max(0.0, min(1.0, consistency)))
        
        return np.mean(consistencies) if consistencies else 0.0
    
    def _calculate_preference_clarity(self, raw_data: Dict[str, Any]) -> float:
        """Calculate clarity of creative preferences"""
        creative_preferences = raw_data.get('creative_preferences', {})
        
        if not creative_preferences:
            return 0.0
        
        # Check for clear, consistent preferences
        clarity_indicators = []
        
        # Check if preferences are well-defined (not all neutral)
        if 'poetry_style_preference' in creative_preferences:
            clarity_indicators.append(1.0)  # Clear choice made
        
        if 'inspiration_sources' in creative_preferences:
            sources = creative_preferences['inspiration_sources']
            if len(sources) > 1 and len(sources) < 6:  # Not too few or too many
                clarity_indicators.append(1.0)
            else:
                clarity_indicators.append(0.5)
        
        return np.mean(clarity_indicators) if clarity_indicators else 0.0
    
    def _calculate_task_quality_metrics(self, 
                                      raw_data: Dict[str, Any],
                                      processed_data: Dict[str, Any],
                                      task: DataCollectionTask) -> Dict[str, float]:
        """Calculate quality metrics for a task"""
        quality_metrics = {}
        
        # Attention check score
        quality_metrics['attention_check_score'] = 1.0 if raw_data.get('attention_check_passed', False) else 0.0
        
        # Response time quality
        response_times = raw_data.get('response_times', [])
        if response_times:
            mean_time = np.mean(response_times)
            expected_time = task.expected_duration.total_seconds() / len(response_times)
            
            # Quality is higher when response times are reasonable (not too fast or slow)
            time_quality = 1.0 - min(1.0, abs(mean_time - expected_time) / expected_time)
            quality_metrics['response_time_quality'] = max(0.0, time_quality)
        else:
            quality_metrics['response_time_quality'] = 0.5
        
        # Engagement quality
        quality_metrics['engagement_quality'] = raw_data.get('task_engagement', 0.5)
        
        # Task-specific quality metrics
        if task.task_type == 'sacred_geometry_evaluation':
            quality_metrics['rating_consistency'] = 1.0 - min(1.0, processed_data.get('geometry_preference_variance', 0.5))
            
        elif task.task_type == 'archetypal_preference_assessment':
            quality_metrics['preference_coherence'] = processed_data.get('thematic_coherence', 0.5)
            
        elif task.task_type == 'creative_output_evaluation':
            quality_metrics['rating_validity'] = processed_data.get('ranking_validity', 0.5)
            quality_metrics['rating_consistency'] = processed_data.get('rating_consistency', 0.5)
            
        elif task.task_type == 'frequency_signature_validation':
            quality_metrics['personality_consistency'] = processed_data.get('personality_consistency', 0.5)
            quality_metrics['preference_clarity'] = processed_data.get('creative_preference_clarity', 0.5)
        
        # Calculate overall quality score
        quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def _assign_quality_flag(self, quality_metrics: Dict[str, float]) -> DataQualityFlag:
        """Assign quality flag based on quality metrics"""
        overall_quality = quality_metrics.get('overall_quality', 0.0)
        
        if overall_quality >= self.quality_thresholds['excellent']:
            return DataQualityFlag.EXCELLENT
        elif overall_quality >= self.quality_thresholds['good']:
            return DataQualityFlag.GOOD
        elif overall_quality >= self.quality_thresholds['acceptable']:
            return DataQualityFlag.ACCEPTABLE
        elif overall_quality >= self.quality_thresholds['poor']:
            return DataQualityFlag.POOR
        else:
            return DataQualityFlag.INVALID
    
    def _calculate_session_quality_metrics(self, session: DataCollectionSession) -> Dict[str, float]:
        """Calculate quality metrics for a complete session"""
        if not session.tasks:
            return {'overall_score': 0.0}
        
        # Aggregate task quality metrics
        task_qualities = []
        attention_check_scores = []
        engagement_scores = []
        
        for task in session.tasks:
            if task.quality_metrics:
                task_qualities.append(task.quality_metrics.get('overall_quality', 0.0))
                attention_check_scores.append(task.quality_metrics.get('attention_check_score', 0.0))
                engagement_scores.append(task.quality_metrics.get('engagement_quality', 0.0))
        
        # Calculate session-level metrics
        session_quality = {
            'overall_score': np.mean(task_qualities) if task_qualities else 0.0,
            'attention_check_pass_rate': np.mean(attention_check_scores) if attention_check_scores else 0.0,
            'average_engagement': np.mean(engagement_scores) if engagement_scores else 0.0,
            'task_completion_rate': len([t for t in session.tasks if t.status == DataCollectionStatus.COMPLETED]) / len(session.tasks),
            'session_duration_appropriateness': self._calculate_duration_appropriateness(session)
        }
        
        return session_quality
    
    def _calculate_duration_appropriateness(self, session: DataCollectionSession) -> float:
        """Calculate how appropriate the session duration was"""
        if not session.total_duration:
            return 0.0
        
        expected_duration = session.session_duration.total_seconds()
        actual_duration = session.total_duration.total_seconds()
        
        # Quality is higher when actual duration is close to expected
        duration_ratio = actual_duration / expected_duration
        
        if 0.8 <= duration_ratio <= 1.2:  # Within 20% of expected
            return 1.0
        elif 0.5 <= duration_ratio <= 2.0:  # Within reasonable bounds
            return 0.5
        else:
            return 0.0
    
    def _calculate_quality_summary(self, sessions: List[DataCollectionSession]) -> Dict[str, Any]:
        """Calculate quality summary for all sessions"""
        if not sessions:
            return {}
        
        # Aggregate quality metrics
        overall_scores = []
        attention_check_rates = []
        engagement_scores = []
        quality_flags = []
        
        for session in sessions:
            if session.overall_quality_score:
                overall_scores.append(session.overall_quality_score)
            
            for task in session.tasks:
                if task.quality_metrics:
                    attention_check_rates.append(task.quality_metrics.get('attention_check_score', 0.0))
                    engagement_scores.append(task.quality_metrics.get('engagement_quality', 0.0))
                
                if task.quality_flag:
                    quality_flags.append(task.quality_flag.value)
        
        # Calculate summary statistics
        quality_summary = {
            'mean_overall_quality': np.mean(overall_scores) if overall_scores else 0.0,
            'quality_distribution': {
                'excellent': len([f for f in quality_flags if f == 'excellent']) / len(quality_flags) if quality_flags else 0,
                'good': len([f for f in quality_flags if f == 'good']) / len(quality_flags) if quality_flags else 0,
                'acceptable': len([f for f in quality_flags if f == 'acceptable']) / len(quality_flags) if quality_flags else 0,
                'poor': len([f for f in quality_flags if f == 'poor']) / len(quality_flags) if quality_flags else 0,
                'invalid': len([f for f in quality_flags if f == 'invalid']) / len(quality_flags) if quality_flags else 0
            },
            'attention_check_pass_rate': np.mean(attention_check_rates) if attention_check_rates else 0.0,
            'mean_engagement': np.mean(engagement_scores) if engagement_scores else 0.0,
            'data_usability_rate': len([f for f in quality_flags if f in ['excellent', 'good', 'acceptable']]) / len(quality_flags) if quality_flags else 0
        }
        
        return quality_summary
    
    def _calculate_completion_statistics(self, sessions: List[DataCollectionSession]) -> Dict[str, Any]:
        """Calculate completion statistics for all sessions"""
        if not sessions:
            return {}
        
        completed_sessions = [s for s in sessions if s.status == DataCollectionStatus.COMPLETED]
        failed_sessions = [s for s in sessions if s.status == DataCollectionStatus.FAILED]
        
        completion_stats = {
            'total_sessions': len(sessions),
            'completed_sessions': len(completed_sessions),
            'failed_sessions': len(failed_sessions),
            'completion_rate': len(completed_sessions) / len(sessions),
            'average_completion_time': np.mean([s.total_duration.total_seconds() for s in completed_sessions if s.total_duration]) if completed_sessions else 0,
            'median_completion_time': np.median([s.total_duration.total_seconds() for s in completed_sessions if s.total_duration]) if completed_sessions else 0
        }
        
        return completion_stats
    
    def _update_collection_statistics(self, sessions: List[DataCollectionSession]):
        """Update global collection statistics"""
        self.collection_stats['total_sessions'] += len(sessions)
        
        completed_sessions = [s for s in sessions if s.status == DataCollectionStatus.COMPLETED]
        failed_sessions = [s for s in sessions if s.status == DataCollectionStatus.FAILED]
        
        self.collection_stats['completed_sessions'] += len(completed_sessions)
        self.collection_stats['failed_sessions'] += len(failed_sessions)
        
        # Update quality statistics
        if completed_sessions:
            quality_scores = [s.overall_quality_score for s in completed_sessions if s.overall_quality_score]
            if quality_scores:
                self.collection_stats['average_quality_score'] = np.mean(quality_scores)
        
        # Update completion time statistics
        completion_times = [s.total_duration.total_seconds() for s in completed_sessions if s.total_duration]
        if completion_times:
            self.collection_stats['average_completion_time'] = np.mean(completion_times)
        
        # Update attention check statistics
        attention_check_results = []
        for session in sessions:
            for task in session.tasks:
                if task.quality_metrics and 'attention_check_score' in task.quality_metrics:
                    attention_check_results.append(task.quality_metrics['attention_check_score'])
        
        if attention_check_results:
            self.collection_stats['attention_check_pass_rate'] = np.mean(attention_check_results)
    
    def _simulate_environmental_conditions(self) -> Dict[str, Any]:
        """Simulate environmental conditions for data collection"""
        return {
            'device_type': random.choice(['desktop', 'laptop', 'tablet', 'mobile']),
            'browser': random.choice(['chrome', 'firefox', 'safari', 'edge']),
            'connection_quality': random.choice(['excellent', 'good', 'fair', 'poor']),
            'time_of_day': random.choice(['morning', 'afternoon', 'evening', 'night']),
            'noise_level': random.choice(['quiet', 'moderate', 'noisy']),
            'lighting': random.choice(['excellent', 'good', 'adequate', 'poor']),
            'interruptions': random.randint(0, 3)
        }
    
    def _get_participant_profile(self, participant_id: str) -> Optional[ParticipantProfile]:
        """Get participant profile (placeholder - would integrate with recruitment system)"""
        # This would integrate with the participant recruitment system
        # For now, return None and use defaults
        return None
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data collection statistics"""
        return {
            'global_stats': self.collection_stats,
            'active_sessions': len(self.active_sessions),
            'completed_sessions': len(self.completed_sessions),
            'quality_thresholds': self.quality_thresholds,
            'task_templates': list(self.task_templates.keys())
        }