"""
Participant Recruitment System for MUSE Validation

This module provides comprehensive participant recruitment capabilities for
validation studies, including demographic targeting, screening, and management.
"""

import asyncio
import json
import logging
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import numpy as np
from scipy import stats


class ParticipantStatus(Enum):
    """Status of participant in recruitment process"""
    INVITED = "invited"
    SCREENED = "screened"
    ELIGIBLE = "eligible"
    ENROLLED = "enrolled"
    ACTIVE = "active"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"
    EXCLUDED = "excluded"


class RecruitmentChannel(Enum):
    """Channels for participant recruitment"""
    ONLINE_PLATFORM = "online_platform"
    SOCIAL_MEDIA = "social_media"
    ACADEMIC_NETWORK = "academic_network"
    PROFESSIONAL_NETWORK = "professional_network"
    DIRECT_OUTREACH = "direct_outreach"
    REFERRAL = "referral"
    ADVERTISEMENT = "advertisement"


@dataclass
class ParticipantProfile:
    """Individual participant profile with demographics and traits"""
    id: str
    demographics: Dict[str, Any]
    personality_traits: Dict[str, float]
    creative_preferences: Dict[str, Any]
    mathematical_background: Dict[str, Any]
    
    # Recruitment tracking
    recruitment_channel: RecruitmentChannel
    recruitment_date: datetime
    status: ParticipantStatus
    
    # Screening results
    screening_score: Optional[float] = None
    eligibility_criteria_met: Dict[str, bool] = None
    
    # Participation history
    experiments_participated: List[str] = None
    completion_rate: float = 0.0
    reliability_score: float = 1.0
    
    # Contact and consent
    contact_info: Dict[str, str] = None
    consent_given: bool = False
    consent_date: Optional[datetime] = None
    
    # Metadata
    created_at: datetime = None
    last_updated: datetime = None


@dataclass
class RecruitmentCriteria:
    """Criteria for participant recruitment"""
    target_sample_size: int
    demographic_requirements: Dict[str, Any]
    personality_requirements: Dict[str, Any]
    experience_requirements: Dict[str, Any]
    exclusion_criteria: Dict[str, Any]
    
    # Recruitment strategy
    preferred_channels: List[RecruitmentChannel]
    recruitment_timeline: timedelta
    incentive_structure: Dict[str, Any]
    
    # Quality requirements
    minimum_completion_rate: float = 0.8
    minimum_reliability_score: float = 0.7
    maximum_previous_participation: int = 3


class ParticipantRecruitmentSystem:
    """
    Comprehensive participant recruitment system for validation studies
    
    Provides realistic simulation of participant recruitment including
    demographic targeting, screening, and management capabilities.
    """
    
    def __init__(self):
        """Initialize the recruitment system"""
        self.logger = logging.getLogger(__name__)
        
        # Participant database
        self.participants: Dict[str, ParticipantProfile] = {}
        self.recruitment_campaigns: Dict[str, Dict[str, Any]] = {}
        
        # Recruitment statistics
        self.recruitment_stats = {
            'total_invited': 0,
            'total_screened': 0,
            'total_enrolled': 0,
            'total_completed': 0,
            'conversion_rates': {},
            'channel_effectiveness': {}
        }
        
        # Demographic distributions for simulation
        self.demographic_distributions = {
            'age': {
                'mean': 35,
                'std': 12,
                'min': 18,
                'max': 65
            },
            'education': {
                'distribution': {
                    'high_school': 0.3,
                    'some_college': 0.2,
                    'college': 0.35,
                    'graduate': 0.15
                }
            },
            'income': {
                'distribution': {
                    'low': 0.25,
                    'medium': 0.5,
                    'high': 0.25
                }
            },
            'gender': {
                'distribution': {
                    'male': 0.48,
                    'female': 0.48,
                    'other': 0.04
                }
            },
            'ethnicity': {
                'distribution': {
                    'white': 0.6,
                    'hispanic': 0.18,
                    'black': 0.13,
                    'asian': 0.06,
                    'other': 0.03
                }
            }
        }
        
        self.logger.info("Participant Recruitment System initialized")
    
    async def recruit_participants(self, 
                                 criteria: RecruitmentCriteria,
                                 experiment_id: str) -> List[ParticipantProfile]:
        """
        Recruit participants for a validation experiment
        
        Args:
            criteria: Recruitment criteria and requirements
            experiment_id: ID of the experiment requiring participants
            
        Returns:
            List of recruited participant profiles
        """
        campaign_id = f"campaign_{experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create recruitment campaign
        campaign = {
            'id': campaign_id,
            'experiment_id': experiment_id,
            'criteria': criteria,
            'start_date': datetime.now(),
            'status': 'active',
            'participants_recruited': [],
            'recruitment_progress': 0.0
        }
        
        self.recruitment_campaigns[campaign_id] = campaign
        
        try:
            # Phase 1: Generate potential participant pool
            potential_participants = await self._generate_participant_pool(criteria)
            
            # Phase 2: Screen participants
            screened_participants = await self._screen_participants(potential_participants, criteria)
            
            # Phase 3: Select eligible participants
            eligible_participants = await self._select_eligible_participants(screened_participants, criteria)
            
            # Phase 4: Enroll participants
            enrolled_participants = await self._enroll_participants(eligible_participants, criteria)
            
            # Update campaign
            campaign['participants_recruited'] = [p.id for p in enrolled_participants]
            campaign['recruitment_progress'] = 1.0
            campaign['status'] = 'completed'
            campaign['end_date'] = datetime.now()
            
            # Update statistics
            self._update_recruitment_statistics(campaign, enrolled_participants)
            
            self.logger.info(f"Recruited {len(enrolled_participants)} participants for experiment {experiment_id}")
            
            return enrolled_participants
            
        except Exception as e:
            campaign['status'] = 'failed'
            campaign['error'] = str(e)
            self.logger.error(f"Recruitment failed for experiment {experiment_id}: {e}")
            raise
    
    async def _generate_participant_pool(self, criteria: RecruitmentCriteria) -> List[ParticipantProfile]:
        """Generate a realistic pool of potential participants"""
        # Generate 3-5x the target sample size to account for screening and dropout
        pool_size = criteria.target_sample_size * random.randint(3, 5)
        
        participants = []
        
        for i in range(pool_size):
            # Generate demographics
            demographics = self._generate_demographics()
            
            # Generate personality traits (Big Five)
            personality_traits = self._generate_personality_traits()
            
            # Generate creative preferences
            creative_preferences = self._generate_creative_preferences()
            
            # Generate mathematical background
            mathematical_background = self._generate_mathematical_background()
            
            # Select recruitment channel
            channel = random.choice(criteria.preferred_channels)
            
            # Create participant profile
            participant = ParticipantProfile(
                id=f"participant_{uuid.uuid4().hex[:8]}",
                demographics=demographics,
                personality_traits=personality_traits,
                creative_preferences=creative_preferences,
                mathematical_background=mathematical_background,
                recruitment_channel=channel,
                recruitment_date=datetime.now(),
                status=ParticipantStatus.INVITED,
                experiments_participated=[],
                completion_rate=random.beta(8, 2),  # Most participants have high completion rates
                reliability_score=random.beta(9, 1),  # Most participants are reliable
                contact_info={
                    'email': f"participant{i}@example.com",
                    'phone': f"+1-555-{random.randint(1000, 9999)}"
                },
                consent_given=False,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            participants.append(participant)
            self.participants[participant.id] = participant
        
        # Simulate recruitment delay
        await asyncio.sleep(0.1)
        
        return participants
    
    def _generate_demographics(self) -> Dict[str, Any]:
        """Generate realistic demographic data"""
        # Age
        age = np.random.normal(
            self.demographic_distributions['age']['mean'],
            self.demographic_distributions['age']['std']
        )
        age = max(self.demographic_distributions['age']['min'], 
                 min(self.demographic_distributions['age']['max'], int(age)))
        
        # Education
        education_dist = self.demographic_distributions['education']['distribution']
        education = np.random.choice(
            list(education_dist.keys()),
            p=list(education_dist.values())
        )
        
        # Income
        income_dist = self.demographic_distributions['income']['distribution']
        income = np.random.choice(
            list(income_dist.keys()),
            p=list(income_dist.values())
        )
        
        # Gender
        gender_dist = self.demographic_distributions['gender']['distribution']
        gender = np.random.choice(
            list(gender_dist.keys()),
            p=list(gender_dist.values())
        )
        
        # Ethnicity
        ethnicity_dist = self.demographic_distributions['ethnicity']['distribution']
        ethnicity = np.random.choice(
            list(ethnicity_dist.keys()),
            p=list(ethnicity_dist.values())
        )
        
        return {
            'age': age,
            'education': education,
            'income': income,
            'gender': gender,
            'ethnicity': ethnicity,
            'location': {
                'country': 'US',
                'state': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']),
                'urban_rural': np.random.choice(['urban', 'suburban', 'rural'], p=[0.4, 0.5, 0.1])
            }
        }
    
    def _generate_personality_traits(self) -> Dict[str, float]:
        """Generate Big Five personality traits"""
        return {
            'openness': np.random.beta(2, 2),
            'conscientiousness': np.random.beta(2, 2),
            'extraversion': np.random.beta(2, 2),
            'agreeableness': np.random.beta(2, 2),
            'neuroticism': np.random.beta(2, 2)
        }
    
    def _generate_creative_preferences(self) -> Dict[str, Any]:
        """Generate creative preferences and experience"""
        return {
            'poetry_style': np.random.choice(['epic', 'lyric', 'narrative', 'free_verse']),
            'emotional_range': np.random.choice(['tragic', 'comedic', 'sacred', 'balanced']),
            'creative_experience': np.random.choice(['none', 'some', 'experienced'], p=[0.4, 0.4, 0.2]),
            'artistic_interests': np.random.choice([
                ['writing', 'reading'],
                ['music', 'performance'],
                ['visual_arts', 'design'],
                ['digital_media', 'technology']
            ]),
            'creative_motivation': np.random.choice(['personal', 'professional', 'academic', 'hobby']),
            'collaboration_preference': np.random.choice(['individual', 'small_group', 'large_group'])
        }
    
    def _generate_mathematical_background(self) -> Dict[str, Any]:
        """Generate mathematical background and affinity"""
        # Mathematical education level
        education_level = np.random.choice(['basic', 'intermediate', 'advanced'], p=[0.5, 0.3, 0.2])
        
        # Mathematical affinity
        affinity = np.random.beta(2, 2)
        
        # Specific interests
        interests = []
        if random.random() < 0.3:
            interests.append('geometry')
        if random.random() < 0.2:
            interests.append('number_theory')
        if random.random() < 0.25:
            interests.append('statistics')
        if random.random() < 0.15:
            interests.append('calculus')
        if random.random() < 0.1:
            interests.append('abstract_algebra')
        
        return {
            'education_level': education_level,
            'mathematical_affinity': affinity,
            'specific_interests': interests,
            'comfort_with_math': np.random.beta(2, 2),
            'problem_solving_style': np.random.choice(['analytical', 'intuitive', 'visual', 'verbal']),
            'mathematical_anxiety': np.random.beta(1, 3)  # Most people have low math anxiety
        }
    
    async def _screen_participants(self, 
                                 participants: List[ParticipantProfile],
                                 criteria: RecruitmentCriteria) -> List[ParticipantProfile]:
        """Screen participants based on criteria"""
        screened = []
        
        for participant in participants:
            # Calculate screening score
            screening_score = self._calculate_screening_score(participant, criteria)
            
            # Update participant
            participant.screening_score = screening_score
            participant.status = ParticipantStatus.SCREENED
            participant.last_updated = datetime.now()
            
            # Check if meets minimum screening threshold
            if screening_score >= 0.6:  # 60% threshold
                screened.append(participant)
        
        # Simulate screening delay
        await asyncio.sleep(0.05)
        
        return screened
    
    def _calculate_screening_score(self, 
                                 participant: ParticipantProfile,
                                 criteria: RecruitmentCriteria) -> float:
        """Calculate screening score based on criteria fit"""
        score = 0.0
        total_weight = 0.0
        
        # Demographic fit
        demo_score = self._score_demographic_fit(participant.demographics, criteria.demographic_requirements)
        score += demo_score * 0.3
        total_weight += 0.3
        
        # Personality fit
        personality_score = self._score_personality_fit(participant.personality_traits, criteria.personality_requirements)
        score += personality_score * 0.25
        total_weight += 0.25
        
        # Experience fit
        experience_score = self._score_experience_fit(participant.creative_preferences, criteria.experience_requirements)
        score += experience_score * 0.2
        total_weight += 0.2
        
        # Reliability and completion rate
        reliability_score = (participant.completion_rate + participant.reliability_score) / 2
        score += reliability_score * 0.15
        total_weight += 0.15
        
        # Previous participation (negative if too many)
        participation_penalty = min(1.0, len(participant.experiments_participated) / criteria.maximum_previous_participation)
        participation_score = 1.0 - participation_penalty
        score += participation_score * 0.1
        total_weight += 0.1
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _score_demographic_fit(self, demographics: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """Score demographic fit"""
        if not requirements:
            return 1.0
        
        score = 0.0
        total_criteria = 0
        
        for criterion, requirement in requirements.items():
            total_criteria += 1
            
            if criterion == 'age_range':
                age = demographics.get('age', 30)
                min_age, max_age = requirement
                if min_age <= age <= max_age:
                    score += 1.0
            elif criterion == 'education_level':
                education = demographics.get('education', 'high_school')
                if education in requirement:
                    score += 1.0
            elif criterion == 'gender':
                gender = demographics.get('gender', 'other')
                if gender in requirement:
                    score += 1.0
            elif criterion == 'location':
                location = demographics.get('location', {})
                if location.get('country') in requirement.get('countries', []):
                    score += 1.0
        
        return score / total_criteria if total_criteria > 0 else 1.0
    
    def _score_personality_fit(self, traits: Dict[str, float], requirements: Dict[str, Any]) -> float:
        """Score personality trait fit"""
        if not requirements:
            return 1.0
        
        score = 0.0
        total_criteria = 0
        
        for trait, requirement in requirements.items():
            if trait in traits:
                total_criteria += 1
                trait_value = traits[trait]
                
                if isinstance(requirement, dict):
                    min_val = requirement.get('min', 0.0)
                    max_val = requirement.get('max', 1.0)
                    
                    if min_val <= trait_value <= max_val:
                        score += 1.0
                elif isinstance(requirement, (int, float)):
                    # Exact match with tolerance
                    if abs(trait_value - requirement) <= 0.2:
                        score += 1.0
        
        return score / total_criteria if total_criteria > 0 else 1.0
    
    def _score_experience_fit(self, preferences: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """Score creative experience fit"""
        if not requirements:
            return 1.0
        
        score = 0.0
        total_criteria = 0
        
        for criterion, requirement in requirements.items():
            total_criteria += 1
            
            if criterion == 'creative_experience':
                experience = preferences.get('creative_experience', 'none')
                if experience in requirement:
                    score += 1.0
            elif criterion == 'artistic_interests':
                interests = preferences.get('artistic_interests', [])
                if any(interest in requirement for interest in interests):
                    score += 1.0
            elif criterion == 'collaboration_preference':
                preference = preferences.get('collaboration_preference', 'individual')
                if preference in requirement:
                    score += 1.0
        
        return score / total_criteria if total_criteria > 0 else 1.0
    
    async def _select_eligible_participants(self, 
                                          screened: List[ParticipantProfile],
                                          criteria: RecruitmentCriteria) -> List[ParticipantProfile]:
        """Select eligible participants from screened pool"""
        # Sort by screening score
        screened.sort(key=lambda p: p.screening_score, reverse=True)
        
        eligible = []
        
        for participant in screened:
            # Check eligibility criteria
            eligibility = self._check_eligibility(participant, criteria)
            
            participant.eligibility_criteria_met = eligibility
            
            if all(eligibility.values()):
                participant.status = ParticipantStatus.ELIGIBLE
                eligible.append(participant)
            else:
                participant.status = ParticipantStatus.EXCLUDED
            
            participant.last_updated = datetime.now()
        
        # Select target sample size from eligible participants
        target_size = min(criteria.target_sample_size, len(eligible))
        selected = eligible[:target_size]
        
        # Simulate selection delay
        await asyncio.sleep(0.03)
        
        return selected
    
    def _check_eligibility(self, participant: ParticipantProfile, criteria: RecruitmentCriteria) -> Dict[str, bool]:
        """Check if participant meets eligibility criteria"""
        eligibility = {}
        
        # Basic requirements
        eligibility['screening_score'] = participant.screening_score >= 0.6
        eligibility['completion_rate'] = participant.completion_rate >= criteria.minimum_completion_rate
        eligibility['reliability_score'] = participant.reliability_score >= criteria.minimum_reliability_score
        eligibility['previous_participation'] = len(participant.experiments_participated) <= criteria.maximum_previous_participation
        
        # Exclusion criteria
        exclusions = criteria.exclusion_criteria
        if exclusions:
            for criterion, exclusion in exclusions.items():
                if criterion == 'age':
                    age = participant.demographics.get('age', 30)
                    eligibility[f'not_{criterion}'] = age not in exclusion
                elif criterion == 'education':
                    education = participant.demographics.get('education', 'high_school')
                    eligibility[f'not_{criterion}'] = education not in exclusion
                elif criterion == 'previous_experiments':
                    prev_experiments = set(participant.experiments_participated)
                    excluded_experiments = set(exclusion)
                    eligibility[f'not_{criterion}'] = not prev_experiments.intersection(excluded_experiments)
        
        return eligibility
    
    async def _enroll_participants(self, 
                                 eligible: List[ParticipantProfile],
                                 criteria: RecruitmentCriteria) -> List[ParticipantProfile]:
        """Enroll eligible participants"""
        enrolled = []
        
        for participant in eligible:
            # Simulate consent process
            consent_probability = self._calculate_consent_probability(participant, criteria)
            
            if random.random() < consent_probability:
                participant.consent_given = True
                participant.consent_date = datetime.now()
                participant.status = ParticipantStatus.ENROLLED
                enrolled.append(participant)
            else:
                participant.status = ParticipantStatus.WITHDRAWN
            
            participant.last_updated = datetime.now()
        
        # Simulate enrollment delay
        await asyncio.sleep(0.05)
        
        return enrolled
    
    def _calculate_consent_probability(self, participant: ParticipantProfile, criteria: RecruitmentCriteria) -> float:
        """Calculate probability of participant giving consent"""
        base_probability = 0.7  # Base consent rate
        
        # Adjust based on incentives
        incentive_structure = criteria.incentive_structure
        if incentive_structure.get('monetary_incentive', 0) > 0:
            base_probability += 0.15
        
        if incentive_structure.get('non_monetary_incentive'):
            base_probability += 0.1
        
        # Adjust based on participant traits
        if participant.personality_traits.get('conscientiousness', 0.5) > 0.7:
            base_probability += 0.1
        
        if participant.creative_preferences.get('creative_motivation') == 'academic':
            base_probability += 0.1
        
        # Adjust based on recruitment channel
        channel_adjustments = {
            RecruitmentChannel.ACADEMIC_NETWORK: 0.1,
            RecruitmentChannel.PROFESSIONAL_NETWORK: 0.05,
            RecruitmentChannel.REFERRAL: 0.15,
            RecruitmentChannel.DIRECT_OUTREACH: 0.05,
            RecruitmentChannel.ONLINE_PLATFORM: 0.0,
            RecruitmentChannel.SOCIAL_MEDIA: -0.05,
            RecruitmentChannel.ADVERTISEMENT: -0.1
        }
        
        base_probability += channel_adjustments.get(participant.recruitment_channel, 0.0)
        
        return min(1.0, max(0.0, base_probability))
    
    def _update_recruitment_statistics(self, campaign: Dict[str, Any], enrolled: List[ParticipantProfile]):
        """Update recruitment statistics"""
        # Update global stats
        self.recruitment_stats['total_invited'] += len(self.participants)
        self.recruitment_stats['total_enrolled'] += len(enrolled)
        
        # Calculate conversion rates
        if len(self.participants) > 0:
            self.recruitment_stats['conversion_rates']['invitation_to_enrollment'] = len(enrolled) / len(self.participants)
        
        # Update channel effectiveness
        for participant in enrolled:
            channel = participant.recruitment_channel.value
            if channel not in self.recruitment_stats['channel_effectiveness']:
                self.recruitment_stats['channel_effectiveness'][channel] = {'enrolled': 0, 'total': 0}
            self.recruitment_stats['channel_effectiveness'][channel]['enrolled'] += 1
        
        # Count total by channel
        for participant in self.participants.values():
            channel = participant.recruitment_channel.value
            if channel not in self.recruitment_stats['channel_effectiveness']:
                self.recruitment_stats['channel_effectiveness'][channel] = {'enrolled': 0, 'total': 0}
            self.recruitment_stats['channel_effectiveness'][channel]['total'] += 1
    
    def get_recruitment_analytics(self) -> Dict[str, Any]:
        """Get comprehensive recruitment analytics"""
        analytics = {
            'overview': {
                'total_participants': len(self.participants),
                'total_campaigns': len(self.recruitment_campaigns),
                'active_campaigns': len([c for c in self.recruitment_campaigns.values() if c['status'] == 'active']),
                'completed_campaigns': len([c for c in self.recruitment_campaigns.values() if c['status'] == 'completed'])
            },
            'conversion_funnel': {
                'invited': len([p for p in self.participants.values() if p.status == ParticipantStatus.INVITED]),
                'screened': len([p for p in self.participants.values() if p.status == ParticipantStatus.SCREENED]),
                'eligible': len([p for p in self.participants.values() if p.status == ParticipantStatus.ELIGIBLE]),
                'enrolled': len([p for p in self.participants.values() if p.status == ParticipantStatus.ENROLLED]),
                'completed': len([p for p in self.participants.values() if p.status == ParticipantStatus.COMPLETED])
            },
            'demographic_breakdown': self._calculate_demographic_breakdown(),
            'channel_performance': self._calculate_channel_performance(),
            'quality_metrics': self._calculate_quality_metrics(),
            'recruitment_stats': self.recruitment_stats
        }
        
        return analytics
    
    def _calculate_demographic_breakdown(self) -> Dict[str, Any]:
        """Calculate demographic breakdown of participants"""
        enrolled_participants = [p for p in self.participants.values() if p.status == ParticipantStatus.ENROLLED]
        
        if not enrolled_participants:
            return {}
        
        breakdown = {
            'age_distribution': {},
            'education_distribution': {},
            'gender_distribution': {},
            'location_distribution': {}
        }
        
        # Age distribution
        ages = [p.demographics.get('age', 30) for p in enrolled_participants]
        breakdown['age_distribution'] = {
            'mean': np.mean(ages),
            'median': np.median(ages),
            'std': np.std(ages),
            'min': np.min(ages),
            'max': np.max(ages)
        }
        
        # Education distribution
        educations = [p.demographics.get('education', 'high_school') for p in enrolled_participants]
        education_counts = {}
        for edu in educations:
            education_counts[edu] = education_counts.get(edu, 0) + 1
        breakdown['education_distribution'] = education_counts
        
        # Gender distribution
        genders = [p.demographics.get('gender', 'other') for p in enrolled_participants]
        gender_counts = {}
        for gender in genders:
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
        breakdown['gender_distribution'] = gender_counts
        
        # Location distribution
        locations = [p.demographics.get('location', {}).get('state', 'unknown') for p in enrolled_participants]
        location_counts = {}
        for location in locations:
            location_counts[location] = location_counts.get(location, 0) + 1
        breakdown['location_distribution'] = location_counts
        
        return breakdown
    
    def _calculate_channel_performance(self) -> Dict[str, Any]:
        """Calculate recruitment channel performance"""
        performance = {}
        
        for channel_name, stats in self.recruitment_stats.get('channel_effectiveness', {}).items():
            if stats['total'] > 0:
                conversion_rate = stats['enrolled'] / stats['total']
                performance[channel_name] = {
                    'total_invited': stats['total'],
                    'enrolled': stats['enrolled'],
                    'conversion_rate': conversion_rate,
                    'effectiveness_score': conversion_rate * stats['enrolled']  # Weighted by volume
                }
        
        return performance
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate participant quality metrics"""
        enrolled_participants = [p for p in self.participants.values() if p.status == ParticipantStatus.ENROLLED]
        
        if not enrolled_participants:
            return {}
        
        completion_rates = [p.completion_rate for p in enrolled_participants]
        reliability_scores = [p.reliability_score for p in enrolled_participants]
        screening_scores = [p.screening_score for p in enrolled_participants if p.screening_score is not None]
        
        return {
            'completion_rate': {
                'mean': np.mean(completion_rates),
                'median': np.median(completion_rates),
                'std': np.std(completion_rates)
            },
            'reliability_score': {
                'mean': np.mean(reliability_scores),
                'median': np.median(reliability_scores),
                'std': np.std(reliability_scores)
            },
            'screening_score': {
                'mean': np.mean(screening_scores) if screening_scores else 0,
                'median': np.median(screening_scores) if screening_scores else 0,
                'std': np.std(screening_scores) if screening_scores else 0
            }
        }
    
    def get_participant(self, participant_id: str) -> Optional[ParticipantProfile]:
        """Get participant by ID"""
        return self.participants.get(participant_id)
    
    def update_participant_status(self, participant_id: str, status: ParticipantStatus) -> bool:
        """Update participant status"""
        if participant_id in self.participants:
            self.participants[participant_id].status = status
            self.participants[participant_id].last_updated = datetime.now()
            return True
        return False
    
    def get_campaign_participants(self, campaign_id: str) -> List[ParticipantProfile]:
        """Get participants for a specific campaign"""
        if campaign_id not in self.recruitment_campaigns:
            return []
        
        campaign = self.recruitment_campaigns[campaign_id]
        participant_ids = campaign.get('participants_recruited', [])
        
        return [self.participants[pid] for pid in participant_ids if pid in self.participants]