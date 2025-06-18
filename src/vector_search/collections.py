import logging
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CollectionIndexes:
    name: str
    indexes: List[Tuple[str, str]]


tender_indexes = CollectionIndexes(
    name='tenders',
    indexes=[
        # Core classifications
        ('cpv_code', 'keyword'),
        ('tender_size', 'keyword'),
        ('complexity', 'keyword'),
        ('tender_profile', 'keyword'),

        # Location and language
        ('location_code', 'keyword'),
        ('languages_required', 'keyword'),
        ('requires_physical_presence', 'bool'),

        # Technical requirements
        ('required_tools_or_stack', 'keyword'),
        ('required_certifications', 'keyword'),
        ('certification_level', 'keyword'),

        # Collaboration model
        ('collaboration_type', 'keyword'),
        ('experience_required', 'keyword'),
        ('formality_level', 'keyword'),

        # Status and timeline
        ('status', 'keyword'),
        ('deadline', 'keyword'),
        ('duration_months', 'integer'),
    ]
)

company_indexes = CollectionIndexes(
    name='companies',
    indexes=[
        # Core classifications
        ('cpv_codes', 'keyword'),
        ('size', 'keyword'),
        ('technologies', 'keyword'),

        # Location and language
        ('location', 'keyword'),
        ('location_code', 'keyword'),
        ('languages', 'keyword'),

        # Capabilities and compliance
        ('expertise_areas', 'keyword'),
        ('specializations', 'keyword'),
        ('certifications', 'keyword'),
    ]
)
