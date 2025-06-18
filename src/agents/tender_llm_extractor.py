import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

from dacite import from_dict, Config
from llm_adapters import model_config
from llm_adapters.llm_adapter import LLMClientFactory
from transformers import GPT2TokenizerFast

from src.documents_parser import ParsedDocumentData
from src.utils.helpers import empty_dict_from_dataclass, chunk_text_for_user_prompt, dataclass_to_openai_schema

logger = logging.getLogger(__name__)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


class TenderComplexity(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TenderSize(str, Enum):
    XS = "xs"  # < 10k EUR
    S = "s"  # 10k - 100k EUR
    M = "m"  # 100k - 1M EUR
    L = "l"  # 1M - 10M EUR
    XL = "xl"  # > 10M EUR


class MatchingScore(str, Enum):
    EXCELLENT = "excellent"  # 9-10
    GOOD = "good"  # 7-8
    MODERATE = "moderate"  # 5-6
    POOR = "poor"  # 3-4
    VERY_POOR = "very_poor"  # 1-2


@dataclass
class TenderMatchingProfile:
    """Scoring and matching data for company-tender alignment"""

    # Overall matching scores (0-10 scale)
    technical_complexity_score: Optional[float] = field(
        default=None,
        metadata={
            "description":
                "Technical complexity score from "
                "0-10 (0=very simple, 10=extremely complex)"
        }
    )

    financial_attractiveness_score: Optional[float] = field(
        default=None,
        metadata={
            "description":
                "Financial attractiveness score from "
                "0-10 based on value, payment terms, risk"
        }
    )

    competition_intensity_score: Optional[float] = field(
        default=None,
        metadata={
            "description":
                "Expected competition intensity from 0-10 "
                "(0=low competition, 10=very high competition)"
        }
    )

    urgency_score: Optional[float] = field(
        default=None,
        metadata={
            "description":
                "Urgency score from 0-10 based on timeline and deadline pressure"
        }
    )

    # Categorical assessments
    tender_size_category: Optional[TenderSize] = field(
        default=None,
        metadata={
            "description": f"Size category based on estimated value: {[e.value for e in TenderSize]}",
            "enum": [e.value for e in TenderSize]
        }
    )

    complexity_category: Optional[TenderComplexity] = field(
        default=None,
        metadata={
            "description": "Overall complexity category",
            "enum": [e.value for e in TenderComplexity]
        }
    )

    # Justifications (brief explanations)
    complexity_justification: Optional[str] = field(
        default=None,
        metadata={
            "description":
                "Brief explanation of complexity assessment (2-3 sentences in English)"
        }
    )

    attractiveness_justification: Optional[str] = field(
        default=None,
        metadata={
            "description":
                "Brief explanation of financial attractiveness (2-3 sentences in English)"
        }
    )

    competition_justification: Optional[str] = field(
        default=None,
        metadata={
            "description":
                "Brief explanation of expected competition level (2-3 sentences in English)"
        }
    )

    # Strategic tags for matching
    preferred_company_size: List[str] = field(
        default_factory=list,
        metadata={
            "description": "Preferred company sizes for this tender"

        }
    )

    required_experience_level: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Required experience levels for this tender: "
                "'junior', 'mid', 'senior', 'expert', 'specialist'"
        }
    )

    collaboration_model: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Expected collaboration models: "
                "'individual', 'small_team', 'large_team', 'consortium', 'subcontracting'"
        }
    )


@dataclass
class TenderMatchingProfileCz(TenderMatchingProfile):
    # Justifications (brief explanations)
    complexity_justification: Optional[str] = field(
        default=None,
        metadata={
            "description":
                "Brief explanation of complexity assessment (2-3 sentences in Czech)"
        }
    )

    attractiveness_justification: Optional[str] = field(
        default=None,
        metadata={
            "description":
                "Brief explanation of financial attractiveness (2-3 sentences in Czech)"
        }
    )

    competition_justification: Optional[str] = field(
        default=None,
        metadata={
            "description":
                "Brief explanation of expected competition level (2-3 sentences in Czech)"
        }
    )


@dataclass
class EnhancedSemanticTags:
    """Enhanced tagging system for better vector search matching"""

    # Technology and domain tags
    technology_stack: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Specific technologies, tools, or systems mentioned, for example: "
                "'java', 'python', 'aws', 'oracle', "
                "'autocad', 'sap', 'machinery', 'equipment', etc."
        }
    )

    domain_expertise: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Domain areas or industry sectors: "
                "'healthcare', 'finance', 'government', 'education', 'logistics',"
                " 'construction', 'manufacturing', 'energy', 'transport', etc."
        }
    )

    service_types: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Service types or deliverables: "
                "'development', 'consulting', 'maintenance', 'integration', "
                "'training', 'construction', 'design', 'supply', 'installation', 'research', etc."
        }
    )

    # Methodological and process tags
    methodologies: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Methodologies, standards, or approaches: 'agile', 'scrum', "
                "'waterfall', 'devops', 'lean', 'iso_standards', 'project_management', 'quality_assurance', etc."
        }
    )

    # Compliance and certification tags
    required_certifications: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Required certifications, licenses, or standards: "
                "'iso27001', 'gdpr', 'pci_dss', 'iso9001', 'ce_marking', "
                "'professional_license', 'safety_certification', etc."
        }
    )

    # Multilingual support
    language_requirements: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Language requirements for communication or deliverables: "
                "'czech', 'english', 'german', 'slovak', etc."
        }
    )

    # Geographic and operational tags
    location_preferences: List[str] = field(
        default_factory=list,
        metadata={
            "description":
                "Location or delivery preferences: 'on_site', 'remote', 'hybrid', 'prague',"
                " 'brno', 'nationwide', 'regional', 'international', etc."
        }
    )


@dataclass
class LLMExtractedData:
    """
    Enhanced semantic data optimized for multilingual vector search and company matching.
    """

    # Core summaries (multilingual support)
    executive_summary: str = field(
        metadata={"description": "Concise executive summary in English (3-4 sentences) - key purpose and value"}
    )

    technical_summary: Optional[str] = field(
        default=None,
        metadata={"description": "Technical requirements summary in English (2-3 sentences)"}
    )

    scope_and_deliverables: Optional[str] = field(
        default=None,
        metadata={"description": "Detailed scope description with key deliverables in English"}
    )

    # Enhanced tagging for vector search
    semantic_tags: Optional[EnhancedSemanticTags] = field(
        default=None,
        metadata={"description": "Comprehensive semantic tags for matching"}
    )

    # Matching and scoring
    matching_profile: Optional[TenderMatchingProfile] = field(
        default=None,
        metadata={"description": "Scoring and matching assessment data"}
    )

    # Original fields (enhanced)
    key_technologies_or_skills: List[str] = field(
        default_factory=list,
        metadata={"description": "Specific technologies and skills with standardized names"}
    )

    main_challenges_or_risks: List[str] = field(
        default_factory=list,
        metadata={"description": "Key project risks and challenges"}
    )

    evaluation_criteria_summary: Optional[str] = field(
        default=None,
        metadata={"description": "Key evaluation criteria in English"}
    )

    # Additional context for matching
    target_vendor_profile: Optional[str] = field(
        default=None,
        metadata={"description": "Description of ideal vendor characteristics in English"}
    )

    budget_and_timeline_context: Optional[str] = field(
        default=None,
        metadata={"description": "Budget constraints and timeline pressures context in English"}
    )

    searchable_keywords: List[str] = field(
        default_factory=list,
        metadata={"description": "Key English keywords for vector search optimization"}
    )


@dataclass
class LLMExtractedDataCz(LLMExtractedData):
    # Core summaries (multilingual support)
    executive_summary: str = field(
        metadata={"description": "Concise executive summary in Czech (3-4 sentences) - key purpose and value"}
    )

    technical_summary: Optional[str] = field(
        default=None,
        metadata={"description": "Technical requirements summary in Czech (2-3 sentences)"}
    )

    scope_and_deliverables: Optional[str] = field(
        default=None,
        metadata={"description": "Detailed scope description with key deliverables in Czech"}
    )

    evaluation_criteria_summary: Optional[str] = field(
        default=None,
        metadata={"description": "Key evaluation criteria in Czech"}
    )

    # Additional context for matching
    target_vendor_profile: Optional[str] = field(
        default=None,
        metadata={"description": "Description of ideal vendor characteristics in Czech"}
    )

    budget_and_timeline_context: Optional[str] = field(
        default=None,
        metadata={"description": "Budget constraints and timeline pressures context in Czech"}
    )


@dataclass
class SemanticTenderDetail:
    """
    Enhanced semantic data structure optimized for multilingual vector search and company matching.
    """

    # Enhanced LLM extraction
    llm_extracted: Optional[LLMExtractedData] = field(
        default=None,
        metadata={"description": "Enhanced structured semantic data extracted by the LLM."}
    )

    # Computed scores and metrics
    estimated_value_eur_computed: Optional[float] = field(
        default=None,
        metadata={"description": "Computed estimated value in EUR"}
    )

    # Comprehensive scoring system
    technical_complexity_score: Optional[float] = field(
        default=None,
        metadata={"description": "Technical complexity score (0-10)"}
    )

    financial_attractiveness_score: Optional[float] = field(
        default=None,
        metadata={"description": "Financial attractiveness score (0-10)"}
    )

    market_opportunity_score: Optional[float] = field(
        default=None,
        metadata={"description": "Market opportunity score (0-10)"}
    )

    competition_risk_score: Optional[float] = field(
        default=None,
        metadata={"description": "Competition risk score (0-10)"}
    )

    # Enhanced tagging system
    semantic_tags: List[str] = field(
        default_factory=list,
        metadata={"description": "Primary semantic tags for vector search"}
    )

    technology_tags: List[str] = field(
        default_factory=list,
        metadata={"description": "Technology-specific tags"}
    )

    domain_tags: List[str] = field(
        default_factory=list,
        metadata={"description": "Business domain and industry tags"}
    )

    service_tags: List[str] = field(
        default_factory=list,
        metadata={"description": "Service type and methodology tags"}
    )

    # Data quality metrics
    data_completeness_score: Optional[float] = field(
        default=None,
        metadata={"description": "Score indicating data quality and completeness (0-1)"}
    )

    extraction_confidence_score: Optional[float] = field(
        default=None,
        metadata={"description": "Confidence in LLM extraction quality (0-1)"}
    )


def is_higher_confidence(new_data: Dict[str, Any], existing_data: Dict[str, Any]) -> bool:
    """
    Compares confidence scores between new and existing data for a given key.

    Assumes structure:
    {
        "some_field": "value",
        "some_field_confidence": 0.9,
        ...
    }
    """
    new_conf = new_data.get(f"extraction_confidence_score", 0.0)
    existing_conf = existing_data.get(f"extraction_confidence_score", 0.0)
    return new_conf > existing_conf


@dataclass
class SemanticTenderDetailCz(SemanticTenderDetail):
    llm_extracted: Optional[LLMExtractedDataCz] = field(
        default=None,
        metadata={"description": "Enhanced structured semantic data extracted by the LLM."}
    )


class BaseLLMProcessor:
    def __init__(self):
        self.tokenizer = tokenizer
        self.model_config = None
        self.output_model = None
        self.client = None
        self.function_name = None
        self.function_schema = None

    def _get_system_prompt(self):
        raise NotImplementedError

    def setup(self):
        raise NotImplementedError

    @staticmethod
    def _accumulate_data(new_data: Dict[str, Any], accumulated: Dict[str, Any]):
        # Enhanced merging logic
        for key, value in new_data.items():
            if isinstance(value, list):
                # Merge lists and remove duplicates
                existing = accumulated.get(key, [])
                accumulated[key] = list(set(existing + value))
            elif isinstance(value, dict):
                # Merge dictionaries
                existing = accumulated.get(key, {})
                if existing is None:
                    accumulated[key] = value
                else:
                    existing.update(value)
                    accumulated[key] = existing
            elif isinstance(value, bool):
                # For booleans, use OR logic (if any chunk says True, result is True)
                accumulated[key] = accumulated.get(key, False) or value
            elif value is not None and value != "":
                # For strings and numbers, prefer longer/more detailed content
                existing = accumulated.get(key)
                if existing is None or (
                        isinstance(value, str) and len(value) > len(str(existing))
                ) or is_higher_confidence(new_data, accumulated):
                    accumulated[key] = value
        return accumulated

    def _extract_from_documents(
            self,
            documents_data: List[ParsedDocumentData],
            tender_metadata: Dict[str, Any],
            initial_accumulated: Dict[str, Any]
    ) -> Dict[str, Any]:

        accumulated = initial_accumulated.copy()
        total_calls = 0

        for doc in documents_data:
            logger.info(f"Processing document: {doc.id} - {doc.name}")

            tender_metadata_str = json.dumps(tender_metadata, ensure_ascii=False, indent=2)
            already_extracted = json.dumps(accumulated, ensure_ascii=False, indent=2)

            chunks = chunk_text_for_user_prompt(
                m_config=self.client.config,
                system_prompt=self._get_system_prompt(),
                user_prompt_template=tender_metadata_str + already_extracted,
                full_text=doc.full_text
            )
            logger.info(f"Total chunks: {len(chunks)}")

            for i, chunk in enumerate(chunks):
                user_prompt = f"""
                Tender Metadata:
                {tender_metadata_str}

                Previously Extracted Information (to enrich and build upon):
                {already_extracted}

                Current Document Chunk (Document: {doc.name}, Chunk {i + 1}/{len(chunks)}):
                {chunk}
                """

                logger.info(f"LLM Call #{total_calls + 1} for document {doc.id}, chunk {i + 1}")
                logger.debug(f"Chunk token count: {len(self.tokenizer.encode(chunk))}")

                response = self.client.call_with_functions(
                    system_prompt=self._get_system_prompt(),
                    user_prompt=user_prompt,
                    functions=self.function_schema
                )

                parsed_output = response.choices[0].message.function_call.arguments
                new_data = json.loads(parsed_output)

                # Enhanced merging logic
                for key, value in new_data.items():
                    if isinstance(value, list):
                        # Merge lists and remove duplicates
                        existing = accumulated.get(key, [])
                        accumulated[key] = list(set(existing + value))
                    elif isinstance(value, dict):
                        # Merge dictionaries
                        existing = accumulated.get(key, {})
                        if existing is None:
                            accumulated[key] = value
                        else:
                            existing.update(value)
                            accumulated[key] = existing
                    elif isinstance(value, bool):
                        # For booleans, use OR logic (if any chunk says True, result is True)
                        accumulated[key] = accumulated.get(key, False) or value
                    elif value is not None and value != "":
                        # For strings and numbers, prefer longer/more detailed content
                        existing = accumulated.get(key)
                        if existing is None or (
                                isinstance(value, str) and len(value) > len(str(existing))
                        ) or is_higher_confidence(new_data, accumulated):
                            accumulated[key] = value

            total_calls += 1

        return from_dict(
            data_class=SemanticTenderDetail,
            data=accumulated,
            config=Config(cast=[Enum])
        )

    def process(self, documents: List[Any], metadata: Dict[str, Any]):
        initial_result = empty_dict_from_dataclass(self.output_model)
        return self._extract_from_documents(documents, metadata, initial_result)


class TenderExtractorModel(model_config.ModelConfig):
    MODEL = 'gpt-4o'
    CONTEXT_WINDOW = 128000
    RESPONSE_TOKENS = 4000
    MAX_TOKENS = CONTEXT_WINDOW - RESPONSE_TOKENS
    TEMPERATURE = 0.2
    FREQUENCY_PENALTY = 0.1
    PRESENCE_PENALTY = 0.05


class DetailSummaryModel(model_config.ModelConfig):
    MODEL = 'gpt-4o-mini'
    CONTEXT_WINDOW = 128000
    RESPONSE_TOKENS = 3000
    MAX_TOKENS = CONTEXT_WINDOW - RESPONSE_TOKENS
    TEMPERATURE = 0.2
    FREQUENCY_PENALTY = 0.1
    PRESENCE_PENALTY = 0.05


class TenderExtractorCZ(BaseLLMProcessor):
    def __init__(self):
        super().__init__()
        self.model_config = TenderExtractorModel()
        self.output_model = SemanticTenderDetailCz
        self.function_name = "extract_comprehensive_tender_data"
        self.setup()

    def setup(self):
        """
        Initialize the LLM client and create the OpenAI function call schema using helpers.
        """
        try:
            # Initialize the LLM client with the model configuration
            logger.info(f"Setting up TenderExtractor with model: {self.model_config.MODEL}")
            self.client = LLMClientFactory.create_llm_client(self.model_config)

            # Create the OpenAI function call schema using the helper
            self.function_schema = [
                dataclass_to_openai_schema(
                    cls=self.output_model,
                    function_name=self.function_name,
                    function_desc="Extract comprehensive procurement intelligence and semantic data "
                                  "optimized for multilingual vector search and company-tender matching"
                )
            ]

            logger.info(f"TenderExtractor setup completed successfully")
            logger.debug(f"Function schema created for: {self.function_name}")

        except Exception as e:
            logger.error(f"Failed to setup TenderExtractor: {e}")
            raise

    def _get_system_prompt(self):
        return """You are an expert procurement intelligence analyst specializing in Czech and EU tenders, 
        with focus on MULTILINGUAL VECTOR SEARCH OPTIMIZATION and COMPANY-TENDER MATCHING.

**PRIMARY MISSION:**
Extract comprehensive, multilingual semantic data optimized for vector search and company matching algorithms. 
Your output will power AI-driven tender-company matching systems.

**CRITICAL REQUIREMENTS:**

1. **MULTILINGUAL OPTIMIZATION**
   - Provide summaries in BOTH Czech and English
   - Use Czech for client-facing content, English for technical tags
   - Optimize keywords for multilingual vector search
   - Ensure cultural and linguistic nuance preservation

2. **SCORING AND ASSESSMENT** 
   - Provide numerical scores (0-10 scale) for key dimensions
   - Include brief justifications for each score
   - Assess complexity, attractiveness, competition, urgency
   - Categorize tender size and complexity

3. **ENHANCED SEMANTIC TAGGING**
   - Technology stack with standardized names
   - Domain expertise areas
   - Service types and methodologies
   - Required certifications and compliance
   - Language and location requirements

4. **COMPANY MATCHING INTELLIGENCE**
   - Preferred company sizes and experience levels
   - Collaboration models (individual/team/consortium)
   - Target vendor profile description
   - Competition intensity assessment

5. **VECTOR SEARCH OPTIMIZATION**
   - Extract searchable keywords in Czech and English
   - Use standardized technology and domain terminology
   - Focus on semantic relationships, not just keyword matching
   - Consider synonyms and alternative expressions

**SCORING GUIDELINES:**

**Technical Complexity (0-10):**
- 0-2: Basic administrative/simple tasks
- 3-4: Standard implementation with known technologies
- 5-6: Moderate complexity with some technical challenges
- 7-8: High complexity requiring specialized expertise
- 9-10: Cutting-edge, research-level complexity

**Financial Attractiveness (0-10):**
- Consider: value size, payment terms, risk level, profit potential
- 0-2: Low value, high risk, poor terms
- 5-6: Moderate value with acceptable terms
- 8-10: High value, good terms, low financial risk

**Competition Intensity (0-10):**
- 0-2: Niche requirements, few qualified vendors
- 5-6: Moderate competition with several potential bidders
- 8-10: High competition, many qualified vendors

**OUTPUT REQUIREMENTS:**
- Use function call: extract_comprehensive_tender_data
- Preserve Czech language for summaries and descriptions
- Provide English translations for key summaries
- Use standardized English terms for all tags and categories
- Include confidence scores and justifications
- Focus on actionable intelligence for bid/no-bid decisions

**CONFIDENCE SCORING:**
Rate your extraction confidence (0-1):
- 1.0: Explicitly stated in documents
- 0.8: Clearly inferrable from context  
- 0.6: Reasonable interpretation
- 0.4: Educated guess based on patterns
- 0.2: Low confidence estimate

Extract data that enables AI-powered matching between tenders 
and companies based on capability, capacity, and strategic fit."""
