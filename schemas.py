"""
Pydantic schema definitions for pitch deck analysis.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class IndustryType(str, Enum):
    """Industry categories for pitch decks."""
    FINTECH = "fintech"
    HEALTHCARE = "healthcare"
    EDTECH = "edtech"
    ECOMMERCE = "ecommerce"
    SAAS = "saas"
    MARKETPLACE = "marketplace"
    SOCIAL = "social"
    TRAVEL = "travel"
    FOODTECH = "foodtech"
    TRANSPORTATION = "transportation"
    REAL_ESTATE = "real_estate"
    GAMING = "gaming"
    ENTERPRISE = "enterprise"
    CONSUMER = "consumer"
    BIOTECH = "biotech"
    CLIMATE = "climate"
    OTHER = "other"


class BusinessModel(str, Enum):
    """Business model types."""
    SUBSCRIPTION = "subscription"
    FREEMIUM = "freemium"
    MARKETPLACE = "marketplace"
    ADVERTISING = "advertising"
    TRANSACTION_FEE = "transaction_fee"
    ENTERPRISE_SALES = "enterprise_sales"
    LICENSING = "licensing"
    HARDWARE = "hardware"
    HYBRID = "hybrid"
    OTHER = "other"


class FundingStage(str, Enum):
    """Funding stage categories."""
    PRE_SEED = "pre_seed"
    SEED = "seed"
    SERIES_A = "series_a"
    SERIES_B = "series_b"
    SERIES_C = "series_c"
    SERIES_D_PLUS = "series_d_plus"
    GROWTH = "growth"
    UNKNOWN = "unknown"


class MetricData(BaseModel):
    """Individual metric information."""
    metric_name: Optional[str] = Field(default="Unknown Metric", description="Name of the metric (e.g., 'Monthly Active Users', 'Revenue')")
    value: Optional[str] = Field(default="Not specified", description="The metric value as displayed (e.g., '1M users', '$2M ARR')")
    timeframe: Optional[str] = Field(None, description="Timeframe for the metric (e.g., 'monthly', 'annual')")


class CompetitorInfo(BaseModel):
    """Competitor analysis information."""
    competitor_name: Optional[str] = Field(default="Unknown Competitor", description="Name of the competitor")
    positioning: Optional[str] = Field(None, description="How this competitor is positioned relative to the company")


class TeamMember(BaseModel):
    """Team member information."""
    name: Optional[str] = Field(default="Unknown Team Member", description="Name of the team member")
    role: Optional[str] = Field(default="Unknown Role", description="Role or title")
    background: Optional[str] = Field(None, description="Brief background or previous experience")


class SlideContent(BaseModel):
    """Content analysis for individual slides."""
    slide_number: int = Field(default=1, description="Slide number in the deck")
    slide_type: Optional[str] = Field(default="unknown", description="Type of slide (e.g., 'title', 'problem', 'solution', 'market', 'traction', 'team', 'financials', 'ask')")
    key_points: List[str] = Field(default_factory=list, description="Key points or bullet points from this slide")
    has_charts: bool = Field(default=False, description="Whether the slide contains charts or graphs")
    has_images: bool = Field(default=False, description="Whether the slide contains images or photos")


class PitchDeckAnalysis(BaseModel):
    """Complete analysis of a pitch deck."""
    
    # Basic Information
    company_name: Optional[str] = Field(default="Unknown Company", description="Name of the company")
    tagline: Optional[str] = Field(None, description="Company tagline or one-liner description")
    industry: IndustryType = Field(default=IndustryType.OTHER, description="Primary industry category")
    founded_year: Optional[int] = Field(None, description="Year the company was founded")
    
    # Business Model
    business_model: BusinessModel = Field(default=BusinessModel.OTHER, description="Primary business model")
    target_customer: Optional[str] = Field(default="Not clearly specified", description="Description of target customer or market segment")
    value_proposition: Optional[str] = Field(default="Not clearly specified", description="Core value proposition")
    
    # Problem & Solution
    problem_statement: Optional[str] = Field(default="Not clearly specified", description="Problem the company is solving")
    solution_description: Optional[str] = Field(default="Not clearly specified", description="Description of the solution")
    
    # Market & Competition
    market_size: Optional[str] = Field(None, description="Total addressable market size if mentioned")
    competitors: List[CompetitorInfo] = Field(default_factory=list, description="Key competitors mentioned")
    competitive_advantage: Optional[str] = Field(None, description="Stated competitive advantage or differentiation")
    
    # Traction & Metrics
    key_metrics: List[MetricData] = Field(default_factory=list, description="Key performance metrics")
    customer_testimonials: List[str] = Field(default_factory=list, description="Customer quotes or testimonials")
    partnerships: List[str] = Field(default_factory=list, description="Key partnerships mentioned")
    
    # Team
    team_members: List[TeamMember] = Field(default_factory=list, description="Key team members")
    advisors: List[str] = Field(default_factory=list, description="Advisors or board members mentioned")
    
    # Financials & Funding
    funding_stage: FundingStage = Field(default=FundingStage.UNKNOWN, description="Current funding stage")
    funding_amount_requested: Optional[str] = Field(None, description="Amount of funding requested")
    use_of_funds: List[str] = Field(default_factory=list, description="How funds will be used")
    financial_projections: Optional[str] = Field(None, description="Financial projections if mentioned")
    
    # Slide Analysis
    total_slides: int = Field(default=0, description="Total number of slides in the deck")
    slide_breakdown: List[SlideContent] = Field(default_factory=list, description="Analysis of individual slides")
    
    # Meta Information
    deck_quality: Optional[str] = Field(default="average", description="Overall assessment of deck quality (professional/good/average/poor)")
    key_strengths: List[str] = Field(default_factory=list, description="Key strengths of the pitch")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Areas that could be improved")
    
    # Additional Context
    notes: Optional[str] = Field(None, description="Additional notes or observations")


class ProcessingResult(BaseModel):
    """Result of processing a single PDF."""
    pdf_filename: str = Field(default="unknown.pdf", description="Original PDF filename")
    total_slides: int = Field(default=0, description="Number of slides processed")
    images_generated: List[str] = Field(default_factory=list, description="List of generated image filenames")
    gemini_analysis: Optional[PitchDeckAnalysis] = Field(None, description="Gemini analysis result")
    processing_time: float = Field(default=0.0, description="Total processing time in seconds")
    success: bool = Field(default=False, description="Whether processing was successful")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")


class DatasetEntry(BaseModel):
    """Entry for the fine-tuning dataset."""
    image_path: str = Field(default="", description="Path to the processed image")
    slide_number: int = Field(default=1, description="Slide number")
    company_name: str = Field(default="Unknown Company", description="Company name")
    slide_analysis: SlideContent = Field(default_factory=SlideContent, description="Analysis of this specific slide")
    full_deck_context: PitchDeckAnalysis = Field(default_factory=PitchDeckAnalysis, description="Full deck analysis for context")
    instruction: str = Field(default="Analyze this pitch deck slide", description="Instruction prompt for the model")
    response: str = Field(default="No analysis available", description="Expected response from the model")
