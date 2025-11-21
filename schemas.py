"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogs" collection
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any

# Example schemas (replace with your own):

class User(BaseModel):
    """
    Users collection schema
    Collection name: "user" (lowercase of class name)
    """
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    """
    Products collection schema
    Collection name: "product" (lowercase of class name)
    """
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Brand analysis schemas

class Review(BaseModel):
    source: Optional[str] = Field(None, description="Where the review was found")
    author: Optional[str] = None
    rating: Optional[float] = Field(None, ge=0, le=5)
    title: Optional[str] = None
    text: Optional[str] = None
    url: Optional[str] = None

class SentimentScores(BaseModel):
    positive: float
    negative: float
    neutral: float
    overall: float

class SEOBasics(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    keywords: List[str] = []
    h1: List[str] = []
    h2: List[str] = []
    h3: List[str] = []

class BrandAnalysis(BaseModel):
    url: HttpUrl
    domain: str
    favicon: Optional[str] = None
    tech: List[str] = []
    social_links: Dict[str, str] = {}
    color_palette: List[str] = []
    seo: SEOBasics
    keywords: List[str]
    reviews: List[Review] = []
    sentiment: SentimentScores
    summary: str
    raw_samples: Dict[str, Any] = {}
