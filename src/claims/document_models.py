from decimal import Decimal

from pydantic import BaseModel, Field

from claims.enums import DocumentType


class MoneyAmount(BaseModel):
    amount: Decimal = Field(ge=0)
    currency: str = Field(
        min_length=3,
        max_length=3,
        pattern=r"^[A-Z]{3}$",
    )

class QuoteLineItem(BaseModel):
    description: str
    price: MoneyAmount

class DocumentSource(BaseModel):
    document_type: DocumentType
    filename: str
    page: int = Field(ge=1)

class GarageQuoteExtraction(BaseModel):
    garage: str
    claim_id: str

    parts: tuple[QuoteLineItem, ...]
    labor: MoneyAmount
    total: MoneyAmount

    source: DocumentSource