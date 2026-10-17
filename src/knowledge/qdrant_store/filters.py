"""Payload indexes and strict filter construction for knowledge queries."""

from datetime import date

from qdrant_client import models

from knowledge.models import KnowledgeSourceType

PAYLOAD_INDEXES = {
    "product": models.PayloadSchemaType.KEYWORD,
    "source_type": models.PayloadSchemaType.KEYWORD,
    "language": models.PayloadSchemaType.KEYWORD,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "effective_from_ordinal": models.PayloadSchemaType.INTEGER,
    "effective_to_ordinal": models.PayloadSchemaType.INTEGER,
}


def knowledge_filter(
    source_type: KnowledgeSourceType,
    *,
    product: str | None = None,
    language: str | None = None,
    source_id: str | None = None,
    as_of_date: date | None = None,
) -> models.Filter:
    conditions = [models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type.value))]
    for field, value in (("product", product), ("language", language), ("source_id", source_id)):
        if value is not None:
            conditions.append(models.FieldCondition(key=field, match=models.MatchValue(value=value)))
    if as_of_date is not None:
        for field, bounds in (
            ("effective_from_ordinal", models.Range(lte=as_of_date.toordinal())),
            ("effective_to_ordinal", models.Range(gt=as_of_date.toordinal())),
        ):
            conditions.append(
                models.Filter(
                    should=[
                        models.IsNullCondition(is_null=models.PayloadField(key=field)),
                        models.FieldCondition(key=field, range=bounds),
                    ]
                )
            )
    return models.Filter(must=conditions)
