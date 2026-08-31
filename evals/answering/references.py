def build_fact_references(context_passages: list[dict], required_facts: list[dict]) -> list[dict]:
    """Attach each frozen required fact to its frozen supporting passage text."""
    passages_by_id = {passage["id"]: passage["text"] for passage in context_passages}
    references = []

    for required_fact in required_facts:
        supporting_ids = required_fact.get("supporting_passage_ids", [])
        unknown_ids = sorted(set(supporting_ids) - passages_by_id.keys())
        if unknown_ids:
            unknown = ", ".join(unknown_ids)
            raise ValueError(
                f"Required fact {required_fact['id']} used unknown passage IDs: {unknown}"
            )

        references.append({
            "id": required_fact["id"],
            "fact": required_fact["fact"],
            "supporting_passages": [
                {"id": passage_id, "text": passages_by_id[passage_id]}
                for passage_id in supporting_ids
            ],
        })

    return references
