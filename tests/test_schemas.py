import json

from dokulanc.config import CLASSES
from dokulanc.schemas import SCHEMAS, empty_result, inline_schema, scalar_fields


def test_every_class_has_schema():
    assert set(SCHEMAS) == set(CLASSES)


def test_inline_schema_has_no_refs_and_requires_all():
    for model in SCHEMAS.values():
        schema = inline_schema(model)
        text = json.dumps(schema)
        assert "$ref" not in text and "$defs" not in text

        def walk(node):
            if isinstance(node, dict):
                if node.get("type") == "object" and "properties" in node:
                    assert set(node["required"]) == set(node["properties"])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for v in node:
                    walk(v)

        walk(schema)


def test_empty_result_validates():
    for t, model in SCHEMAS.items():
        model.model_validate(empty_result(t))
        assert scalar_fields(t)
