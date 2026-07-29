from pathlib import Path

import yaml

from fasterrag.config.schema import Settings


def parse_reference_example(config_reference: Path) -> dict[str, object]:
    text = config_reference.read_text(encoding="utf-8")
    _, _, after = text.partition("```yaml\n")
    block, _, _ = after.partition("```")
    parsed = yaml.safe_load(block)
    assert isinstance(parsed, dict)
    return parsed


def test_canonical_config_matches_the_reference_example(
    canonical_config: Path, config_reference: Path
) -> None:
    canonical = yaml.safe_load(canonical_config.read_text(encoding="utf-8"))
    assert canonical == parse_reference_example(config_reference)


def test_canonical_config_states_every_schema_default(canonical_config: Path) -> None:
    canonical = yaml.safe_load(canonical_config.read_text(encoding="utf-8"))
    assert canonical == Settings().model_dump(mode="json")
