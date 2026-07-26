"""Tests for label harmonization."""
import pytest

from app.training.harmonize_labels import (
    harmonize_label,
    is_anemia,
    remap_tree,
    write_label_mapping_yaml,
    split_train_val_test,
)


def test_harmonize_label_lowercase():
    assert harmonize_label("Anemic") == "anemia"
    assert harmonize_label("YES") == "anemia"
    assert harmonize_label("No") == "normal"
    assert harmonize_label("normal") == "normal"


def test_harmonize_label_unknown_raises():
    with pytest.raises(ValueError, match="Unknown label"):
        harmonize_label("purple")


def test_is_anemia_handles_synonyms():
    assert is_anemia("anemia")
    assert is_anemia("anemic")
    assert is_anemia("Yes")
    assert not is_anemia("No")
    assert not is_anemia("normal")


def test_remap_tree_groups_by_canonical_label(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    (src / "Anemic").mkdir(parents=True)
    (src / "Anemic" / "a.png").touch()
    (src / "Normal").mkdir()
    (src / "Normal" / "n.png").touch()

    mapping = remap_tree(str(src), str(dst))

    assert (dst / "anemia" / "a.png").exists()
    assert (dst / "normal" / "n.png").exists()
    assert mapping == {"a.png": "anemia", "n.png": "normal"}


def test_write_label_mapping_yaml(tmp_path):
    rules = {"anemia": ["anemia", "anemic", "yes"], "normal": ["normal", "no"]}
    path = tmp_path / "LABEL_MAPPING.yaml"
    write_label_mapping_yaml(rules, str(path))
    content = path.read_text()
    assert "anemia" in content
    assert "yes" in content


def test_split_train_val_test(tmp_path):
    src = tmp_path / "canonical"
    (src / "anemia").mkdir(parents=True)
    (src / "normal").mkdir(parents=True)
    for i in range(20):
        (src / "anemia" / f"a{i}.jpg").touch()
    for i in range(20):
        (src / "normal" / f"n{i}.jpg").touch()
    dst = tmp_path / "split"
    counts = split_train_val_test(str(src), str(dst))
    assert counts["train"]["anemia"] == 16  # 80%
    assert counts["val"]["anemia"] == 2     # 10%
    assert counts["test"]["anemia"] == 2    # 10%
