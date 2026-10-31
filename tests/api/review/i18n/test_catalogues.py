"""Full TOML semantics, complete catalogue precedence and FR/EN parity."""

import pytest

from frontend.tooling.compile_locales import compile_catalogues


def write(root, name, text):
    (root / name).write_text(text, encoding="utf-8")


def test_uses_full_toml_semantics_including_literal_and_multiline_strings(tmp_path):
    write(tmp_path, "fr.toml", '[messages]\n"A.B = # {count}" = """Éléments {count}\nDeuxième ligne"""\n')
    write(tmp_path, "en.toml", "[messages]\n'A.B = # {count}' = '''Items {count}\nSecond line'''\n")
    result = compile_catalogues(tmp_path)
    assert result["fr"]["A.B = # {count}"] == "Éléments {count}\nDeuxième ligne"
    assert result["en"]["A.B = # {count}"] == "Items {count}\nSecond line"


def test_missing_translation_key_is_an_explicit_build_error(tmp_path):
    write(tmp_path, "fr.toml", '[messages]\n"Available" = "Disponible"\n')
    write(tmp_path, "en.toml", '[messages]\n"Available" = "Available"\n"Missing" = "Missing"\n')
    with pytest.raises(ValueError, match="key mismatch.*Missing"):
        compile_catalogues(tmp_path)


def test_interpolation_placeholders_must_match_between_languages(tmp_path):
    write(tmp_path, "fr.toml", '[messages]\n"Items {count}" = "Éléments"\n')
    write(tmp_path, "en.toml", '[messages]\n"Items {count}" = "Items {count}"\n')
    with pytest.raises(ValueError, match="placeholders differ"):
        compile_catalogues(tmp_path)


def test_non_string_values_are_rejected(tmp_path):
    write(tmp_path, "fr.toml", '[messages]\n"Count" = 0\n')
    write(tmp_path, "en.toml", '[messages]\n"Count" = "Count"\n')
    with pytest.raises(ValueError, match="must be strings"):
        compile_catalogues(tmp_path)


def test_duplicate_keys_are_rejected_by_the_real_toml_parser(tmp_path):
    write(tmp_path, "fr.toml", '[messages]\n"Yes" = "Oui"\n"Yes" = "Oui"\n')
    write(tmp_path, "en.toml", '[messages]\n"Yes" = "Yes"\n')
    with pytest.raises(ValueError, match="overwrite"):
        compile_catalogues(tmp_path)


def test_complete_catalogues_take_precedence_over_temporary_fragments(tmp_path):
    for locale in ("fr", "en"):
        write(tmp_path, f"{locale}.toml", '[messages]\n"Source" = "Source"\n')
        write(tmp_path, f"common.{locale}.toml", "not valid temporary TOML")
    assert compile_catalogues(tmp_path) == {"fr": {"Source": "Source"}, "en": {"Source": "Source"}}


def test_a_partial_final_catalogue_pair_is_not_silently_combined_with_fragments(tmp_path):
    write(tmp_path, "fr.toml", '[messages]\n"Source" = "Source"\n')
    write(tmp_path, "common.en.toml", '[messages]\n"Source" = "Source"\n')
    with pytest.raises(ValueError, match="must exist together"):
        compile_catalogues(tmp_path)


def test_fragments_merge_deterministically_while_retaining_key_parity(tmp_path):
    write(tmp_path, "common.fr.toml", '[messages]\n"Evidence" = "Preuves"\n')
    write(tmp_path, "common.en.toml", '[messages]\n"Evidence" = "Evidence"\n')
    write(tmp_path, "narratives.fr.toml", '[messages]\n"Evidence" = "Éléments probants"\n')
    write(tmp_path, "narratives.en.toml", '[messages]\n"Evidence" = "Evidence"\n')
    assert compile_catalogues(tmp_path)["fr"]["Evidence"] == "Éléments probants"
