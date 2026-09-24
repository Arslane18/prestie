import pytest

from prestie.character.lua_table import LuaParseError, parse_saved_variables


def test_parses_top_level_assignments():
    text = 'FirstDB = {\n["a"] = 1,\n}\nSecondDB = "x"\n'

    assert parse_saved_variables(text) == {"FirstDB": {"a": 1}, "SecondDB": "x"}


def test_parses_scalars():
    text = 'DB = {\n["i"] = -42,\n["f"] = 1.5e3,\n["t"] = true,\n["n"] = false,\n}\n'

    assert parse_saved_variables(text)["DB"] == {
        "i": -42,
        "f": 1500.0,
        "t": True,
        "n": False,
    }


def test_nil_values_are_dropped_like_in_lua():
    assert parse_saved_variables('DB = {\n["gone"] = nil,\n["kept"] = 1,\n}')["DB"] == {
        "kept": 1
    }


def test_sequence_tables_become_lists_in_order():
    text = 'DB = {\n{\n["id"] = 1,\n},\n{\n["id"] = 2,\n},\n}\n'

    assert parse_saved_variables(text)["DB"] == [{"id": 1}, {"id": 2}]


def test_explicit_consecutive_integer_keys_become_a_list():
    assert parse_saved_variables('DB = {\n[2] = "b",\n[1] = "a",\n}')["DB"] == ["a", "b"]


def test_sparse_integer_keys_stay_a_dict():
    assert parse_saved_variables('DB = {\n[1] = "a",\n[5] = "e",\n}')["DB"] == {
        1: "a",
        5: "e",
    }


def test_empty_table_is_an_empty_list():
    # Lua cannot tell an empty array from an empty map; callers normalize.
    assert parse_saved_variables("DB = {\n}")["DB"] == []


def test_decodes_string_escapes_and_utf8():
    text = 'DB = "Quote \\" back \\\\ tab\\t nl\\n dec\\065 La fin d’un prêtre"'

    assert parse_saved_variables(text)["DB"] == (
        'Quote " back \\ tab\t nl\n decA La fin d’un prêtre'
    )


def test_backslash_newline_is_a_newline():
    assert parse_saved_variables('DB = "a\\\nb"')["DB"] == "a\nb"


def test_handles_crlf_and_comments():
    text = 'DB = {\r\n\t"x", -- [1]\r\n\t"y", -- [2]\r\n}\r\n'

    assert parse_saved_variables(text)["DB"] == ["x", "y"]


def test_accepts_identifier_keys_and_semicolons():
    assert parse_saved_variables("DB = { a = 1; b = 2 }")["DB"] == {"a": 1, "b": 2}


def test_empty_file_has_no_variables():
    assert parse_saved_variables("") == {}


@pytest.mark.parametrize(
    "text",
    [
        'DB = {\n["a"] = 1,\n',  # unterminated table
        'DB = "open',  # unterminated string
        "DB = os.exit()",  # code, not data
        "DB = {\n[\"a\"] 1\n}",  # missing '='
        "= 1",  # missing name
    ],
)
def test_rejects_invalid_input(text):
    with pytest.raises(LuaParseError):
        parse_saved_variables(text)


def test_error_reports_the_line_number():
    with pytest.raises(LuaParseError, match="line 3"):
        parse_saved_variables('DB = {\n["a"] = 1,\n["b"] = ?,\n}')
