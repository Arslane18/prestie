from prestie.catalog import COVERED_SPECS, spec_by_id, spec_by_key


def test_catalog_covers_blood_dk_and_the_three_priest_specs():
    assert [spec.key for spec in COVERED_SPECS] == [
        "blood-death-knight",
        "discipline-priest",
        "holy-priest",
        "shadow-priest",
    ]


def test_specs_are_found_by_game_id_and_by_key():
    assert spec_by_id(256).name == "Discipline Priest"
    assert spec_by_id(258).role == "dps"
    assert spec_by_key("blood-death-knight").spec_id == 250
    assert spec_by_id(65) is None  # Holy Paladin: not covered
    assert spec_by_key("holy-paladin") is None


def test_keys_and_ids_are_unique():
    assert len({spec.key for spec in COVERED_SPECS}) == len(COVERED_SPECS)
    assert len({spec.spec_id for spec in COVERED_SPECS}) == len(COVERED_SPECS)


def test_class_guides_are_found_by_the_addon_class_token():
    from prestie.catalog import specs_for_class

    assert [spec.key for spec in specs_for_class("PRIEST")] == [
        "discipline-priest",
        "holy-priest",
        "shadow-priest",
    ]
    assert specs_for_class("MAGE") == ()
