from prestie.catalog import COVERED_SPECS, spec_by_id, spec_by_key


# Spec ids and roles as returned by the Blizzard Game Data API
# (/data/wow/playable-specialization/index, Midnight).
GAME_SPECS = {
    62: "DAMAGE", 63: "DAMAGE", 64: "DAMAGE", 65: "HEALER", 66: "TANK", 70: "DAMAGE",
    71: "DAMAGE", 72: "DAMAGE", 73: "TANK", 102: "DAMAGE", 103: "DAMAGE", 104: "TANK",
    105: "HEALER", 250: "TANK", 251: "DAMAGE", 252: "DAMAGE", 253: "DAMAGE",
    254: "DAMAGE", 255: "DAMAGE", 256: "HEALER", 257: "HEALER", 258: "DAMAGE",
    259: "DAMAGE", 260: "DAMAGE", 261: "DAMAGE", 262: "DAMAGE", 263: "DAMAGE",
    264: "HEALER", 265: "DAMAGE", 266: "DAMAGE", 267: "DAMAGE", 268: "TANK",
    269: "DAMAGE", 270: "HEALER", 577: "DAMAGE", 581: "TANK", 1467: "DAMAGE",
    1468: "HEALER", 1473: "DAMAGE", 1480: "DAMAGE",
}
ICY_VEINS_ROLE = {"TANK": "tank", "HEALER": "healing", "DAMAGE": "dps"}


def test_catalog_covers_every_playable_spec_with_its_game_role():
    assert {spec.spec_id for spec in COVERED_SPECS} == set(GAME_SPECS)
    for spec in COVERED_SPECS:
        assert spec.role == ICY_VEINS_ROLE[GAME_SPECS[spec.spec_id]], spec.key


def test_class_tokens_are_the_class_slug_without_dashes_in_upper_case():
    for spec in COVERED_SPECS:
        assert spec.class_token == spec.wow_class.replace("-", "").upper()


def test_atypical_specs_use_the_icy_veins_slugs():
    assert spec_by_id(1473).key == "augmentation-evoker"
    assert spec_by_id(1480).key == "devourer-demon-hunter"
    assert spec_by_id(253).key == "beast-mastery-hunter"


def test_specs_are_found_by_game_id_and_by_key():
    assert spec_by_id(256).name == "Discipline Priest"
    assert spec_by_id(258).role == "dps"
    assert spec_by_key("blood-death-knight").spec_id == 250
    assert spec_by_id(65).name == "Holy Paladin"
    assert spec_by_id(999) is None
    assert spec_by_key("holy-warrior") is None


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
    assert specs_for_class("DEMONHUNTER")[-1].name == "Devourer Demon Hunter"
    assert specs_for_class("NOTACLASS") == ()
