from prestie.ingestion.icy_veins.pages import BLOOD_DK_PAGES, GuidePage


def test_url_is_built_from_slug():
    page = GuidePage(slug="blood-death-knight-leveling-guide", content_type="leveling")

    assert page.url == "https://www.icy-veins.com/wow/blood-death-knight-leveling-guide"


def test_blood_dk_catalog_covers_mvp_pages_without_duplicates():
    slugs = [page.slug for page in BLOOD_DK_PAGES]

    assert len(slugs) == 8
    assert len(set(slugs)) == len(slugs)
    assert "blood-death-knight-pve-tank-guide" in slugs
    assert "blood-death-knight-pve-tank-stat-priority" in slugs


def test_blood_dk_pages_are_tagged_with_class_and_spec():
    assert all(page.wow_class == "death-knight" for page in BLOOD_DK_PAGES)
    assert all(page.spec == "blood" for page in BLOOD_DK_PAGES)


def test_generated_blood_dk_pages_keep_the_original_slugs_and_types():
    assert {(page.slug, page.content_type) for page in BLOOD_DK_PAGES} == {
        ("blood-death-knight-pve-tank-guide", "overview"),
        ("blood-death-knight-leveling-guide", "leveling"),
        ("blood-death-knight-pve-tank-rotation-cooldowns-abilities", "rotation"),
        ("blood-death-knight-pve-tank-stat-priority", "stat_priority"),
        ("blood-death-knight-pve-tank-spec-builds-talents", "talents"),
        ("blood-death-knight-pve-tank-spell-summary", "mechanics"),
        ("blood-death-knight-pve-tank-easy-mode", "beginner"),
        ("blood-death-knight-pve-tank-mythic-plus-tips", "mythic_plus"),
    }


def test_priest_pages_follow_the_same_structure_with_their_role():
    from prestie.catalog import spec_by_key
    from prestie.ingestion.icy_veins.pages import guide_pages

    shadow = {page.slug: page for page in guide_pages(spec_by_key("shadow-priest"))}
    holy = {page.slug for page in guide_pages(spec_by_key("holy-priest"))}

    assert "shadow-priest-pve-dps-stat-priority" in shadow
    assert "shadow-priest-leveling-guide" in shadow
    assert "holy-priest-pve-healing-spec-builds-talents" in holy
    page = shadow["shadow-priest-pve-dps-stat-priority"]
    assert (page.wow_class, page.spec, page.content_type) == (
        "priest",
        "shadow",
        "stat_priority",
    )


def test_all_pages_cover_every_spec_of_the_catalog_once():
    from prestie.catalog import COVERED_SPECS
    from prestie.ingestion.icy_veins.pages import ALL_PAGES

    slugs = [page.slug for page in ALL_PAGES]
    assert len(slugs) == 8 * len(COVERED_SPECS)
    assert len(set(slugs)) == len(slugs)
