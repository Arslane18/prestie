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
