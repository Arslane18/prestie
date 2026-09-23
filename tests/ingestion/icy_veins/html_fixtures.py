"""Synthetic HTML mimicking the Icy Veins guide template.

Real pages stay out of the repository (copyrighted content, see CLAUDE.md), so
tests rebuild the relevant markup patterns by hand.
"""


def spell(spell_id: int, name: str) -> str:
    return (
        '<span class="spell_icon_span"><img class="spell_icon" src="x.jpg"/> '
        f'<span class="spell" data-wowhead="spell={spell_id}">{name}</span></span>'
    )


def item(item_id: int, name: str) -> str:
    return (
        '<span class="spell_icon_span"><img class="spell_icon" src="x.jpg"/> '
        f'<span class="q4" data-wowhead="item={item_id}">{name}</span></span>'
    )


def heading(level: int, number: str, text: str, anchor: str | None = None) -> str:
    id_attr = f' id="{anchor}"' if anchor else ""
    return (
        f'<div class="heading_container heading_number_{level}">'
        f"<span>{number}</span> <h{level}{id_attr}>{text}</h{level}></div>"
    )


def page(content: str, *, title: str = "Blood Death Knight Tank Guide — 12.1") -> str:
    return f"""<html><head>
<title>Blood Death Knight Tank Guide - 12.1 - World of Warcraft - Icy Veins</title>
</head><body>
<div class="guide-header__authors">
  <div class="guide-header__author">Mandl</div>
  <div class="guide-header__author">Panthea</div>
</div>
<div class="guide-header__updated">
  <span class="guide-header__updated-label">Last Updated:</span>
  <span class="guide-header__updated-date">Aug 10, 2026 - 7:20 PM</span>
</div>
<main><div class="container">
<h1>{title}</h1>
<div class="guide-page-content guide-page-content--wow-retail">
<div class="table-of-contents"><a href="#x">Table of contents entry</a></div>
{content}
<div class="changelog_wrapper"><p>21 Aug. 2026: Adjusted things.</p></div>
<section class="internal-links"><h3>In The Same Category</h3></section>
<div class="cta-block__wrapper-new"><p>Support us on Patreon</p></div>
<script>var tracking = 1;</script>
</div></div></main></body></html>"""
