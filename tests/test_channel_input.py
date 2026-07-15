"""Covers channels_store.normalize_channel_input -- /addchannel accepting a
pasted t.me URL (or @name / bare name) and extracting the channel username.
"""

from __future__ import annotations

import pytest

from listener.channels_store import normalize_channel_input


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("https://t.me/ba3bou3_deals", "ba3bou3_deals"),
        ("http://t.me/ba3bou3_deals", "ba3bou3_deals"),
        ("t.me/ba3bou3_deals", "ba3bou3_deals"),
        ("https://www.t.me/ba3bou3_deals", "ba3bou3_deals"),
        ("https://t.me/s/ba3bou3_deals", "ba3bou3_deals"),  # web-preview link
        ("@ba3bou3_deals", "ba3bou3_deals"),
        ("ba3bou3_deals", "ba3bou3_deals"),
        ("  @ba3bou3_deals  ", "ba3bou3_deals"),
        ("https://t.me/ba3bou3_deals/", "ba3bou3_deals"),  # trailing slash
        ("https://t.me/ba3bou3_deals?utm=x", "ba3bou3_deals"),  # query junk
        ("t.me/CouponsEgypt#top", "CouponsEgypt"),  # fragment
    ],
)
def test_normalizes_to_bare_username(raw, expected):
    assert normalize_channel_input(raw) == expected


def test_private_invite_links_returned_untouched():
    # No @username to extract -- hand the whole thing to get_entity as-is.
    assert normalize_channel_input("https://t.me/+AbCdEf123") == "https://t.me/+AbCdEf123"
    assert normalize_channel_input("https://t.me/joinchat/AbCdEf123") == "https://t.me/joinchat/AbCdEf123"
