#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

# 带音标字符。
phonetic_symbol = {
    "ā": "a1",
    "á": "a2",
    "ǎ": "a3",
    "à": "a4",
    "ē": "e1",
    "é": "e2",
    "ě": "e3",
    "è": "e4",
    "ō": "o1",
    "ó": "o2",
    "ǒ": "o3",
    "ò": "o4",
    "ī": "i1",
    "í": "i2",
    "ǐ": "i3",
    "ì": "i4",
    "ū": "u1",
    "ú": "u2",
    "ǔ": "u3",
    "ù": "u4",
    "ü": "v0",
    "ǘ": "v2",
    "ǚ": "v3",
    "ǜ": "v4",
    "ń": "n2",
    "ň": "n3",
    "": "m2"
}

phonetic_symbol2 = {x : (phonetic_symbol[x][0],int(phonetic_symbol[x][1])) for x in phonetic_symbol}

