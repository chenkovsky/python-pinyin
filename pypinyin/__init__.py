#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""汉语拼音转换工具."""

from __future__ import unicode_literals

__title__ = 'pypinyin'
__version__ = '0.6.0'
__author__ = 'mozillazg, 闲耘, chenkovsky'
__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2014 mozillazg, 闲耘, chenkovsky'

__all__ = ['pinyin', 'lazy_pinyin', 'slug',
           'STYLE_NORMAL', 'NORMAL',
           'STYLE_TONE', 'TONE',
           'STYLE_TONE2', 'TONE2',
           'STYLE_INITIALS', 'INITIALS',
           'STYLE_FINALS', 'FINALS',
           'STYLE_FINALS_TONE', 'FINALS_TONE',
           'STYLE_FINALS_TONE2', 'FINALS_TONE2',
           'STYLE_FIRST_LETTER', 'FIRST_LETTER']

from collections import deque
from copy import deepcopy
from itertools import chain
import re
import sys
import marisa_trie

from . import phrases_dict, phonetic_symbol, pinyin_dict, phrases_ngram, zhuyin

py3k = sys.version_info >= (3, 0)
if py3k:
    unicode = str
    callable = lambda x: getattr(x, '__call__', None)

# 词组拼音库
PHRASES_NGRAM = phrases_ngram.phrases_ngram.copy()
PHRASE_NGRAM_LAST_WORD_SET = set([x[-1] for x in PHRASES_NGRAM])
# 词语拼音库
PHRASES_DICT = phrases_dict.phrases_dict.copy()
PHRASES_DICT.update({"".join(t) : sum(v,[]) for t, v in PHRASES_NGRAM.items()})
# 单字拼音库
PINYIN_DICT = pinyin_dict.pinyin_dict.copy()
updated = True
def before_pinyin():
  global updated, PHRASE_TRIE, PHRASES_NGRAM_ORDER
  if updated:
    updated = False
    PHRASE_TRIE = marisa_trie.Trie([x for x in PHRASES_DICT])
    arr = [len(x) for x in PHRASES_NGRAM]
    arr.append(1)
    PHRASES_NGRAM_ORDER = max(arr)
# 声母表
_INITIALS = 'zh,ch,sh,b,p,m,f,d,t,n,l,g,k,h,j,q,x,r,z,c,s,yu,y,w'.split(',')
# 带声调字符与使用数字标识的字符的对应关系，类似： {u'ā': 'a1'}
PHONETIC_SYMBOL = phonetic_symbol.phonetic_symbol.copy()
# 所有的带声调字符
re_phonetic_symbol_source = ''.join(PHONETIC_SYMBOL.keys())
# 匹配带声调字符的正则表达式
RE_PHONETIC_SYMBOL = r'[' + re.escape(re_phonetic_symbol_source) + r']'
# 匹配使用数字标识声调的字符的正则表达式
RE_TONE2 = r'([aeoiuvnm])([0-4])$'

# 拼音风格
PINYIN_STYLE = {
    'NORMAL': 0,          # 普通风格，不带声调
    'TONE': 1,            # 标准风格，声调在韵母的第一个字母上
    'TONE2': 2,           # 声调在拼音之后，使用数字 1~4 标识
    'INITIALS': 3,        # 仅保留声母部分
    'FIRST_LETTER': 4,    # 仅保留首字母
    'FINALS': 5,          # 仅保留韵母部分，不带声调
    'FINALS_TONE': 6,     # 仅保留韵母部分，带声调
    'FINALS_TONE2': 7,    # 仅保留韵母部分，声调在拼音之后，使用数字 1~4 标识
    'ZHU':8               # 台湾注音
}
# 普通风格，不带声调
NORMAL = STYLE_NORMAL = PINYIN_STYLE['NORMAL']
# 标准风格，声调在韵母的第一个字母上
TONE = STYLE_TONE = PINYIN_STYLE['TONE']
# 声调在拼音之后，使用数字 1~4 标识
TONE2 = STYLE_TONE2 = PINYIN_STYLE['TONE2']
# 仅保留声母部分
INITIALS = STYLE_INITIALS = PINYIN_STYLE['INITIALS']
# 仅保留首字母
FIRST_LETTER = STYLE_FIRST_LETTER = PINYIN_STYLE['FIRST_LETTER']
# 仅保留韵母部分，不带声调
FINALS = STYLE_FINALS = PINYIN_STYLE['FINALS']
# 仅保留韵母部分，带声调
FINALS_TONE = STYLE_FINALS_TONE = PINYIN_STYLE['FINALS_TONE']
# 仅保留韵母部分，声调在拼音之后，使用数字 1~4 标识
FINALS_TONE2 = STYLE_FINALS_TONE2 = PINYIN_STYLE['FINALS_TONE2']
# 台湾注音
ZHU = STYLE_ZHU = PINYIN_STYLE['ZHU']


def seg(hans):
    if getattr(seg, 'no_jieba', None):
        return hans
    if seg.jieba is None:
        try:
            import jieba
            seg.jieba = jieba
            return jieba.cut(hans)
        except ImportError:
            seg.no_jieba = True
            return hans
    else:
        return seg.jieba.cut(hans)
seg.jieba = None

def load_phrases_ngram(phrases_ngram):
  """载入用户自定义的词组拼音库
  :param phrases_ngram: 词组拼音库。比如： ``{("朝阳", "群众"): [[['cháo'],['yáng']], [['qún'], ['zhòng']]]}``
  :type phrases_ngram: dict
  """
  PHRASES_NGRAM.update(phrases_ngram)
  PHRASE_NGRAM_LAST_WORD_SET |= set([x[-1] for x in phrases_ngram])
  PHRASES_DICT.update({"".join(t) : v for t, v in phrases_ngram.items()})
  global updated
  updated = True

def load_single_dict(pinyin_dict):
    """载入用户自定义的单字拼音库

    :param pinyin_dict: 单字拼音库。比如： ``{0x963F: u"ā,ē"}``
    :type pinyin_dict: dict
    """
    PINYIN_DICT.update(pinyin_dict)
    global updated
    updated = True


def load_phrases_dict(phrases_dict):
    """载入用户自定义的词语拼音库

    :param phrases_dict: 词语拼音库。比如： ``{u"阿爸": [[u"ā"], [u"bà"]]}``
    :type phrases_dict: dict
    """
    PHRASES_DICT.update(phrases_dict)
    global updated
    updated = True


def initial(pinyin):
    """获取单个拼音中的声母.

    :param pinyin: 单个拼音
    :type pinyin: unicode
    :return: 声母
    :rtype: unicode
    """
    for i in _INITIALS:
        if pinyin.startswith(i):
            return i
    return ''

def zhu(pinyin):
  """将拼音转换成台湾注音
  """
  tone = [0]
  def _replace(m):
      symbol = m.group(0)
      tone[0] = phonetic_symbol.phonetic_symbol2[symbol][1]
      return phonetic_symbol.phonetic_symbol2[symbol][0]
  py = re.sub(RE_PHONETIC_SYMBOL, _replace , pinyin)
  print(tone)
  return zhuyin.hanpin2zhu[py]+zhuyin.zhuyin_tones[tone[0]]

def tag_tone(py, tone):
  if "a" in py:
    return py.replace("a", zhuyin.pinyin_tone_dict["a"][tone])
  if "o" in py:
    return py.replace("o", zhuyin.pinyin_tone_dict["o"][tone])
  if "e" in py:
    return py.replace("e", zhuyin.pinyin_tone_dict["e"][tone])
  if "iu" in py:
    return py.replace("u", zhuyin.pinyin_tone_dict["u"][tone])
  if "ui" in py:
    return py.replace("i", zhuyin.pinyin_tone_dict["i"][tone])
  if "i" in py:
    return py.replace("i", zhuyin.pinyin_tone_dict["i"][tone])
  if "u" in py:
    return py.replace("u", zhuyin.pinyin_tone_dict["u"][tone])
  if "ü" in py:
    return py.replace("ü", zhuyin.pinyin_tone_dict["ü"][tone])
  return py

def pin(zhu):
  """将台湾注音转换成拼音
  """
  if len(zhu) == 0:
    return ""
  #if zhu[-1] in zhuyin.zhuyin_tones:
  return [tag_tone(zhuyin.zhu2hanpin[z[:-1]], zhuyin.zhuyin_tones2num[z[-1]]) for z in zhu.split()]
  #else:
  #  return tag_tone(zhuyin.zhu2hanpin[zhu], 1)

def final(pinyin):
    """获取单个拼音中的韵母.

    :param pinyin: 单个拼音
    :type pinyin: unicode
    :return: 韵母
    :rtype: unicode
    """
    initial_ = initial(pinyin) or None
    if not initial_:
        return pinyin
    return ''.join(pinyin.split(initial_, 1))


def toFixed(pinyin, style):
    """根据拼音风格格式化带声调的拼音.

    :param pinyin: 单个拼音
    :param style: 拼音风格
    :return: 根据拼音风格格式化后的拼音字符串
    :rtype: unicode
    """
    # 声母
    if style == INITIALS:
        return initial(pinyin)
    elif style == ZHU:
        return zhu(pinyin)

    def _replace(m):
        symbol = m.group(0)  # 带声调的字符
        # 不包含声调
        if style in [NORMAL, FIRST_LETTER, FINALS]:
            # 去掉声调: a1 -> a
            return re.sub(RE_TONE2, r'\1', PHONETIC_SYMBOL[symbol])
        # 使用数字标识声调
        elif style in [TONE2, FINALS_TONE2]:
            # 返回使用数字标识声调的字符
            return PHONETIC_SYMBOL[symbol]
        # 声调在头上
        else:
            return symbol

    # 替换拼音中的带声调字符
    py = re.sub(RE_PHONETIC_SYMBOL, _replace, pinyin)

    # 首字母
    if style == FIRST_LETTER:
        py = py[0]
    # 韵母
    elif style in [FINALS, FINALS_TONE, FINALS_TONE2]:
        py = final(py)
    return py


def _handle_nopinyin_char(char, errors='default'):
    """处理没有拼音的字符"""
    if callable(errors):
        return errors(char)

    if errors == 'default':
        return char
    elif errors == 'ignore':
        return None
    elif errors == 'replace':
        return unicode('%x' % ord(char))

def prefix_pinyin(phrases, style, heteronym, errors='default'):
  """前缀匹配表组单词.

  :param phrases: 单词
  :param errors: 指定如何处理没有拼音的字符，详情请参考
                 :py:func:`~pypinyin.pinyin`
  :return: 返回拼音列表，多音字会有多个拼音项
  :rtype: list
  """
  py = []
  s =  phrases
  while len(s) > 0:
    prefixes = sorted([(x,len(x)) for x in PHRASE_TRIE.prefixes(s)], key=lambda x: -x[1])
    #print(prefixes)
    if len(prefixes) == 0:
      prefix = s[0]
      s = s[1:]
    else:
      prefix = prefixes[0][0]
      s = s[len(prefix):]
    py += phrases_pinyin(prefix, style, heteronym, errors, missing_word = None)
  return py

def ngram_pinyin(pinyins, words, style):
  """在非返回多音字的情景下，通过ngram，纠正拼音标注
  :param pys: 当前拼音
  :param words: 所有词
  :param style: 指定拼音风格
  """
  offset = 0
  word_py = []
  queue = deque()
  for word in words:
    word_py.append(pinyins[offset:offset + len(word)])
    offset += len(word)
    queue.append(word)
    if len(queue) > PHRASES_NGRAM_ORDER:
      queue.popleft()
    if word in PHRASE_NGRAM_LAST_WORD_SET:
      word_tuple = tuple(queue)
      for i in range(PHRASES_NGRAM_ORDER, 1, -1):
        if word_tuple in PHRASES_NGRAM:
          #采用ngram里面的标注取代原来的标注
          word_py = word_py[:-len(word_tuple)] + PHRASES_NGRAM[word_tuple]
          break
        word_tuple = word_tuple[1:]
  return sum(word_py,[])


def single_pinyin(han, style, heteronym, errors='default'):
    """单字拼音转换.

    :param han: 单个汉字
    :param errors: 指定如何处理没有拼音的字符，详情请参考
                   :py:func:`~pypinyin.pinyin`
    :return: 返回拼音列表，多音字会有多个拼音项
    :rtype: list
    """
    num = ord(han)
    if num not in PINYIN_DICT:
        py = _handle_nopinyin_char(han, errors=errors)
        return [py] if py else None
    pys = PINYIN_DICT[num].split(",")  # 字的拼音列表
    if not heteronym:
        return [toFixed(pys[0], style)]

    # 输出多音字的多个读音
    # 临时存储已存在的拼音，避免多音字拼音转换为非音标风格出现重复。
    py_cached = {}
    pinyins = []
    for i in pys:
        py = toFixed(i, style)
        if py in py_cached:
            continue
        py_cached[py] = py
        pinyins.append(py)
    return pinyins


def phrases_pinyin(phrases, style, heteronym, errors='default', missing_word = "longest_prefix"):
    """词语拼音转换.

    :param phrases: 词语
    :param errors: 指定如何处理没有拼音的字符
    :param missing_word:当这个词不存在的时候，如何处理，missing_word = "longest_prefix"时，采用最长前缀匹配单词，比如分出了单词朝阳区的。但是单词表只有朝阳区，那么会优先匹配朝阳区，而非一个一个字地标注。
    :return: 拼音列表
    :rtype: list
    """
    py = []
    if phrases in PHRASES_DICT:
        py = deepcopy(PHRASES_DICT[phrases])
        for idx, item in enumerate(py):
            py[idx] = [toFixed(item[0], style=style)]
    else:
        #@TODO 应该来个前缀search
        if missing_word == "longest_prefix":
          return prefix_pinyin(phrases, style=style, heteronym=heteronym,
                                   errors=errors)
        for i in phrases:
            single = single_pinyin(i, style=style, heteronym=heteronym,
                                   errors=errors)
            if single:
                py.append(single)
    return py


def _pinyin(words, style, heteronym, errors):
    re_hans = re.compile(r'''^(?:
                         [\u3400-\u4dbf]    # CJK 扩展 A:[3400-4DBF]
                         |[\u4e00-\u9fff]    # CJK 基本:[4E00-9FFF]
                         |[\uf900-\ufaff]    # CJK 兼容:[F900-FAFF]
                         )+$''', re.X)
    pys = []
    # 初步过滤没有拼音的字符
    if re_hans.match(words):
        pys = phrases_pinyin(words, style=style, heteronym=heteronym,
                             errors=errors)
    else:
        if re.match(r'^[a-zA-Z0-9_]+$', words):
            pys.append([words])
        else:
            for word in words:
                # 字母汉字混合的固定词组（这种情况来自分词结果）
                if not (re_hans.match(word)
                        or re.match(r'^[a-zA-Z0-9_]+$', word)
                        ):
                    py = _handle_nopinyin_char(word, errors=errors)
                    pys.append([py]) if py else None
                else:
                    pys.extend(_pinyin(word, style, heteronym, errors))
    return pys


def pinyin(hans, style=TONE, heteronym=False, errors='default', ret_tokenized = False, ngram = True):
    """将汉字转换为拼音.

    :param hans: 汉字字符串( ``u'你好吗'`` )或列表( ``[u'你好', u'吗']`` ).

                 如果用户安装了 ``jieba`` , 将使用 ``jieba`` 对字符串进行
                 分词处理。可以通过传入列表的方式禁用这种行为。

                 也可以使用自己喜爱的分词模块对字符串进行分词处理,
                 只需将经过分词处理的字符串列表传进来就可以了。
    :type hans: unicode 字符串或字符串列表
    :param style: 指定拼音风格
    :param errors: 指定如何处理没有拼音的字符

                   * ``'default'``: 保留原始字符
                   * ``'ignore'``: 忽略该字符
                   * ``'replace'``: 替换为去掉 ``\\u`` 的 unicode 编码字符串
                     (``u'\\u90aa'`` => ``u'90aa'``)
                   * callable 对象: 回调函数之类的可调用对象。如果 ``erros``
                     参数 的值是个可调用对象，那么程序会回调这个函数:
                     ``func(char)``::

                         def foobar(char):
                             return 'a'
                         pinyin(u'あ', errors=foobar)

    :param heteronym: 是否启用多音字
    :param ret_tokenized: 返回的时候是否返回分词后的词
    :return: 拼音列表
    :rtype: list

    Usage::

      >>> from pypinyin import pinyin
      >>> import pypinyin
      >>> pinyin(u'中心')
      [[u'zh\u014dng'], [u'x\u012bn']]
      >>> pinyin(u'中心', heteronym=True)  # 启用多音字模式
      [[u'zh\u014dng', u'zh\xf2ng'], [u'x\u012bn']]
      >>> pinyin(u'中心', style=pypinyin.INITIALS)  # 设置拼音风格
      [[u'zh'], [u'x']]
      >>> pinyin(u'中心', style=pypinyin.TONE2)
      [[u'zho1ng'], [u'xi1n']]
    """
    before_pinyin()
    # 对字符串进行分词处理
    if isinstance(hans, unicode):
        hans = [x for x in seg(hans)]
    pys = []
    for words in hans:
        pys.extend(_pinyin(words, style, heteronym, errors))
    if ngram and not heteronym:
      pys = ngram_pinyin(pys, hans, style)
    if ret_tokenized:
      return (pys, hans)
    return pys


def slug(hans, style=NORMAL, heteronym=False, separator='-', errors='default'):
    """生成 slug 字符串.

    :param hans: 汉字
    :type hans: unicode or list
    :param style: 指定拼音风格
    :param heteronym: 是否启用多音字
    :param separstor: 两个拼音间的分隔符/连接符
    :param errors: 指定如何处理没有拼音的字符，详情请参考
                   :py:func:`~pypinyin.pinyin`
    :return: slug 字符串.

    ::

      >>> import pypinyin
      >>> pypinyin.slug(u'中国人')
      u'zhong-guo-ren'
      >>> pypinyin.slug(u'中国人', separator=u' ')
      u'zhong guo ren'
      >>> pypinyin.slug(u'中国人', style=pypinyin.INITIALS)
      u'zh-g-r'
    """
    return separator.join(chain(*pinyin(hans, style=style, heteronym=heteronym,
                                        errors=errors)
                                ))


def lazy_pinyin(hans, style=NORMAL, errors='default'):
    """不包含多音字的拼音列表.

    与 :py:func:`~pypinyin.pinyin` 的区别是返回的拼音是个字符串，
    并且每个字只包含一个读音.

    :param hans: 汉字
    :type hans: unicode or list
    :param style: 指定拼音风格
    :param errors: 指定如何处理没有拼音的字符，详情请参考
                   :py:func:`~pypinyin.pinyin`
    :return: 拼音列表(e.g. ``['zhong', 'guo', 'ren']``)
    :rtype: list

    Usage::

      >>> from pypinyin import lazy_pinyin
      >>> import pypinyin
      >>> lazy_pinyin(u'中心')
      [u'zhong', u'xin']
      >>> lazy_pinyin(u'中心', style=pypinyin.TONE)
      [u'zh\u014dng', u'x\u012bn']
      >>> lazy_pinyin(u'中心', style=pypinyin.INITIALS)
      [u'zh', u'x']
      >>> lazy_pinyin(u'中心', style=pypinyin.TONE2)
      [u'zho1ng', u'xi1n']
    """
    return list(chain(*pinyin(hans, style=style, heteronym=False,
                              errors=errors)))
