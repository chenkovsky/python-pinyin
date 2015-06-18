from __future__ import unicode_literals
__author__ = 'chenkovsky'

#如果分词后的单词里面有[... "朝阳", "群众" ...],那么会按照当前的ngram来标注
#如果分词后的单词里面有[... "....朝阳群众..." ...],且字典里面没有词语正好匹配，那么会按照当前的ngram来标注

phrases_ngram = {
    ("朝阳", "群众"): [[['cháo'],['yáng']], [['qún'], ['zhòng']]]
}