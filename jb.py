import jieba

l = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")


for word in l:
    print word.encode('utf-8')
