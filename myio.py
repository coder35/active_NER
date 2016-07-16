# encoding=utf-8
import sys
import io
import re
import json
import pickle

def readsen(filename, is_segment):
	f = io.open(filename, encoding = "utf-8")
	sens = []
	for line in f:
		if is_segment:
			words = [word.strp() for word in jieba.cut(line) if word.strip()]
			sens.append(words)
		else:
			line = re.sub("\d+\.\d+|\d+", "NUM", line)
			chars = list(line)
			i = 0                                                                            
			while i < len(chars):
				if chars[i].islower() or chars[i].isupper():
					start = i
					while i < len(chars) and (chars[i].islower() or chars[i].isupper()):
						i +=1
					end = i
					chars = chars[:start] + ["".join(chars[start:end])]+chars[end:]                          
					i = start

				i += 1
			sens.append(chars)

	return sens


def sen2index(tokendata, dic, default = 0):#[token, token, token] -> [1,3,5]

	indexdata = []
	for item in tokendata:
		senindex = [dic.get(re.sub("\d+\.\d+|\d+", "NUM", token), default) for token in item]
	 	
	 	indexdata.append(senindex)

		
	return indexdata 

def read_weibodata(filename, tokendic, tagdic):
	data = json.load(open(filename))
	indexdata = []

	for chars, tags in data:
		charsindex = [tokendic.get(char,0) for char in chars]
		tagsindex = [tagdic.get(tag,0) for tag in tags]
		indexdata.append([charsindex, tagsindex])

	return indexdata




def read_bosondata(filename, tokendic, tagdic):


	train = io.open(filename, encoding = 'utf-8')
	data = []

	entities = {}
	for line in train:
		line = re.sub("\d+\.\d+|\d+", "NUM", line)
		lpos = line.find("{{")
		if lpos < 0:
			tags = ["NONE"]*len(line)
		else:
			tags = ["NONE"]*lpos
		#print line
		#print lpos
		#print line[lpos]
		while lpos >= 0:
			colon = line.find(":",lpos)
			tag = line[lpos+2: colon]

			entities.setdefault(tag, 0)
			entities[tag] += 1
			rpos = line.find("}}", lpos)
			entity = line[colon + 1:rpos]
			#print line[:lpos]
			#print entity
			line = line[:lpos] + entity + line[rpos+2:]
			tags = tags[:lpos] + [tag]*len(entity)+["NONE"]*(len(line) - lpos - len(entity))
			#print line
				#break
			lpos = line.find("{{")
		i = 0 
		chars = list(line)
		while i < len(chars):
			if chars[i].islower() or chars[i].isupper():
				start = i
				while i < len(chars) and (chars[i].islower() or chars[i].isupper()):
					i +=1
				end = i
				chars = chars[:start] + ["".join(chars[start:end])]+chars[end:]
				tags = tags[:start+1] + tags[end:]
				#print " ".join(chars).encode("utf-8")
				#print " ".join(tags).encode("utf-8")
				assert len(chars) == len(tags)
				i = start
			i += 1
		newchars = []
		newtags = []
		for char, tag in zip(chars, tags):
			if char.strip():
				newchars.append(char.strip())
				if tag == "time":
					newtags.append("NONE")
				else:
					newtags.append(tag)

		data.append((newchars, newtags))


	print entities
	indexdata = []
	for chars, tags in data:
		charsindex = [tokendic.get(char,0) for char in chars]
		tagsindex = [tagdic.get(tag,0) for tag in tags]
		indexdata.append([charsindex, tagsindex])

	return indexdata


def create_ins(data, context_len, neg_label,padding_x, padding_y):#sentence -> token
	
	posdata = []
	negdata = []
	for seq in data:
		px = [padding_x]*context_len + seq[0] + [padding_x]*context_len
		py = [padding_y]*context_len + seq[1] + [padding_y]*context_len

		for i, label in enumerate(seq[1]):
			ins_x = px[i:i+2*context_len+1]
			#ins_y = [padding_y]*context_len + [label] + [padding_y]*context_len
			ins_y = py[i:i+2*context_len+1]
			if label == neg_label:
				negdata.append((ins_x, ins_y))
			else:
				posdata.append((ins_x, ins_y))

	return posdata, negdata

def create_lang_ins(data, context_len, padding_x,padding_y):

	stat = {}
	for seq in data:
		for char in seq:
			stat.setdefault(char, 0)
			stat[char] += 1

	chars = stat.items()

	print "char  num  :  "+str(len(chars))

	chars.sort(lambda x,y: cmp(y[1], x[1]))
	chars = chars[:4000]

	#debug
	#print chars[:10]

	dic = {}
	for index, item in enumerate(chars):
		dic[item[0]] = index + 1



	lang_data = []
	for seq in data:
		px = [padding_x]*context_len + seq + [padding_x]*context_len

		for index in xrange(context_len, len(px) - context_len):
			newx = px[index - context_len: index + context_len + 1]
			newx[len(newx)/2] = padding_x # musk center char
			newy = [padding_y]*context_len + [dic.get(px[index],0)] + [padding_y]*context_len
			lang_data.append((newx,newy))




	return lang_data







def masktarget(data,mask_id):

	for seq in data:
		x = seq[0]
		x[len(x)/2] = mask_id



if __name__ == "__main__":
	#chardata = readsen("weibo_rawdata", False)
	#tokendic = pickle.load(open("chardic"))
	#indexdata = sen2index(chardata, tokendic)
	'''

	data = pickle.load(open(sys.argv[1]))
	chardic = pickle.load(open(sys.argv[2]))
	labeldic = pickle.load(open(sys.argv[3]))

	print "weibo data num  ............. "+str(len(data))
	charindex = sen2index([item[0] for item in data], chardic)
	labelindex = sen2index([item[1] for item in data], labeldic, labeldic["O"])

	indexdata = zip(charindex, labelindex)

	pickle.dump(indexdata, open(sys.argv[4], "w+"))
	'''

	#embeddings= pickle.load(open("charembeddings")) # embedding matrix 
	tokendic = pickle.load(open("chardic"))# (word : index)
	read_bosondata("bosondata.txt", tokendic, {})
