import sys
import pr
import jieba
import io

def readsen(filename):
	f = io.open(filename, encoding = "utf-8")
	sens = []
	for line in f:
		words = [word.strp() for word in jieba.cut(line) if word.strip()]
		sens.append(words)


	return sens


def sen2index(rawdata, dic):
	indexdata = []
	for item in rawdata:
		senindex = [dic.get(re.sub("\d+\.\d+|\d+", "NUM", word), 0) for word in item]
	 	senindex = [np.asarray([windex], dtype = np.int32)  for windex in senindex]
	 	indexdata.append(senindex)

		
	return indexdata 

if __name__ == "__main__":
	inputfile = sys.argv[1]
	storedir = sys.argv[2]

	if os.path.exists(storedir):
    	shutil.rmtree(storedir)
	os.mkdir(storedir)


	dim = 128
	em_num = 1
	top_n = 1


	print "loading model ......................."
	embeddic = pickle.load(open("finetune_epoch2_large_mask0.3_fixem/model_dic"))
	wdic = pickle.load(open("finetune_epoch2_large_mask0.3_fixem/tokendic"))
		padding_id = wdic["<padding>"]
	pids = [padding_id]
	loadedmodel = pickle.load(open("finetune_epoch2_large_mask0.3_fixem/model_epoch11_w"))
	tags = pickle.load(open("finetune_epoch2_large_mask0.3_fixem/tags"))
	

	print " loading   raw  data ..................."
	rawdata = readsen(inputfile)
	indexdata = sen2index(rawdata. wdic)

	net_size = [dim, 100*2,70*2, len(tags)]
	print "model  initialization ..............network size  "+str(net_size)
	model = semimodel(len(tags),em_num, net_size,dropout = 0, embeddic = embeddic, premodel = loadedmodel)

    print "model evaluating preparing.........................."
    model.evaluate_ready()

    predict  = []

    for item in indexdata:
    	predict.append([ins[0] for ins in model.decode(item, top_n)])

    extracted_ens = pr.extracted_ens(tags, predict)

    outputfile = open(storedir+"/select_sentence", 'w+')

    for index, item in enumerate(predict):
    	if len(item) > 0:
    		outputfile.write("".join(rawdata[index]))


    outputfile.close()




	        


