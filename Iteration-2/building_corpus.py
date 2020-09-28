def preprocess(line):
	punctuations = '''!()[]{};:'"\,<>./?@#-$%=^&*_~'''
	no_punc_line=""
	
	line = line.lower() #lowercasing
	
	for letter in line: #Removing Punctuations
		if letter not in punctuations:
			no_punc_line = no_punc_line + letter
		
	return no_punc_line.strip()

def GetCorpus(inputfile_1, corpusfile):
    f1 = open(inputfile_1,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    for line in f1:
        passage = line.strip().lower().split(",")
        for p in passage:
            p=preprocess(p)
            fw.write(p+' ')
        fw.write('\n')
        
              
    f1.close()
    fw.close()
    
GetCorpus('DB.csv', 'corpus')
