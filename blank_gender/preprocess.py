"Preprocess takes the dataset and recognizes the male and female pronouns and names. Each paragraph is then saved in a corresponding file: "
" male.txt for all sentences containing male protagonists, female.txt for females and unresolved.txt for uncertain/unresolved text."

from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import sys



f_name = sys.argv[1]

f = open(f_name, "r")

txt  = []
a = 0
for i in f.readlines()[1:]:
    # print(i)
    tmp = i.split(',')
    # print(tmp)

    # # for a in tmp:
    # #     print('ZIN', a)
    # # print(tmp[1:])
    # # print()
    # # br
    # print(tmp[1])
    # br
    try:
        txt.append(tmp[1])
    except:
        a += 1
print(len(txt))

print(txt[:5])

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz",cuda_device=-1)
# predictor._model = predictor._model.cuda()
print(predictor.cuda_device)
print("predictor loaded")

male = []
female = []
unresolved = []
def getPron(para):
    pron = []
    pron_p = []
    output = predictor.predict(
        document=para
    )

    if output['clusters'] != []:
        cluster_len = [len(i) for i in output['clusters']]
        for tup in output['clusters'][cluster_len.index(max(cluster_len))]:

            s = tup[0]
            t = tup[1]
            if s==t:
                pron_p.append((s,s))
            else:
                pron_p.append((s,t+1))
            pron.append(output['document'][s:t + 1])
    else:
        return None, None

    return pron, pron_p

malePron = ["He","he","him","Him"]
femalePron = ["She","she","Her","her"]

cnt = 0
for para in txt:

    try:
        # print('para', para)
        pron,pron_p = getPron(para)
        # print(pron, pron_p)
        pron = [" ".join(i) for i in pron]
        if bool(set(pron) & set(malePron)) and not bool(set(pron) & set(femalePron)) :
            a = open('male.txt', 'a')
            a.write(para+"\n")
            a.close()
        elif bool(set(pron) & set(femalePron)) and not bool(set(pron) & set(malePron)):
            a = open('female.txt', 'a')
            a.write(para+"\n")
            a.close()
        else:
            a = open('unresolved.txt', 'a')
            a.write(para+"\n")
            a.close()

        cnt +=1
        if (cnt%100==0):
            print(cnt)
    except:
        continue
