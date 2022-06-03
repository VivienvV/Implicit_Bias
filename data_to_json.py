from base64 import encode
import json
from collections import defaultdict
import random
import checklist
from checklist.editor import Editor
import spacy
import torch
nlp = spacy.load("en_core_web_trf")
import sys


VERSIONstr = '1' 
TARGET = 'name' 
BIAS_TYPE = 'implicit_gender' 
#create dict setup
data_dict = {
    "version" : VERSIONstr,
    "data" : {  "intrasentence" : []}
}

# Read data, split sentences based on '.'
with open('female_masked_pos_train.txt') as f:
    linesfem = f.read().replace('\n', '').split('.')
with open('male_masked_pos_train.txt') as f:
    linesmen = f.read().replace('\n', '').split('.')
lines = linesfem + linesmen

#take out sentences that still contain gendered pronouns
count = 0
for line in lines: 
    if 'her' in line.split() or 'his' in line.split() or len(line.split())<2:
        lines.remove(line)
        count += 1
print(f"{count} sentences removes that still contain his/her pronouns")

##### load names from different sources

#################################
######## CheckList names########
#################################

editor = Editor()
checklist_names = defaultdict(dict)
checklist_names['merged']['males'] = editor.lexicons.male
checklist_names['merged']['females'] = editor.lexicons.female

#################################
### names from Mortage dataset###
#################################
mortage_names = torch.load('../namedict.pt')

#################################
#### singly tokenized names #####
#################################
singly_tokenized_names = torch.load('namedict_token.pt')

#########################
#### gendered nouns #####
#########################
nouns = defaultdict(dict)
nouns['females'] = ['woman', 'daughter', 'mother', 'girl', 'aunt']
nouns['males'] = ['man', 'son', 'father', 'boy', 'uncle' ]

################################
#### gendered nouns + race #####
################################
whitenouns = defaultdict(dict)
whitenouns['females'] = ['white woman', 'white daughter', 'white mother', 'white girl', 'white aunt']
whitenouns['males'] = ['white man', 'white son', 'white father', 'white boy', 'white uncle' ]

blacknouns = defaultdict(dict)
blacknouns['females'] = ['black woman', 'black daughter', 'black mother', 'black girl', 'black aunt']
blacknouns['males'] = ['black man', 'black son', 'black father', 'black boy', 'black uncle' ]

asiannouns = defaultdict(dict)
asiannouns['females'] = ['asian woman', 'asian daughter', 'asian mother', 'asian girl', 'asian aunt']
asiannouns['males'] = ['asian man', 'asian son', 'asian father', 'asian boy', 'asian uncle' ]

hispanicnouns = defaultdict(dict)
hispanicnouns['females'] = ['hispanic woman', 'hispanic daughter', 'hispanic mother', 'hispanic girl', 'hispanic aunt']
hispanicnouns['males'] = ['hispanic man', 'hispanic son', 'hispanic father', 'hispanic boy', 'hispanic uncle' ]

#### functions to fill masked words with gendered continuations


def fill_blank(text, blank_string, names, genders=['males', 'females'], race='merged'):
    '''
    function that replaces masked words from sentences with names from a given dict
    '''
    filled_texts = defaultdict(dict)
    #for race in races:
    for gender in genders:
        filled_texts[race][gender] = text.replace(blank_string, random.choice(names[race][gender]))
    return filled_texts[race]

def fill_pronoun(text, blank_string, genders=['males', 'females'], races=['merged']):
    '''
    function that replaces masked words from sentences with pronouns
    '''
    filled_texts = defaultdict(dict)
    for race in races:
        for gender in genders:
            if gender == 'males':
                poss = 'his'
                subj = 'he'
                obj = 'him'
            if gender == 'females':
                poss = 'her'
                subj = 'she'
                obj = 'her'
            poss_string = text.replace(f"{blank_string}'s", poss)
            doc = nlp(poss_string)
            temp_string = [token for token in doc]
            for i, word in enumerate(doc):
                if word.text == 'ProtagonistA' and (word.dep_ == 'dative' or word.dep_ == 'pobj' or word.dep_ == 'dobj'): 
                    temp_string[i] = obj
                elif word.text == 'ProtagonistA':
                    temp_string[i] = subj
            filled_texts[race][gender] = ' '.join([str(token) for token in temp_string])
    return filled_texts['merged']

def fill_pronoun_noun(text, blank_string, noundict, noun_idx = None, genders=['males', 'females'], races=['merged']):
    '''
    function that replaces masked words from sentences with nouns (object), or prounouns (subject, dative)
    '''
    if noun_idx == None:
        noun_idx = random.randint(0, len(noundict['females'])-1)
    
    filled_texts = defaultdict(dict)
    for race in races:
        for gender in genders:
            if gender == 'males':
                poss = 'his'
                obj = 'him'
            if gender == 'females':
                poss = 'her'
                obj = 'her'
            poss_string = text.replace(f"{blank_string}'s", poss)
            doc = nlp(poss_string)
            temp_string = [token for token in doc]
            for i, word in enumerate(doc):
                if word.text == 'ProtagonistA' and (word.dep_ == 'dative' or word.dep_ == 'pobj' or word.dep_ == 'dobj'): 
                    temp_string[i] = obj
                elif word.text == 'ProtagonistA':
                    temp_string[i] = 'the ' + noundict[gender][noun_idx]
            filled_texts[race][gender] = ' '.join([str(token) for token in temp_string])
    return filled_texts['merged']

def main(argv=None):

    if argv is None:
        argv = sys.argv
    SETTING = argv[1]
    ID = str(argv[2])
    print(SETTING)

    # loop through sentences
    for i, line in enumerate(lines): 
        #create labels
        context = line 
        if context == ' ':
            continue
        id = str(i)

        ### fill masked words with continuations that match the given setting
        if SETTING == 'white':
            sentences = fill_blank(context, 'ProtagonistA', mortage_names, genders=['male', 'female'], race='white')
        elif SETTING == 'tokenized' : 
            sentences = fill_blank(context, 'ProtagonistA', singly_tokenized_names, genders=['male', 'female'], race='white')

            malesent = sentences['male']
            femalesent = sentences['female']
            
        elif SETTING[:2] == 'f_':
            _,female_race,_,male_race = SETTING.split('_')
            malesent = fill_blank(context, 'ProtagonistA', mortage_names, genders=['male'], race=male_race)['male']
            femalesent = fill_blank(context, 'ProtagonistA', mortage_names, genders=['female'], race=female_race)['female']


        
        else: 
            if SETTING == 'names':
                sentences = fill_blank(context, 'ProtagonistA', names)
            elif SETTING == 'checklist':
                sentences = fill_blank(context, 'ProtagonistA', checklist_names)
            elif SETTING == 'pronouns':
                sentences = fill_pronoun(context, 'ProtagonistA')
            elif SETTING == 'nouns':
                sentences = fill_pronoun_noun(context, 'ProtagonistA', nouns)
            elif SETTING == 'nounx':
                sentences = fill_pronoun_noun(context, 'ProtagonistA', nouns, int(ID))
            elif SETTING == 'white_nouns':
                sentences = fill_pronoun_noun(context, 'ProtagonistA', whitenouns)
            elif SETTING == 'black_nouns':
                sentences = fill_pronoun_noun(context, 'ProtagonistA', blacknouns)
            elif SETTING == 'asian_nouns':
                sentences = fill_pronoun_noun(context, 'ProtagonistA', asiannouns)
            elif SETTING == 'hispanic_nouns':
                sentences = fill_pronoun_noun(context, 'ProtagonistA', hispanicnouns)
            malesent = sentences['males']
            femalesent = sentences['females']

        # create dict entry
        entry = {
            "id" : id,
            "target" : TARGET,
            "bias_type" : BIAS_TYPE,
            "context" : context, ## this is the sentence, which contains the BLANK
            "sentences": [ ## this contains the completed sentences
                {
                    "sentence" : malesent,
                    "gold_label" : "stereotype",
                    "id" : id + 'a'
                },
                {
                    "sentence" : femalesent,
                    "gold_label" : "anti-stereotype",
                    "id" : id + 'b'
                }
            ] 
        }
        data_dict["data"]["intrasentence"].append(entry)

    # save data to json files
    out_file = open(f"gender_data_{SETTING}_{ID}.json", "w")
    
    json.dump(data_dict, out_file, indent = 4, ensure_ascii=True)
    
    out_file.close()

if __name__ == '__main__':
    main()