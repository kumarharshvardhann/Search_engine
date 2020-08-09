# python3 search_phase2.py /home/supriya19/Desktop/sem3/IRE/phase_2/chunk_files /home/supriya19/Desktop/sem3/IRE/phase_2/sampleQueriesAndResults/queryfile /home/supriya19/Desktop/sem3/IRE/phase_2/output/out
import xml.sax as sx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
import re
import sys
import time
import gc
import math
import os
from math import log10
index_filename = "inverted_index.txt"
id_title_filename = "id_title_mapping.txt"
current_directory_path = os.getcwd()
stop_words = set(stopwords.words('english')) 
arguments = sys.argv[1:]
index_folder_path = './finalIndex'
id_title_map = {}
inverted_index = {}
secondary_index = {}
secondary_index_idtitle = {}
path_to_index_folder = arguments[0]
path_to_input_query_file = arguments[1]
# path_to_output_file = arguments[2]
score = {}
total_docs = 28694
category_list = ['body', 'ref', 'title', 'category', 'infobox']

# #Fetching the total number of documents
# try:
#     f = open(path_to_index_folder+"/total_docs","r")
#     line = f.readline()
#     total_docs = int(line)


# except:
#     print("Can't find the  total docs file in chunk_files Folder.")
#     print("Re - run the program when the file is in the same folder.")
#     sys.exit(1)

# Fetching the id-title secondary dictionary
try:
    f = open(index_folder_path+"/docToTitle.txt","r")
    line = f.readline()[:-1]
    while line:
        temp = line.split("#")
        doc_id = temp[0]
        title = temp[1].split('|')[0]
        secondary_index_idtitle[doc_id] = title
        line = f.readline()[:-1]
    f.close()
    # print(secondary_index_idtitle)
except:
    print("Can't find the seocndaryyyy index file in 'chunk_files' Folder.")
    print("Re - run the program when the file is in the same folder.")
    sys.exit(1)



# Fetching the term-posting list secondary dictionary
# try:
f = open(index_folder_path+"/secondaryIndex","r")
line = f.readline()[:-1]
while line:
    temp = line.split(":")
    range = temp[0]
    file_name = temp[1]
    # file_name = index_folder_path + '/index_file_' + temp[1]
    secondary_index[range] = file_name
    line = f.readline()[:-1]
f.close()
# except:
#     print("Can't find the seocndary index file in 'index' Folder.")
#     print("Re - run the program when the file is in the same folder.")
#     sys.exit(1)


def tokenize(tmp_str):
    words = []
    tmp_str = re.sub(r'[^\x00-\x7F]+',' ', tmp_str)
    tokens = tmp_str.split()
    ps = PorterStemmer()

    for word in tokens:
        if word not in stop_words and word.isalnum():
            if len(word) < 200:
                temp_word = ps.stem(word)
                if len(temp_word) < 3:
                    continue
                words.append(temp_word)
    return words

def print_result(set_intersection_result):
    for doc_id in set_intersection_result:
        print(id_title_map[doc_id])


def file_to_word(tokenized_unique_words):
    f_t_w = {}
    lst = list(secondary_index.keys())
    # print(secondary_index.keys())
    done = []
    for l in lst:
        temp = l.split("-")
        lower_limit = temp[0]
        upper_limit = temp[1]
        for word in tokenized_unique_words:
            if word > lower_limit and word < upper_limit:
                if word not in f_t_w:
                    f_t_w[secondary_index[l]] = []
                file_name =  secondary_index[l] 
                # print("file_name : ",file_name) 
                f_t_w[secondary_index[l]].append(word)
    print(" ftw : ",f_t_w)
    return f_t_w    


def return_posting_list(file_path, list_of_wrds):
    result = {}
    # print(" list of words : ",list_of_wrds)
    # print(" file path : ",file_path)
    f = open(file_path,'r')
    inverted_idx ={}
    line = f.readline()
    while line:

        temp = line.split("-")
        # temp = line.split("-")
        term = temp[0]
        posting = temp[1]
        if term in list_of_wrds:
            # print(" inside")
            result[term] = posting 
        line = f.readline()
    print(" result : ",result)
    f.close()
    return result

def return_doc_freq(file_name, doc_id):
    f = open(file_name,'r')
    line = f.readline()[:-1]
    freq = 0
    title = ''
    while line:
        temp = line.split("->")
        did = "".join(temp[0].split())
        # print(doc_id, did)
        if int(did) == doc_id :
            title_freq = temp[1]
            xtemp = title_freq.split("|")
            if len(xtemp) >1:
                title = xtemp[0] 
                freq = xtemp[1]
                break
            if freq == 0:
                freq = 1
        line = f.readline()
        # print("----------")
    f.close()
    return title,freq

def return_title(score_of_docs):
    for keys in secondary_index_idtitle:
        lower_limit = int(keys.split("-")[0])
        upper_limit = int(keys.split("-")[1])
        for doc_id in score_of_docs:
            if type(score_of_docs[doc_id]) != str:
                if int(doc_id) > lower_limit and int(doc_id) < upper_limit:
                    title,freq = return_doc_freq(secondary_index_idtitle[keys], int(doc_id))
                    score_of_docs[doc_id] = title
    return score_of_docs



def calculate_tfidf(dicts, tokenized_unique_words, query_length, query_type):
    global score
    print(" inside calculate_tf_idf and term is : ",dicts)
    for term in dicts:
        posting_list = dicts[term] 
        docs = posting_list.split(",")
        doc_containing_token = len(docs)
        idf = log10((float(total_docs)/doc_containing_token))
        print(" idf : ",idf)
        for doc in docs :
            doc_id = doc.split(":")[0]
            t = doc.split(":")[1]
            cats = t.split("#")
            freq = 0
            file_name = ''
            for cat in cats:
                print(" inside cat ")
                if query_type == "field":
                    field_category = cat[0]
                    if field_category in tokenized_unique_words[term]:
                        freq += int(cat[1:]) 
                elif query_type == "normal":
                    freq += int(cat[1:])
            title = secondary_index_idtitle[doc_id]
            tf = log10(float(freq)+1)
            Wt_d = tf*idf
            print("Wt_d : ",Wt_d)
            if query_type == "field":
                Wt_q = float(len(tokenized_unique_words[term])) * idf 
            elif query_type == "normal":
                Wt_q = float(tokenized_unique_words[term]) * idf
            print("Wt_q : ",Wt_q)
            cosine_similarity = Wt_d*Wt_q  # Cosine similarity between query and doc
            score[doc_id] = cosine_similarity

def queryNormal(words_in_query):
    global score
    tokenized_words = tokenize(words_in_query)
    query_length = len(tokenized_words)
    # print(" tokenized query : ",tokenized_words)
    i = 0
    tokenized_unique_words = {}
    for wrd in tokenized_words:
        if wrd not in tokenized_unique_words:
            tokenized_unique_words[wrd] = 0
        tokenized_unique_words[wrd] += 1
    f_t_w = file_to_word(tokenized_unique_words)
    for file_path in f_t_w:
        words_in_this_file = f_t_w[file_path]
        print(words_in_this_file)
        dicts = return_posting_list(file_path, words_in_this_file) #dictionary containing posting lists of the words of the query that belong to this file 
        # print(" dicts : ",dicts)
        calculate_tfidf(dicts, tokenized_unique_words, query_length, "normal") # Calculating the cosine similarity between query and doc by doing dot product to rank docs
    score = sorted(score.items(), key=lambda kv: kv[1])
    score = dict(score)
    
    # ranked_docs = return_title(score)
    print("--------------------- Top 12 results")
    i = 0
    ranked_docs = sorted(score.items(),reverse=True, key = lambda x : x[1])
    print("SCORE : ",ranked_docs)
    for k in ranked_docs:
        if i >= 12:
            break
        # print(k)
        title = secondary_index_idtitle[k[0]]
        print(title)
        i += 1
    print("\n")


def queryField(words_in_query):
    global score

    #  we need to create -> { gandhi : [t,i,c,r], arjun : [b] }
    temp_term_cat = {}
    query_words = words_in_query.split()
    query_length = len(query_words)

    for wrd in query_words:
        if ':' in wrd:

            cat = wrd.split(':')[0]
            if '\'' in cat or '"' in cat:
                cat = cat.replace('\'','')
                cat = cat.replace('"','')
            term = wrd.split(':')[1]
        else:
            term = wrd
        if term not in temp_term_cat:
            temp_term_cat[term] =  []
        tok_term = tokenize(term)
        tok_term = tok_term[0]
        temp_term_cat[tok_term].append(cat[0])
        # print(temp_term_cat)

    f_t_w = file_to_word(list(temp_term_cat.keys())) # fetched all the files that contain these tokens
    for file_path in f_t_w:
        words_in_this_file = f_t_w[file_path]
        dicts = return_posting_list(file_path, words_in_this_file) #dictionary containing posting lists of the words of the query that belong to this file 
        calculate_tfidf(dicts, temp_term_cat, query_length, "field") # Calculating the cosine similarity between query and doc by doing dot product to rank docs
    score = sorted(score.items(), key=lambda kv: kv[1])
    score = dict(score)
    
    i = 0
    ranked_docs = sorted(score.items(), key = lambda x : x[1])
    for k in reversed(ranked_docs):
        if i >= 12:
            break
        # print(k)
        title = secondary_index_idtitle[k[0]]
        print(title)
        i += 1
    print("\n")



# print(" inside main")
# f = open(path_to_input_query_file,'r')
# f2 = open(path_to_output_file,"w+")
# line = f.readline()
# print(line)
while True:
    query = input(" Enter the Query : ")
    score = {}
    start = time.time()
    queryType = ""
    if ":" in query:
        queryType = "Field"
    else:
        queryType = "Normal"

    if queryType == "Normal":
        queryNormal(query)
        
    else:
        queryField(query)
    stop = time.time()
    print(queryType+" Query Took ",stop-start," seconds.")
    # line = f.readline()
# f2.close()
# f.close()