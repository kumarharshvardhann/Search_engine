#python3 creating_index_phase2.py /home/supriya19/Desktop/sem3/IRE/phase_2/data/home/supriya19/Downloads/2018202009/2018202009/data/enwiki-latest-pages-articles26.xml-p42567204p42663461 /home/supriya19/Desktop/sem3/IRE/phase_2

import xml.sax as sx
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import PorterStemmer
from collections import OrderedDict
import re
import os
import sys
import time
import gc
import heapq
xml_filename = 'enwiki-latest-pages-articles26.xml-p42567204p42663461'
xml_filename1 = 'test_data.xml'
index_filename = 'inverted_index.txt'
id_title_filename = 'id_title_mapping.txt'
regex_category = r"\[\[category:(.*?)\]\]"
regex_link = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
stop_words = set(stopwords.words('english')) 
stem_dictionary = {}
arguments = sys.argv[1:]
mod = 25000
global_page_count = 0 
total_time_taken = 0
chunk_no = 0
if arguments[1][-1] != "/":
    folder_path = arguments[1] + "/finalIndex/"
else:
    folder_path = arguments[1] + "finalIndex/"
# secondary_index_idtitle =  open(folder_path+"secondary_index_idtitle", 'w+') # This inex contains range-wise mapping of filename 

    
class WikiXmlHandler(sx.handler.ContentHandler):
    def __init__(self):
        sx.handler.ContentHandler.__init__(self)
        self._buffer = None
        self._values = {}
        self._current_tag = None
        self._pages = []
        self._id_title_map = {}
        self._title_id_map = {}
        self._id_buffer = ''
        self._id_flag = False
        self._inverted_index = {}
        self._title_inverted_index = {}


    def characters(self, content):
        if self._current_tag:
            self._buffer.append(content)

    def startElement(self, name, attrs):
        if name == 'title' :
            self._current_tag = name
            self._buffer = []
            self._id_flag = True #to add only id corresponding to the title and not other id's which are redundant

        elif name == 'id' or name == 'text':

            self._current_tag = name
            self._buffer = []

    def endElement(self, name):
        global global_page_count 
        global chunk_no
        global stem_dictionary
        # if name == self._current_tag and name != 'id':
        
        if name == 'title':
            self._values[name] = ' '.join(self._buffer)

        elif name == 'text':
             text_content = ' '.join(self._buffer)
             self._values[name] = text_content.casefold()

        elif name == 'id' and self._id_flag == True:
            self._values[name] = ' '.join(self._buffer)
            global_page_count += 1
            # print(self._values['id'])

            self._id_title_map[self._values['id']] = self._values['title']
            self._values['title'] = self._values['title'].casefold() 
            self._title_id_map[self._values['title']] = self._values['id']

            self._id_flag = False

        elif name == 'page':
            self._pages.append((self._values['title'], self._values['text']))
            # print(" the count of PAGE is : ",global_page_count)
            mod_val = global_page_count % mod
            if mod_val == 0:
                chunk_no += 1
                # print(" the count of CHUNK is : ",chunk_no)
                self.create_index_in_chunks()
                # Reset all the dictionaries for the next chunk
                
                self._pages = []
                self._inverted_index = {}
                self._title_inverted_index = {}
                self._id_title_map = {}
                self._title_id_map = {}
                stem_dictionary = {} # emptying the dictionary after one file otherwise the size exceeds the limit giving error "RecursionError: maximum recursion depth exceeded in comparison"


    def create_index_in_chunks(self):
        
        global total_time_taken


        gc.disable()
        start1 = time.time()
        self.data_preprocessing()
        f = open(folder_path + "i_" + str(chunk_no),"w")
        for key,val in sorted(self._inverted_index.items()):
            # s =str(key.encode('utf-8'))+"="
            key += "-"
            for k,v in sorted(val.items()):
                key += str(k) + ":"
                for k1,v1 in v.items():
                    key = key + str(k1) + str(v1) + "#"
                key = key[:-1]+","
            key = key[:-1]+"\n"
            f.write(key)
        keys = list(self._id_title_map.keys())
        first_id = keys[0]
        last_id = keys[-1]
        # print("idtitle_file_" + str(first_id) + "-" + str(last_id))
        
        # f2 = open(folder_path + "idtitle_file_" + str(first_id),"w")
        f2 = open(folder_path + "docToTitle.txt","a+")
        # secondary_index_entry = str(first_id) + "-" + str(last_id) + ":" + folder_path + "idtitle_file_" + str(first_id)"
        secondary_index_entry = str(first_id) + "-" + str(last_id) + ":" + folder_path + "docToTitle.txt"


        # secondary_index_idtitle.write(secondary_index_entry+"\n")
        # print(" self._id_title_map : ",self._id_title_map)
        for key,value in self._id_title_map.items():
            print(" id in dict : ",key.strip())
            f2.write(key.strip()+"#"+value+"\n")
        f.close()
        f2.close()
        end1 = time.time()
        gc.enable()
        total_time_taken += end1-start1
        # print(" The time taken at chunk "+ str(chunk_no) +" is : ",total_time_taken)

    def tokenize(self, tmp_str, document_id, key_name):
        global stem_dictionary
        words = []
        tmp_str = re.sub(r'[^\x00-\x7F]+',' ', tmp_str)
        # nltk.download('punkt')
        # tokens = word_tokenize(tmp_str)
        tokens = tmp_str.split()

        # remove all tokens that are not alphanumeric
        ps = PorterStemmer()

        for word in tokens:
            if word.isalnum() and word not in stop_words:
                if word in stem_dictionary:
                    temp_word = stem_dictionary[word] 
                else:
                    if len(word) >= 200: 
                        continue    # Stemmer can't handle this and throws recursion error after 4-5 hours when encounter such a word
                    temp_word = ps.stem(word) 
                    stem_dictionary[word] = temp_word
                
                if len(temp_word) < 3:
                    continue
                if temp_word not in self._inverted_index:
                    self._inverted_index[temp_word] = {}
                if document_id not in self._inverted_index[temp_word]:
                    self._inverted_index[temp_word][document_id] = {}
                if key_name not in self._inverted_index[temp_word][document_id]: 
                    self._inverted_index[temp_word][document_id][key_name] = 0
                self._inverted_index[temp_word][document_id][key_name] += 1
                if key_name == 't':
                    if temp_word not in self._title_inverted_index: 
                        self._title_inverted_index[temp_word] = {}
                    if document_id not in self._title_inverted_index[temp_word]:
                        self._title_inverted_index[temp_word][document_id] = 0
                    self._title_inverted_index[temp_word][document_id] += 1
                words.append(temp_word)
        # make is word.isalnum(), if number is necessary to be handled
        return words

    def data_preprocessing(self):
         # before tokenizing the text extract the category
        
        for tup in self._pages:
            no_of_terms_in_doc = 0
            title = tup[0]
            text = tup[1]
            # text = text.encode(encoding = 'UTF-8')
            
            document_id = self._title_id_map[title]
            # print(" ACTUAL TEXT : ",text)
            # print("===========================================================================")

            #extracting fields------------------------------------------

            #Remove all http links
            links = re.findall(regex_link, tup[1])     
            for link in links:
                text = text.replace(link,"")

            # print(" text after removing links : ",text)
            # print("-----------------------------------------------------------------")
            
            #Extract and replace category
            matches = re.finditer(regex_category, text, re.MULTILINE | re.DOTALL)
            category_name = ''
            tok_category = []
            for matchNum, match in enumerate(matches):
                for groupNum in range(0, len(match.groups())):
                    category = match.group(1)
                    text = text.replace(match.group(0),'')
                    if '|' in category:
                        category_name = category.split('|')
                        # considering the data after pipe in category as the bobdy text data only :
                        temp_txt = category_name[1:]
                        # print(" gadhi wala temp text : ",temp_txt)
                        for j in temp_txt:
                            text += " "+ j

                        category_name = category_name[0]
                        
                    else:
                        category_name = category
                    cat = handler.tokenize(category_name, document_id, 'c')    # 'c' is for category
                    tok_category.append(cat)

            # print(" text after removing category : ",text)
            # print("-----------------------------------------------------------------")
            # print(" tokenized category : ",tok_category)
            # print("=================================================================================")            
            no_of_terms_in_doc += len(tok_category)

            #tokenize Title

            tok_title = self.tokenize(title, document_id, 't') # 't' is for title
            no_of_terms_in_doc += len(tok_title)

            # print(" tokenized title : ",tok_title)
            # print("=================================================================================")    

            #Remove #Redirect
            text = text.replace("#redirect","")

            # print(" text after removing #redirect : ",text)
            # print("=================================================================================")    

            # Extract and remove infobox
            text, infobox_list =  handler.extract_infobox(text)
            parameters = infobox_list.split("\n")
            parameters = parameters[1:]
            tok_infobox = []
            for value in parameters:
                temp = value.split("=")
                if(len(temp)>1):
                    tok_infobox += self.tokenize(temp[1], document_id,'i') # 'i' is for infobox

            
            # print(" text after removing infobox : ",text)
            # print("-----------------------------------------------------------------")
            # print(" tokenized infobox : ",tok_infobox)
            # print("=================================================================================")    
            no_of_terms_in_doc += len(tok_infobox)

            #Extract and remove References
            ref_list =  re.findall(regEx['ref1'], text)
            tok_ref = []
            for ref in ref_list:
                text, ext_ref = handler.extract_references(ref, text)
                tok_ref += self.tokenize(ext_ref, document_id,'r') # 'e' is for external reference
            
            # print(" text after removing references : ",text)
            # print("-----------------------------------------------------------------")
            # print(" tokenized references : ",tok_ref)
            # print("=================================================================================")    
            no_of_terms_in_doc += len(tok_ref)

            #Extract and remove External Links
            external_link_text = re.findall(regEx['ext1'], tup[1])
            tok_extlink = []
            for ext in external_link_text:
                ext_link = handler.extract_external_link(ext)
                text = text.replace("external links","")
                text = text.replace(ext,"")
                tok_extlink += self.tokenize(ext_link, document_id,'l') # 'l' is for external links

            
            # print(" text after removing ext links : ",text)
            # print("-----------------------------------------------------------------")
            # print(" tokenized category : ",tok_extlink)
            # print("=================================================================================")    
            no_of_terms_in_doc += len(tok_extlink)

            tok_bodytxt = self.tokenize(text, document_id,'b') # 'b' is for body text
            no_of_terms_in_doc += len(tok_bodytxt)

            # print(" tokenized bodytext : ",tok_bodytxt)
            # print("=================================================================================")   
            self._id_title_map[document_id] = self._id_title_map[document_id] + "|" + str(no_of_terms_in_doc) 
    def extract_infobox(self,txt):
        count = 0
        temp_string = ""
        i = 0
        flag = 0
        while i<len(txt):
            if( txt[i] == '{' and i+1 < len(txt) ):
                if(i+9<len(txt) and txt[i+2:i+9].casefold() == "infobox"):
                    i += 9
                    count += 2
                    flag = 1
                elif flag == 1:
                    count += 1
                    # 
            if(txt[i] == '}' and flag==1):
                count -= 1
            if(count > 0):
                temp_string += txt[i]
            else:
                flag = 0
            i +=1 
        # print(" temp_string : ",temp_string)
        txt = txt.replace(temp_string,"")
        txt = txt.replace("{{infobox}","")
        # print(" replace ",txt)
        return txt, temp_string

    def extract_references(self,txt, whole_text):
        count = 0
        temp_string = ""
        i = 0
        while i<len(txt):
            if( txt[i] == '{' and i+1 <= len(txt)-1 ):
                count += 1
            if(txt[i] == '}' and i+1 <= len(txt)-1 ):
                count -= 1
                if count==0 and i+1<=len(txt)-1 and txt[i+1] == "\n" and i+2 < len(txt) and txt[i+2]!="{":
                    break
            if count > 0:
                temp_string += txt[i]
            i +=1 
        whole_text = whole_text.replace("==references==","")
        whole_text = whole_text.replace(temp_string+"}","")
        return whole_text,temp_string

    def extract_external_link(self,txt):
        count = 0
        temp_string = ""
        i = 0
        while i<len(txt):
            if( txt[i] == '{' and i+1 < len(txt)-1 ):
                count += 1
            if(txt[i] == '}' and i+1 < len(txt)-1 ):
                count -= 1
            if count > 0:
                temp_string += txt[i]
            i +=1 
        # print(" temp ",temp_string)
        return temp_string
    
        # def fetch_docs():
    #     pass



regEx = {}
# regEx['ibox'] = re.compile(r"\{\{Infobox.*", flags=re.IGNORECASE | re.DOTALL)
regEx['links'] = re.compile(r"(www|http:|https:)+[^\s]+[\w]", flags=re.IGNORECASE | re.DOTALL)
regEx['ref1'] = re.compile(r"(== ?References.*)[ \n]\{\{", flags=re.IGNORECASE | re.DOTALL)
regEx['ext1'] = re.compile(r"(== ?External links.*)\n", flags=re.IGNORECASE | re.DOTALL)
regEx['body_pat'] = re.compile(r"== ?References.*", flags=re.DOTALL)

parser = sx.make_parser()
parser.setFeature(sx.handler.feature_namespaces, 0)
handler = WikiXmlHandler()
parser.setContentHandler(handler)
# current_directory_path = os.getcwd()
data_file_path = arguments[0]
parser.parse(data_file_path)
if len(handler._pages) > 0:
    chunk_no += 1
    handler.create_index_in_chunks()

# secondary_index_idtitle.close()

doc_file = open(folder_path + "total_docs", 'w+')
doc_file.write(str(global_page_count))
doc_file.close()

"""
================================================================================================================================
MERGING THE INDEX FILE CHUNKS TO FORM THE FINAL INDEX FILES WITH TERMS SORTED LEXICOGRAPHICALLY ON WHICH SEARCH CAN BE PERFORMED
================================================================================================================================
"""

#initialization bcoz the above code in commented========
# chunk_no = 3

#========================

start2 = time.time()

# if arguments[1][-1] != "/":
#     folder_path = arguments[1] + "/chunk_files/"
# else:
#     folder_path = arguments[1] + "chunk_files/"


# f1 = open(folder_path+"i_1", 'r+')
# f2 = open(folder_path+"i_2", 'r+')
# f3 = open(folder_path+"i_3", 'r+')
# f4 = open(folder_path+"i_4", 'r+')

file_desc_list = []
file_name_to_file = {}
# opening the chunk files
for i in range(1,chunk_no+1):
    temp_file = open(folder_path + "i_" + str(i), 'r+')
    file_desc_list.append(temp_file)
    file_name_to_file[temp_file.name] = temp_file
    # print(file_desc_list)


secondary_index =  open(folder_path+"secondaryIndex", 'w+') # This inex contains range-wise mapping of filename 
list_of_first_terms = [] # list to be converted to the min heap
term_to_file = {} # map that contains mapping of the term with the file it's present in 
term_posting_temp = {}

# Reading first line from all the files for creating the heap from scratch
heap = []
heapq.heapify(heap)
for file in file_desc_list:
    temp = file.readline()[:-1].split("-")
    heapq.heappush(heap,(temp[0], file.name))
    if temp[0] in term_posting_temp:
        x = term_posting_temp[temp[0]]
        term_posting_temp[temp[0]] =  x + "," + temp[1]
        line = file.readline()
        if not line:
            continue
        temp = line[:-1].split("-")
        flag = True
        while flag:
            if temp[0] not in term_posting_temp: 
                flag = False
                heapq.heappush(heap,(temp[0],file.name))
                term_posting_temp[temp[0]] = temp[1]
            else:
                x = term_posting_temp[temp[0]]
                term_posting_temp[temp[0]] =  x + "," + temp[1]
                line = file.readline()
                if not line:
                    continue
                temp = line[:-1].split("-")
    else:
        heapq.heappush(heap,(temp[0],file.name))
        term_posting_temp[temp[0]] = temp[1]

pointer_position = 0
position_mod = 25000
file_id = 0

# when the size of a global index file is exhausted
if len(term_posting_temp) == position_mod:
    lower_limit = list(term_posting_temp.keys())[0]
    upper_limit = list(term_posting_temp.keys())[-1]
    f = open(folder_path+"index_file_"+ str(file_id), 'w+')  
    file_no = f.name.split('_')[2]
    rng = lower_limit+' ' + file_no +"\n"
    secondary_index.write(rng)
    for key in sorted(term_posting_temp):
        f.write(key+'-'+term_posting_temp[key]+'\n')
    f.close()
    file_id += position_mod
    term_posting_temp = {}
# print("---------------")
# print(heap)
# print("======================================================")

while len(heap) > 0:
    m = pointer_position % position_mod
    top = heapq.heappop(heap)
    # to decide which file next to be read
    file_desc = file_name_to_file[top[1]] # the tuple obtained as top has term and file name 
    
    # Reading the next element
    line = file_desc.readline()
    if not line:
        continue
    temp = line[:-1].split("-")
    heapq.heappush(heap,(temp[0], file_desc.name))
    if temp[0] in term_posting_temp:
        x = term_posting_temp[temp[0]]
        term_posting_temp[temp[0]] =  x + "," + temp[1]
        line = file_desc.readline()
        if not line:
            continue
        temp = line[:-1].split("-")
        flag = True
        while flag:
            if temp[0] not in term_posting_temp: 
                flag = False
                heapq.heappush(heap,(temp[0],file_desc.name))
                term_posting_temp[temp[0]] = temp[1]
            else:
                x = term_posting_temp[temp[0]]
                term_posting_temp[temp[0]] =  x + "," + temp[1]
                line = file_desc.readline()
                if not line:
                    continue
                temp = line[:-1].split("-")
    else:
        heapq.heappush(heap,(temp[0],file_desc.name))
        term_posting_temp[temp[0]] = temp[1]


    if len(term_posting_temp) == position_mod :
        lower_limit = list(term_posting_temp.keys())[0]
        upper_limit = list(term_posting_temp.keys())[-1]
        f = open(folder_path+"index_file_"+ str(file_id), 'w+')  
        rng = lower_limit+'-'+upper_limit +':' + f.name +"\n"
        secondary_index.write(rng)
        for key in sorted(term_posting_temp):
            f.write(key+'-'+term_posting_temp[key]+'\n')
        f.close()
        file_id += position_mod
        term_posting_temp = {}

if len(term_posting_temp) < position_mod-1 :
    lower_limit = list(term_posting_temp.keys())[0]
    upper_limit = list(term_posting_temp.keys())[-1]
    f = open(folder_path+"index_file_"+ str(file_id), 'w+')  
    rng = lower_limit+'-'+upper_limit +':' + f.name +"\n"
    secondary_index.write(rng)
    for key in sorted(term_posting_temp):
        f.write(key+'-'+term_posting_temp[key]+'\n')
    f.close()
    file_id += position_mod
    term_posting_temp = {}

for file in file_desc_list:
    file.close()

end2 = time.time()

total_time_taken += end2-start2

print(" PREPROCESSING + INDEX CREATION TIME FROM CHUNK : ",total_time_taken)