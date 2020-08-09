# WikiPedia Search Engine

## Step 1 : Parsing the Data

To parse the data, run the file "creating_index_phase2.py"
To run the file, the syntax is "python3 creating_index_phase2.py  <path-to-dump-file> <path-to-index-folder>"
The aim is to parse the entire dump and create split files of index that are individually sorted. This function also creates docToTitle map that will map document ID and their title.
Then, merge split files of index and create a meta data of index
The aim is to create final index by parsing all the split index files. This will create multiple final index files that are lexicographically sorted. This function will also create a metadata file of index that will have first first and the index filenumber of each final index file. All the files will be stored in the same folder

## Step 3 : Search

SYNTAX: python3 search.py

This program loads docToTitle file along with final index metadata. The search results will have top 10 results.
Two types of queries are processed:
1. Normal Queries: single word queries, phrase queries
2. Field Queries: This type of queries will have field: <list of words> "field": <list of words> type of input syntax 
	where field can have any of title, body, ref, infobox, category, link as its value