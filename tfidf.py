#!/usr/bin/python

import csv
import string
import math
import copy

csv.field_size_limit(1000000000)

#
# Convert string to lowercase, remove punctuation and split
# on space
#
def tokenize(text, exclude_chars = string.punctuation):
    # Convert all words to lower case
    text = text.lower().translate(None, exclude_chars)

    # Remove whitespace
    return text.replace('\n', ' ').split(" ")


#
# Sort dictionary on values and return top k items
#
def get_top_terms(dict, num_terms = 20):
    list = dict.items()
    list.sort(key=lambda item: item[1], reverse=True)
    return list[0:num_terms]


#
# Main function
#
def main():
    with open('state-of-the-union.csv', 'rb') as csvfile:
        speechreader = csv.reader(csvfile)

        all_speeches_df = {}
        all_speeches_tf = {}
        decades_to_speech_ids = {
            "190.0":[], "191.0":[], "192.0":[], "193.0":[], "194.0":[], "195.0": [],
            "196.0":[], "197.0":[], "198.0":[], "199.0":[], "200.0":[], "201.0": []
        }

        speech_id = 0
        assigned_speech_id = 0

        for row in speechreader:
            
            # Prepare decade to speech_ids mapping
            speech_year = row[0]
            decade_key = str(math.floor(int(speech_year)/10))
            if decade_key in decades_to_speech_ids:
                decades_to_speech_ids[decade_key].append(speech_id);

            if speech_year == '1960':
                assigned_speech_id = speech_id

            speech_list = tokenize(row[1])

            # Count term frequencies and document frequencies
            speech_tf = {}
            for term in speech_list:
                if term in speech_tf:
                    speech_tf[term] += 1
                else: 
                    speech_tf[term] = 1
                    
                    if term in all_speeches_df:
                        all_speeches_df[term] += 1
                    else:
                        all_speeches_df[term] = 1

            all_speeches_tf[speech_id] = speech_tf

            speech_id += 1

        num_docs = speech_id

        # Compute Inverse Document Frequency (IDF)
        for term in all_speeches_df:
            term_df = all_speeches_df[term]
            all_speeches_df[term] = math.log(num_docs/term_df)

        # Compute TF-IDF Vectors
        all_speeches_tfidf = copy.deepcopy(all_speeches_tf)
        for speech_id in all_speeches_tf:
            for term in all_speeches_tf[speech_id]:
                term_tf  = all_speeches_tf[speech_id][term]
                term_df = all_speeches_df[term]
                term_tf_idf = term_tf * term_df
                all_speeches_tfidf[speech_id][term] = term_tf_idf

        # Normalize TF-IDF Vectors
        for speech_id in all_speeches_tfidf:
            norm_accum = 0
            for term in all_speeches_tfidf[speech_id]:
                term_tf_idf  = all_speeches_tfidf[speech_id][term]
                norm_accum += term_tf_idf * term_tf_idf

            norm_accum = math.sqrt(norm_accum)

            for term in all_speeches_tfidf[speech_id]:
                term_tf_idf  = all_speeches_tfidf[speech_id][term]
                all_speeches_tfidf[speech_id][term] = term_tf_idf/norm_accum

        # Extract top 20 terms from assigned speech
        sorted_term_weight_tuples = get_top_terms(all_speeches_tfidf[assigned_speech_id])

        outfile = open('output.txt', 'w')
        outfile.write("#####################################\n")
        outfile.write("######### Top 20 Terms in 1960 ######\n")
        outfile.write("#####################################\n")
        for term_weight_tuple in sorted_term_weight_tuples:
            outfile.write('%-20s %s \n' % (str(term_weight_tuple[0]), str(term_weight_tuple[1])) )
        outfile.write('\n\n')

        # Extract top 20 terms from each decade since 1900
        decade_tfidfs = {}

        for decade, speeches in decades_to_speech_ids.items():
            decade_tfidfs[decade] = {}
            
            for speech_id in speeches:
               for term in  all_speeches_tfidf[speech_id]:
                   term_tf_idf = all_speeches_tfidf[speech_id][term]
                   if term in decade_tfidfs[decade]:
                       decade_tfidfs[decade][term] += term_tf_idf
                   else:
                       decade_tfidfs[decade][term]  = term_tf_idf

        for decade, term_weight_dict in decade_tfidfs.items():
            outfile.write("########################################################\n")
            outfile.write('######### Top 20 Terms in decade starting ' + decade.replace('.', '') + ' #########\n')
            outfile.write("########################################################\n")
            for term_weight_tuple in get_top_terms(term_weight_dict):
                outfile.write('%-20s %s \n' % (str(term_weight_tuple[0]), str(term_weight_tuple[1])) )
            outfile.write('\n\n')

        outfile.close()

if __name__ == '__main__':
    main()
