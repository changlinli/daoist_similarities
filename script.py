import csv
import glob
import math
import re

from collections import Counter

import itertools
from itertools import islice

filename = "nonsimpsave/nonsimpsave copy/淮南子 - Huainanzi-俶真訓.txt"

number_of_texts_to_compare = 1000

def convert_to_ngram(string, n):
    return list(zip(string, string[(n - 1):]))

def group_by_frequency(listOfElems):
    result = { elem:len(list(repetitions)) for (elem, repetitions) in itertools.groupby(listOfElems) }
    return result

def group_ngrams_by_frequency(string, n):
    return group_by_frequency(convert_to_ngrams(string, n))

def retrieve_ngrams_to_frequency(input_file, n):
    ngrams_to_frequency = Counter()
    with open(input_file, newline='') as textfile:
        reader = textfile.readlines()
        for row in reader:
            # Split by punctuation so that we treat each punctuation mark as a
            # separate clause
            split_by_punctuation = re.split('。|，|、|；|：|《|》|？|！|「|」', row)
            for clause in split_by_punctuation:
                partial_ngrams_to_frequency = Counter(retrieve_ngrams_to_frequency_str(clause, n))
                ngrams_to_frequency = ngrams_to_frequency + partial_ngrams_to_frequency
    return ngrams_to_frequency

def retrieve_ngrams_to_frequency_str(input_string, n):
    ngrams_to_frequency = {}
    string_length = len(input_string)
    for idx, char in enumerate(input_string):
        if idx > string_length - n:
            pass
        else:
            current_ngram = ""
            for i in range(n):
                if i == 0:
                    current_ngram = str(char)
                else:
                    current_ngram = current_ngram + input_string[idx + i]
            if current_ngram in ngrams_to_frequency:
                old_absolute_frequency = ngrams_to_frequency[current_ngram]
                ngrams_to_frequency[current_ngram] = old_absolute_frequency + 1
            else:
                ngrams_to_frequency[current_ngram] = 1
    return ngrams_to_frequency

def retrieve_characters_to_frequency(input_file):
    characters_to_frequency = {}
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Drop the first row which is just column headers
        try:
            next(reader)
        except StopIteration:
            print("This file was empty: " + input_file)
        for row in reader:
            character = row[0]
            absolute_frequency = row[1]
            new_entry = {character : int(absolute_frequency)}
            characters_to_frequency.update(new_entry)
    return characters_to_frequency

characters_to_frequency = retrieve_ngrams_to_frequency(filename, 2)

total_number_of_characters = sum(characters_to_frequency.values())

term_frequency_scores = {char : freq / total_number_of_characters for char, freq in
                         characters_to_frequency.items()}

def normalize_as_term_frequencies(characters_to_frequency):
    total_number_of_characters = sum(characters_to_frequency.values())

    term_frequency_scores = {char : freq / total_number_of_characters for char, freq in
                             characters_to_frequency.items()}
    return term_frequency_scores

def retrieve_term_frequences_from_file(input_file):
    return normalize_as_term_frequencies(retrieve_ngrams_to_frequency_str(input_file, 2))

daoist_texts = [filename, "nonsimpsave/nonsimpsave copy/管子 - Guanzi-內業.txt",
                "nonsimpsave/nonsimpsave copy/莊子 - Zhuangzi-齊物論.txt",
                "nonsimpsave/nonsimpsave copy/道德經.txt"]

general_texts = glob.glob("nonsimpsave/nonsimpsave copy/*")

daoist_text_term_frequencies = {}

for file_name in daoist_texts:
    daoist_text_term_frequencies[file_name] = retrieve_term_frequences_from_file(file_name)

general_text_term_frequencies = {}

for file_name in general_texts:
    general_text_term_frequencies[file_name] = retrieve_term_frequences_from_file(file_name)

all_term_frequencies = {**daoist_text_term_frequencies,
                        **general_text_term_frequencies}

#print(all_term_frequencies)

documents_to_distinct_characters = {}

for document, term_frequencies in all_term_frequencies.items():
    distinct_characters = []
    distinct_characters = distinct_characters + list(term_frequencies.keys())
    documents_to_distinct_characters[document] = distinct_characters

#print(documents_to_distinct_characters)

characters_to_documents = {}

for document, characters in documents_to_distinct_characters.items():
    for character in characters:
        currently_put = characters_to_documents.get(character, [])
        currently_put.append(document)
        characters_to_documents[character] = currently_put

#print(characters_to_documents)

document_frequency_scores = {}

for character, documents in characters_to_documents.items():
    document_frequency_scores[character] = len(documents)

#print(document_frequency_scores)

number_of_documents_in_corpus = len(general_texts + daoist_texts)

inverse_document_frequency_scores = {}

for character, frequency_score in document_frequency_scores.items():
    inverse_document_frequency_scores[character] = \
        math.log(number_of_documents_in_corpus / (frequency_score + 1))

#print(inverse_document_frequency_scores)

tf_idf_scores = {}

for document, term_frequencies in all_term_frequencies.items():
    tf_idf_for_document = {}
    for character, frequency in term_frequencies.items():
        idf = inverse_document_frequency_scores[character]
        tf_idf_for_document[character] = frequency * idf
    tf_idf_scores[document] = tf_idf_for_document

#print(tf_idf_scores)

def dot_product_between_two_term_frequencies(term_frequencies_0,
                                             term_frequencies_1):
    current_sum = 0
    for character, term_frequency in term_frequencies_0.items():
        other = term_frequencies_1.get(character, 0)
        current_sum = current_sum + term_frequency * other
    return current_sum

def magnitude(term_frequencies):
    current_sum = 0
    for character, term_frequency in term_frequencies.items():
        current_sum = current_sum + (term_frequency ** 2)
    return math.sqrt(current_sum)

def similarity_between_two_documents(tf_idf_0, tf_idf_1):
    numerator = dot_product_between_two_term_frequencies(tf_idf_0, tf_idf_1)
    denominator = magnitude(tf_idf_0) * magnitude(tf_idf_1)
    if denominator == 0:
        return 0
    else:
        return numerator / denominator

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

#######
# Huainanzi
#######

default_tf_idf_score = tf_idf_scores[filename]

similarities_to_default = {}

for document, tf_idf_score in tf_idf_scores.items():
    similarities_to_default[document] = \
        similarity_between_two_documents(default_tf_idf_score, tf_idf_score)

sorted_similarities_to_huainanzi = {k: v for k, v in sorted(similarities_to_default.items(), key=lambda item: item[1])}

top_25_huainanzi = take(number_of_texts_to_compare, reversed(list(sorted_similarities_to_huainanzi.items())))

print(f"Top {number_of_texts_to_compare} most similar to Huainanzi:")

for filename, similarity_score in top_25_huainanzi:
    print(filename + ": " + repr(similarity_score))

with open('huainanzi_chuzhenxun_similarities.txt', 'w') as fp:
    for filename, similarity_score in top_25_huainanzi:
        fp.write(f'{filename}: {similarity_score}\n')

#######
# Guanzi
#######

guanzi_file_name = "nonsimpsave/nonsimpsave copy/管子 - Guanzi-內業.txt"

guanzi_tf_idf_score = tf_idf_scores[guanzi_file_name]

similarities_to_guanzi = {}

for document, tf_idf_score in tf_idf_scores.items():
    similarities_to_guanzi[document] = \
        similarity_between_two_documents(guanzi_tf_idf_score, tf_idf_score)

sorted_similarities_to_guanzi = {k: v for k, v in sorted(similarities_to_guanzi.items(), key=lambda item: item[1])}

top_25_guanzi = take(number_of_texts_to_compare, reversed(list(sorted_similarities_to_guanzi.items())))

print(f"Top {number_of_texts_to_compare} most similar to Guanzi:")

for filename, similarity_score in top_25_guanzi:
    print(filename + ": " + repr(similarity_score))

with open('guanzi_neiye_smilarities.txt', 'w') as fp:
    for filename, similarity_score in top_25_guanzi:
        fp.write(f'{filename}: {similarity_score}\n')

#######
# Zhuangzi
#######

zhuangzi_file_name = "nonsimpsave/nonsimpsave copy/莊子 - Zhuangzi-齊物論.txt"

benchmark_tf_idf_score = tf_idf_scores[zhuangzi_file_name]

similarities_to_benchmark = {}

for document, tf_idf_score in tf_idf_scores.items():
    similarities_to_benchmark[document] = \
        similarity_between_two_documents(benchmark_tf_idf_score, tf_idf_score)

sorted_similarities_to_benchmark = {k: v for k, v in sorted(similarities_to_benchmark.items(), key=lambda item: item[1])}

top_25_similar_to_benchmark = take(number_of_texts_to_compare, reversed(list(sorted_similarities_to_benchmark.items())))

print(f"Top {number_of_texts_to_compare} most similar to Zhuangzi Qiwulun:")

for filename, similarity_score in top_25_similar_to_benchmark:
    print(filename + ": " + repr(similarity_score))

with open('zhuangzi_qiwulun_smilarities.txt', 'w') as fp:
    for filename, similarity_score in top_25_similar_to_benchmark:
        fp.write(f'{filename}: {similarity_score}\n')

#######
# Daodejing
#######

daodejing_file_name = "nonsimpsave/nonsimpsave copy/道德經.txt"

benchmark_tf_idf_score = tf_idf_scores[daodejing_file_name]

similarities_to_benchmark = {}

for document, tf_idf_score in tf_idf_scores.items():
    similarities_to_benchmark[document] = \
        similarity_between_two_documents(benchmark_tf_idf_score, tf_idf_score)

sorted_similarities_to_benchmark = {k: v for k, v in sorted(similarities_to_benchmark.items(), key=lambda item: item[1])}

top_25_similar_to_benchmark = take(number_of_texts_to_compare, reversed(list(sorted_similarities_to_benchmark.items())))

print(f"Top {number_of_texts_to_compare} most similar to Daodejing:")

for filename, similarity_score in top_25_similar_to_benchmark:
    print(filename + ": " + repr(similarity_score))

with open('daodejing_smilarities.txt', 'w') as fp_daodejing:
    for filename, similarity_score in top_25_similar_to_benchmark:
        fp_daodejing.write(f'{filename}: {similarity_score}\n')
