import csv
import glob
import math

from itertools import islice

filename = "data/daoist/Huainanzi_淮南子_chu_zhen_xun_俶真訓.txt.json.csv"

characters_to_frequency = {}

number_of_texts_to_compare = 1000

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

with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    # Drop the first row which is just column headers
    next(reader)
    for row in reader:
        character = row[0]
        absolute_frequency = row[1]
        new_entry = {character : int(absolute_frequency)}
        characters_to_frequency.update(new_entry)

total_number_of_characters = sum(characters_to_frequency.values())

term_frequency_scores = {char : freq / total_number_of_characters for char, freq in
                         characters_to_frequency.items()}

def normalize_as_term_frequencies(characters_to_frequency):
    total_number_of_characters = sum(characters_to_frequency.values())

    term_frequency_scores = {char : freq / total_number_of_characters for char, freq in
                             characters_to_frequency.items()}
    return term_frequency_scores

def retrieve_term_frequences_from_file(input_file):
    return normalize_as_term_frequencies(retrieve_characters_to_frequency(input_file))

daoist_texts = glob.glob("data/daoist/*.json.csv")

general_texts = glob.glob("data/general/*.json.csv")

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

guanzi_file_name = "data/daoist/Guanzi_管子_Neiye_內業.txt.json.csv"

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

benchmark_file_name = "data/daoist/Zhuangzi_莊子_Qiwulun_齊物論.txt.json.csv"

benchmark_tf_idf_score = tf_idf_scores[benchmark_file_name]

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

benchmark_file_name = "data/daoist/Laozi_老子_Daodejing_道德經.txt.json.csv"

benchmark_tf_idf_score = tf_idf_scores[benchmark_file_name]

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

#distinct_characters = []

#distinct_characters = distinct_characters + list(characters_to_frequency.keys())

#documents_to_words = {}

#documents_to_words.update({filename: distinct_characters})


#words_to_documents = {}

#for document, words in documents_to_words.items():
    #for word in words:
        #currently_put = words_to_documents.get(word, [])
        #currently_put.append(document)
        #words_to_documents[word] = currently_put


#document_frequency_scores = {}

#for word, document in words_to_documents.items():
    #document_frequency_scores[word] = len(document)

#print(document_frequency_scores)

#document_frequency_scores = distinct_characters_to
