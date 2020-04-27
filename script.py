import csv

filename = "data/daoist/Huainanzi_淮南子_chu_zhen_xun_俶真訓.txt.json.csv"

characters_to_frequency = {}

def retrieve_characters_to_frequency(input_file):
    characters_to_frequency = {}
    with open(input_file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        # Drop the first row which is just column headers
        next(reader)
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

print(normalize_as_term_frequencies(retrieve_characters_to_frequency(filename)))


#print(term_frequency_scores)

distinct_characters = []

distinct_characters = distinct_characters + list(characters_to_frequency.keys())

documents_to_words = {}

documents_to_words.update({filename: distinct_characters})


words_to_documents = {}

for document, words in documents_to_words.items():
    for word in words:
        currently_put = words_to_documents.get(word, [])
        currently_put.append(document)
        words_to_documents[word] = currently_put


document_frequency_scores = {}

for word, document in words_to_documents.items():
    document_frequency_scores[word] = len(document)

#print(document_frequency_scores)

#document_frequency_scores = distinct_characters_to
