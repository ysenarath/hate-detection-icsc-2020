import json

from hatebase import HatebaseAPI

key = input('Please enter your api key for https://hatebase.org/: ')

hatebase = HatebaseAPI({"key": key})
filters = {"language": "eng"}
format = "json"
# initialize list for all vocabulary entry dictionaries
en_vocab = {}
response = hatebase.getVocabulary(filters=filters, format=format)
pages = response["number_of_pages"]
# fill the vocabulary list with all entries of all pages
# this might take some time...
for page in range(1, pages + 1):
    filters["page"] = str(page)
    response = hatebase.getVocabulary(filters=filters, format=format)
    result = response["result"]
    en_vocab[result['term']] = result

with open('en_vocab.json', 'w', encoding='utf-8') as json_file:
    json.dump(en_vocab, json_file)
