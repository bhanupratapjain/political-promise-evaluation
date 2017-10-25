import json

if __name__ == '__main__':
    data_repository = 'repo.json'

    jsonFile = open(data_repository, 'r')
    values = json.load(jsonFile)
    file = open('promises.json', 'w')

    data = {}
    data['promises'] = []

    for promise in values['promises']:

        data['promises'].append({
            'promise_title': promise['title'],
            'promise_tag': promise['tags'],
            'promise_description': promise['description']
        })

        json.dump(data, file)

    file.close()
    jsonFile.close()
