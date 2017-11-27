import json

if __name__ == '__main__':
    data_repository = 'repo.json'

    jsonFile = open(data_repository, 'r')
    values = json.load(jsonFile)
    file = open('promises.json', 'w')

    data= []
    for promise in values['promises']:
        data.append({
            'promise_title': promise['title'],
            'promise_tag': promise['tags'],
            'promise_description': promise['description'],
            'promise_status': promise['status']
        })
    json.dump(data, file)
    file.close()
    jsonFile.close()
