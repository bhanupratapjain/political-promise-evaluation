import json

if __name__ == "__main__":
    data = json.load(open('out/executive_orders.json'))

    with open('out/executive_orders.json', 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)