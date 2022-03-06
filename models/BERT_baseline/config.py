config = {
    'domain': 'restaurant-3',
    'device': 'cuda',
    'base_path': ''
}
bert_mapper = {
    'laptop': 'activebus/BERT-DK_laptop',
    'restaurant-3': 'activebus/BERT-DK_rest',
    'restaurant-5': 'activebus/BERT-DK_rest',
    'kto': './data/kto/BERT-DK_kto'
}

sbert_mapper = {
    'laptop': 'all-mpnet-base-v2',
    'restaurant-3': 'all-mpnet-base-v2',
    'restaurant-5': 'all-mpnet-base-v2',
    'kto': 'paraphrase-multilingual-mpnet-base-v2'
}

path_mapper = {
    'laptop': './data/laptop',
    'restaurant-3': './data/restaurant-3',
    'restaurant-5': './data/restaurant-5',
    'kto': './data/kto'
}
aspect_category_mapper = {
    'laptop': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'restaurant-3': ['food', 'place', 'service'],
    'restaurant-5': ['location', 'drinks', 'food', 'ambience', 'service'],
    'kto': ['service', 'app', 'assortiment', 'beschikbaarheid', 'corona', 'kwaliteit', 'winkel', 'personeel',
            'opgeruimd', 'prijzen', 'overig']}

aspect_seed_mapper = {
    'laptop': {
        'support': {"support", "service", "warranty", "coverage", "replace"},
        'os': {"os", "windows", "ios", "mac", "system", "linux"},
        'display': {"display", "screen", "led", "monitor", "resolution"},
        'battery': {"battery", "life", "charge", "last", "power"},
        'company': {"company", "product", "hp", "toshiba", "dell", "apple", "lenovo"},
        'mouse': {"mouse", "touch", "track", "button", "pad"},
        'software': {"software", "programs", "applications", "itunes", "photo"},
        'keyboard': {"keyboard", "key", "space", "type", "keys"}
    },
    'restaurant-3': {
        'food': {"food", "spicy", "sushi", "pizza", "taste", "delicious", "bland", "drinks", "flavourful"},
        'place': {"ambience", "atmosphere", "seating", "surroundings", "environment", "location", "decoration", "spacious", "comfortable", "place"},
        'service': {"tips", "manager", "waitress", "rude", "forgetful", "host", "server", "service", "quick", "staff"}
    },
    'restaurant-5': {
        'location': {"location", "street", "block", "river", "avenue"},
        'drinks': {"drinks", "beverage", "wines", "margaritas", "sake"},
        'food': {"food", "spicy", "sushi", "pizza", "taste"},
        'ambience': {"ambience", "atmosphere", "room", "seating", "environment"},
        'service': {"service", "tips", "manager", "waitress", "servers"}
    },
    'kto': {
        'service': {"scanner", "zelfscan", "handscanner", "houders", "afrekenen", "kassa", "rij", "wachttijd", "servicebalie", "wachttijd", "service", "balie", "servicedesk"},
        'kwaliteit': {"vers", "kwaliteit", "rot", "beschimmeld", "houdbaarheid"},
        'app': {"app", "looproute"},
        'winkel': {"Parkeren","fiets","buiten","parkeerplaats","fietsenstalling","hangjongeren","garage", "smal", "indeling", "toilet", "ruimte", "opzet", "uitbreiden", "ingang", "inpaktafel", "wifi", "internet", "muziek"},
        'assortiment': {"assortiment", "aanbod", "biologische", "vegan"},
        'beschikbaarheid': {"beschikbaarheid", "uitverkocht", "voorraad", "verkrijgbaar", "leeg", "aanvullen", "brood", "afbakken", "bonus"},
        'personeel': {"klantvriendelijk", "begroeten", "behulpzaam", "vriendelijk", "hulp", "personeel", "weinig", "vragen", "aanwezig", "aanspreken", "versperren", "gangpad", "blokkeren", "obstakels", "vakkenvullers"},
        'corona': {"corona", "covid", "mondkapje", "desinfecteren", "coronamaatregelen", "maatregelen", "winkelwagen"},
        'opgeruimd': {"rommelig", "smerig", "afval", "vies", "opruimen", "schoon", "opgeruimd", "spiegelen", "prullenbak"},
        'prijzen': {"prijs", "duur", "goedkoper", "35%"},
        'overig': {"duurzaam", "bloemen", "bezorgen", "reeds", "vermeld", "enquete", "eerdere", "opmerking"}
    }
}

sentiment_category_mapper = {
    'laptop': ['negative', 'positive'],
    'restaurant-3': ['negative', 'positive'],
    'restaurant-5': ['negative', 'positive'],
    'kto': ['negative', 'positive']
}

sentiment_seed_mapper = {
    'laptop': {
        'positive': {"good", "great", 'nice', "excellent", "perfect"},
        'negative': {"bad", "terrible", "horrible", "disappointed ", "awful"}
    },
    'restaurant-3': {
        'positive': {"good", "great", 'nice', "excellent", "perfect"},
        'negative': {"bad", "terrible", "horrible", "disappointed ", "awful"}
    },
    'restaurant-5': {
        'positive': {"good", "great", 'nice', "excellent", "perfect"},
        'negative': {"bad", "terrible", "horrible", "disappointed ", "awful"}
    },
    'kto': {
        'positive': {"goed", 'uitstekend', "excellent", "perfect"},
        'negative': {"slecht", "betere", "teleurgesteld", "verschrikkelijk", "langzaam", "kapot", "klacht", "vies"}
    }
}

batch_size = 24
validation_data_size = 150
hyper_validation_data_size = 0.8
learning_rate = 1e-5
epochs = 3
beta1 = 0.9
beta2 = 0.999
gamma1 = 3
gamma2 = 3
