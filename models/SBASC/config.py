config = {
    'domain': 'restaurant-3',
    'device': 'cuda',
    'base_path': '/dbfs/FileStore/kto/'
}
bert_mapper = {
    'laptop': 'activebus/BERT-DK_laptop',
    'restaurant-3': 'activebus/BERT-DK_rest',
    'restaurant-5': 'activebus/BERT-DK_rest',
    'kto': '/dbfs/FileStore/kto/kto/BERT-DK_kto'
}

sbert_mapper = {
    'laptop': 'all-mpnet-base-v2',
    'restaurant-3': 'all-mpnet-base-v2',
    'restaurant-5': 'all-mpnet-base-v2',
    'kto': 'paraphrase-multilingual-mpnet-base-v2'
}

path_mapper = {
    'laptop': '/dbfs/FileStore/kto/laptop',
    'restaurant-3': '/dbfs/FileStore/kto/restaurant-3',
    'restaurant-5': '/dbfs/FileStore/kto/restaurant-5',
    'kto': '/dbfs/FileStore/kto/kto'
}
aspect_category_mapper = {
    'laptop': ['support', 'os', 'display', 'battery', 'company', 'mouse', 'software', 'keyboard'],
    'restaurant-3': ['food', 'place', 'service'],
    'restaurant-5': ['location', 'drinks', 'food', 'ambience', 'service'],
    'kto': ['service', 'app', 'assortiment', 'beschikbaarheid', 'corona', 'kwaliteit', 'winkel', 'personeel',
            'opgeruimd', 'prijzen', 'overig']}

aspect_seed_sentence_mapper = {
    'restaurant-3': {
        'food': {
            "really good food and always have decent drink_specials .",
            "this is the best sushi buffet we ever had ",
            "the pizza at old chicago is actually pretty good",
            "hamburgers bland and buns dry and cold .",
            "the drinks were amazing as well , my husband and i will definitely be coming back !",
        },
        'place': {
            "great place to chill and enjoy some alone time with a cool ambience about the place !",
            "butterfields is a great environment and atmosphere with waffles on the ceiling , great neighborhood",
            "seating is limited-call ahead for a large_group . decor is updated and modern-will definitely be back again and again ",
            "this place is a total dump .",
            "the views and surroundings were spectacular . "
        },
        'service': {
            "the owner and staff are very friendly and communicative",
            "our server was rude , slow and snapped her fingers when we needed water # # # refills",
            "the service was extremely good as our water cups rarely ran low and the manager was very helpful in getting the bill comped .",
            "and the service is by far the worst i 've ever experienced",
            "the service is very good and the servers are very nice . i brought my sorority sisters to celebrate a bachelorette and the food was served quickly ."
            "the waitresses were friendly and attentive to our questions/comments ."
        }
    },
    'restaurant-5': {
           'location': {
               "convenient location to my apartment ."
               "the views and surroundings were spectacular . ",
               "we had a tremendous view of the river .",
               "located at the end of a magnificent block .",
               "the outdoor atmosphere of sitting on the sidewalk watching the world go by 50 feet away on 6th avenue on a cool evening was wonderful ."
           },
           'food': {
               "the food is good",
               "this is the best sushi buffet we ever had ",
               "all the dishes tasted the same .",
               "the pizza at old chicago is actually pretty good",
               "hamburgers bland and buns dry and cold .",
               # "the tacos were so delicious that when i started eating , i texted my husband to go there on his way home from work ."
           },
           'drinks': {
               "the drinks were amazing as well , my husband and i will definitely be coming back !",
               "they also have nice selection of beverages like fresh teas and lemonade .",
               "the wine list was pretty extensive and had nice , good wines .",
               "really bad margaritas !",
               "asahi bombers for $ 5- $ 6 -- they also have delicious plum sake ."
           },
           'ambience': {
               "great place to chill and enjoy some alone time with a cool ambience about the place !",
               # "butterfields is a great environment and atmosphere with waffles on the ceiling , great neighborhood",
               "seating is limited-call ahead for a large group . decor is updated and modern-will definitely be back again and again ",
               "this place is a total dump .",
               "spacious booths and tables that do n't feel cramped .",
               "the atmosphere at toast was very warm and comfortable",
               },
           'service': {
               "the owner and staff are very friendly and communicative",
               "our server was rude , slow and snapped her fingers when we needed water # # # refills",
               "the service was extremely good as our water cups rarely ran low and the manager was very helpful in getting the bill comped .",
               "and the service is by far the worst i 've ever experienced",
               # "the service is very good and the servers are very nice . i brought my sorority sisters to celebrate a bachelorette and the food was served quickly ."
               "the waitresses were friendly and attentive to our questions/comments ."}
        },
    'laptop': {
            'support': {
                "that 'll give you 3 years service and support the alternative to being stuck with a lemon six months after purchase .",
                "dell 's customer service has been quite unimpressive of late .",
                "battery expension is not covered by warranty and out of warranty repairs for portables are not cheap",
                "apple support service is really good",
                "i sent it in for warranty work and sony wants close to 700 to replace it ."},
            'os': {
                "windows 8 is a really poor os .",
                "this computer runs on windows xp and windows 7 and is fast",
                "the worst thing is that it is running windows 8",
                "os x version 10 is a much better operating system than windows .",
                "the operating system is really intuitive",
                },
            'display': {
                "several of our early testers are complaining about the ultra glossy ( or glassy ) screen and the quality of the display itself .",
                "the screen is excellent .",
                "my only concern was the poor led display quality when viewing at an angle colors appear washed out when viewed at an angle",
                "the screen is clear and the resolution is very nice ."},
            'battery': {
                "the battery says that it lasts for 3 1 2 hours when its fully charged , but it does n't seem to last that long .",
                "battery life it 's a joke compared to today 's 'light and thin' ones",
                "it does not hold a charge for too long but that is not an issue for me",
                "the standard battery in normal usage last me an hour to an hour and a half .",
                "the laptop loses power much more quickly than i would like ."},
            'company': {
                "you should buy apple products its a great business",
                "changed my mind about this company to a very positive experience .",
                "I really like hp .",
                "I hate apple as a company .",
                "I will never buy from lenovo again"
                },
            'mouse': {
                "i did order a wireless mouse",
                "the touch pad is also a bit difficult to get used to .",
                "decent good track pad with gestures built in",
                "mouse is nice touch pad texture is not my favourite prefer a simple matte feel but is still good "},
            'software': {
                "it 's neat that this came with a lot software already installed",
                "i really like all the apps in the mac app store ",
                "the software sometimes crashes",
                "i mostly use this to download music from itunes or amazon .",
                "i dont like all the junkware that comes installed .",
                "itunes is a really good program",
                "the extensive app store that google provides is great ."
              },
            'keyboard': {
                "i love the keyboard that light up at night or when it hard for you to see the keys .",
                "some key like backspace and enter are small .",
                " which i ca n't work with i 'm a touch typist , and the space bar does n't work often when i hit it with my thumb as i 'm typing .",
                "the keyboard has a good click without being stiff to type .",
                "keys work well"}
        }
}

sentiment_category_mapper = {
    'laptop': ['negative', 'positive'],
    'restaurant-3': ['negative', 'positive'],
    'restaurant-5': ['negative', 'positive'],
    'kto': ['negative', 'positive']
}

sentiment_seed_sentence_mapper = {
    'restaurant-3': {
        'positive': {
            "service was great",
            "every bite was super tender and the bun was nice and fresh",
            "friendly staff and warm ambiance",
            "we also ordered pita and hummus and that was delicious",
            "i really loved everything about it",
            "decently priced food and fast delivery",
            "fast delivery and terrific food",
            "very clean , great staff , good food",
            "this restaurant is clean with super friendly staff and awesome food"
        },
        'negative': {
            "i 'm sad to say i ca n't find anything positive to say about the experience",
            "the service over the past year is terrible",
            "any day of the week it 's bad",
            "horrible food . wretched service . vastly overpriced . i would rather starve than eat here again ."
            "horrible , over priced food",
            "the service was awful",
            "food was awful , group of 16 and none of us were please , everything was either cold or tasteless , i almost believe it was from the awl-mart frozen section",
            "and our table was sticky and smelled of beer",
            "not the best experience with the foods . our hot foods were served cold"
            "service was slow"}
    },
    'restaurant-5': {
        'positive': {
            "service was great",
            "every bite was super tender and the bun was nice and fresh",
            "friendly staff and warm ambiance",
            "we also ordered pita and hummus and that was delicious",
            "i really loved everything about it",
            "decently priced food and fast delivery",
            "fast delivery and terrific food",
            "very clean , great staff , good food",
            "this restaurant is clean with super friendly staff and awesome food"
        },
        'negative': {
            "i 'm sad to say i ca n't find anything positive to say about the experience",
            "the service over the past year is terrible",
            "any day of the week it 's bad",
            "horrible food . wretched service . vastly overpriced . i would rather starve than eat here again ."
            "horrible , over priced food",
            "the service was awful",
            "food was awful , group of 16 and none of us were please , everything was either cold or tasteless , i almost believe it was from the awl-mart frozen section",
            "and our table was sticky and smelled of beer",
            "not the best experience with the foods . our hot foods were served cold"
            "service was slow"}
    },
    'laptop': {
        'positive': {
            "this is a very good laptop .",
            "what sets it out as the perfect family laptop the touch screen is also just great for kids who are exposed to tablets and other touch screen devices .",
            'the touchpad is made of a nice frictionless material and it is nice and roomy .',
            "battery life is also excellent .",
            "i 've had three different hp laptops over the years , and honestly , i 've never been disappointed with them ."},
        'negative': {
            "perhaps i got a bad battery , but i think its probably just a weak design .",
            "the closest i got to a mac was in school and i thought they were terrible .",
            "absolutely horrible display",
            "i am shocked and disappointed that this company does such a poor job of support . ",
            "the track pad is nothing short of awful and the mouse buttons are too stiff ."}
    },
}

batch_size = 24
validation_data_size = 150
hyper_validation_data_size = 0.8
learning_rate = 1e-5
epochs = 15
beta1 = 0.97
beta2 = 0.999
gamma1 = 0
gamma2 = 0.5
