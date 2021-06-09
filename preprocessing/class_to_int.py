import json

labels = "/home/FQuinton/Bureau/data_pse/META/labels_str.json"

labels_int = {'CODE9_2018':{},
                'CODE9_2019':{},
                'CODE9_2020':{},
              }
with open(labels) as f:
    data = json.load(f)
    for key, value in data['CODE9_2018'].items():
        if value == "GRASS_PRE":
           labels_int['CODE9_2018'][key] = 1
        elif value == "TRITICA_W":
           labels_int['CODE9_2018'][key] = 2
        elif value == "MAIZE":
           labels_int['CODE9_2018'][key] = 3
        elif value == "RYE_W":
           labels_int['CODE9_2018'][key] = 4
        elif value == "WHEAT_B_W":
           labels_int['CODE9_2018'][key] = 5
        elif value == "RAPE_W":
           labels_int['CODE9_2018'][key] = 6
        elif value == "BARLEY_W":
           labels_int['CODE9_2018'][key] = 7
        elif value == "SUNFLOWER":
           labels_int['CODE9_2018'][key] = 8
        elif value == "VINEYARD":
           labels_int['CODE9_2018'][key] = 9
        elif value == "SOYBEAN":
           labels_int['CODE9_2018'][key] = 10
        elif value == "SORGHUM":
           labels_int['CODE9_2018'][key] = 11
        elif value == "WHEAT_D_W":
           labels_int['CODE9_2018'][key] = 12
        elif value == "ALFALFA":
           labels_int['CODE9_2018'][key] = 13
        elif value == "OAT_W":
           labels_int['CODE9_2018'][key] = 14
        elif value == "LEGU_FODD":
           labels_int['CODE9_2018'][key] = 15
        elif value == "CEREAL_MX":
           labels_int['CODE9_2018'][key] = 16
        elif value == "FL_FR_VEG":
           labels_int['CODE9_2018'][key] = 17
        elif value == "OAT_S":
           labels_int['CODE9_2018'][key] = 18
        elif value == "POTATO":
           labels_int['CODE9_2018'][key] = 19
        elif value == "BARLEY_S":
           labels_int['CODE9_2018'][key] = 20
        elif value == "WOOD_PAS":
           labels_int['CODE9_2018'][key] = 21
        elif value == "NO_EX_SUR":
           labels_int['CODE9_2018'][key] = 22
        elif value == "SPELT":
           labels_int['CODE9_2018'][key] = 23
    for key, value in data['CODE9_2019'].items():
        if value == "GRASS_PRE":
           labels_int['CODE9_2019'][key] = 1
        elif value == "TRITICA_W":
           labels_int['CODE9_2019'][key] = 2
        elif value == "MAIZE":
           labels_int['CODE9_2019'][key] = 3
        elif value == "RYE_W":
           labels_int['CODE9_2019'][key] = 4
        elif value == "WHEAT_B_W":
           labels_int['CODE9_2019'][key] = 5
        elif value == "RAPE_W":
           labels_int['CODE9_2019'][key] = 6
        elif value == "BARLEY_W":
           labels_int['CODE9_2019'][key] = 7
        elif value == "SUNFLOWER":
           labels_int['CODE9_2019'][key] = 8
        elif value == "VINEYARD":
           labels_int['CODE9_2019'][key] = 9
        elif value == "SOYBEAN":
           labels_int['CODE9_2019'][key] = 10
        elif value == "SORGHUM":
           labels_int['CODE9_2019'][key] = 11
        elif value == "WHEAT_D_W":
           labels_int['CODE9_2019'][key] = 12
        elif value == "ALFALFA":
           labels_int['CODE9_2019'][key] = 13
        elif value == "OAT_W":
           labels_int['CODE9_2019'][key] = 14
        elif value == "LEGU_FODD":
           labels_int['CODE9_2019'][key] = 15
        elif value == "CEREAL_MX":
           labels_int['CODE9_2019'][key] = 16
        elif value == "FL_FR_VEG":
           labels_int['CODE9_2019'][key] = 17
        elif value == "OAT_S":
           labels_int['CODE9_2019'][key] = 18
        elif value == "POTATO":
           labels_int['CODE9_2019'][key] = 19
        elif value == "BARLEY_S":
           labels_int['CODE9_2019'][key] = 20
        elif value == "WOOD_PAS":
           labels_int['CODE9_2019'][key] = 21
        elif value == "NO_EX_SUR":
           labels_int['CODE9_2019'][key] = 22
        elif value == "SPELT":
           labels_int['CODE9_2020'][key] = 23
    for key, value in data['CODE9_2020'].items():
        if value == "GRASS_PRE":
            labels_int['CODE9_2020'][key] = 1
        elif value == "TRITICA_W":
            labels_int['CODE9_2020'][key] = 2
        elif value == "MAIZE":
            labels_int['CODE9_2020'][key] = 3
        elif value == "RYE_W":
            labels_int['CODE9_2020'][key] = 4
        elif value == "WHEAT_B_W":
            labels_int['CODE9_2020'][key] = 5
        elif value == "RAPE_W":
            labels_int['CODE9_2020'][key] = 6
        elif value == "BARLEY_W":
            labels_int['CODE9_2020'][key] = 7
        elif value == "SUNFLOWER":
            labels_int['CODE9_2020'][key] = 8
        elif value == "VINEYARD":
            labels_int['CODE9_2020'][key] = 9
        elif value == "SOYBEAN":
            labels_int['CODE9_2020'][key] = 10
        elif value == "SORGHUM":
            labels_int['CODE9_2020'][key] = 11
        elif value == "WHEAT_D_W":
            labels_int['CODE9_2020'][key] = 12
        elif value == "ALFALFA":
            labels_int['CODE9_2020'][key] = 13
        elif value == "OAT_W":
            labels_int['CODE9_2020'][key] = 14
        elif value == "LEGU_FODD":
            labels_int['CODE9_2020'][key] = 15
        elif value == "CEREAL_MX":
            labels_int['CODE9_2020'][key] = 16
        elif value == "FL_FR_VEG":
            labels_int['CODE9_2020'][key] = 17
        elif value == "OAT_S":
            labels_int['CODE9_2020'][key] = 18
        elif value == "POTATO":
            labels_int['CODE9_2020'][key] = 19
        elif value == "BARLEY_S":
            labels_int['CODE9_2020'][key] = 20
        elif value == "WOOD_PAS":
            labels_int['CODE9_2020'][key] = 21
        elif value == "NO_EX_SUR":
            labels_int['CODE9_2020'][key] = 22
        elif value == "SPELT":
            labels_int['CODE9_2020'][key] = 23
with open('/home/FQuinton/Bureau/data_pse/META/labels_int.json',
          'w') as file:
    file.write(json.dumps(labels_int, indent=4))