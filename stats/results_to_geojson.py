import json
from tqdm import tqdm

#Permet d'obtenir un geojson à partir des résultats d'un modèle

rpg_file = '/mnt/71A36E2C77574D51/preprocess/geojson/lpis_stable_all_years_reprojected.json'
rpg_file = '/mnt/71A36E2C77574D51/felix/stage_2021_rotation/GEOJSON/lpis_stable_all_years.geojson'
pred = '/home/FQuinton/Bureau/data_pse/META/stats/compare_pred_value2.json'

def parse_rpg(rpg_file):
    """Reads rpg and returns a dict of pairs (ID_PARCEL : Polygon) and a dict of dict of labels
     {label_name1: {(ID_PARCEL : Label value)},
      label_name2: {(ID_PARCEL : Label value)}
     }
     """
    # Read rpg file
    print('Reading RPG . . .')
    with open(rpg_file) as f:
        data = json.load(f)

        dic = {}
        # Get list of polygons
        for parcel in tqdm(data['features']):
            dic[parcel['properties']['UID_2018']] = {
                'geometry' : parcel['geometry']
            }
        geodic = {'type':'FeatureCollection','crs':{'type':'name', 'properties':{'name':'urn:ogc:def:crs:EPSG::2154'}},'features':[]}
        with open(pred) as g:
            predictions = json.load(g)
            for key,value in tqdm(predictions.items()):
                parcel = {'type':'Feature','properties':{
                    'classe_2018': value['classe_2018'],
                    'classe_2019': value['classe_2019'],
                    'classe_2020': value['classe_2020'],
                    'pred_mdec_2018': value['pred_lab_max_2018'],
                    'pred_mdec_2019': value['pred_lab_max_2019'],
                    'pred_mdec_2020': value['pred_lab_max_2020'],
                    'pred_mall_2018': value['pred_global_max_2018'],
                    'pred_mall_2019': value['pred_global_max_2019'],
                    'pred_mall_2020': value['pred_global_max_2020']
                },
                'geometry':dic[key]['geometry']}
                geodic['features'].append(parcel)
            with open('/home/FQuinton/Bureau/comparaison_pred_verite_terrain2.geojson', 'w') as outfile:
                json.dump(geodic, outfile)

parse_rpg(rpg_file)