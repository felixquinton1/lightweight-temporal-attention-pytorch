import json
from shapely.geometry import Polygon
from tqdm import tqdm
sizes = "/home/FQuinton/Bureau/data_pse/META/sizes.json"
geojson_dic = "/home/FQuinton/Bureau/data_pse/META/geojson_dic.json"

# geojson_dic = {}
# with open(geojson) as g:
#     geoj = json.load(g)['features']
#     for i in geoj:
#         geojson_dic[i['properties']['UID_2018']] = i['geometry']['coordinates'][0]
#
# with open('/home/FQuinton/Bureau/data_pse/META/geojson_dic.json',
#           'w') as file:
#     file.write(json.dumps(geojson_dic, indent=4))

geomfeat = {}
with open(sizes) as f:
    data = json.load(f)
    with open(geojson_dic) as g:
        geoj = json.load(g)
        for key, value in tqdm(data.items()):
            p = Polygon(geoj[key])
            bbox = abs(p.bounds[0] - p.bounds[2]) * abs(p.bounds[1] - p.bounds[3])
            geomfeat[key] = [p.length, p.area, p.area/bbox, p.area/p.length]

with open('/home/FQuinton/Bureau/data_pse/META/geomfeat.json',
          'w') as file:
    file.write(json.dumps(geomfeat, indent=4))
