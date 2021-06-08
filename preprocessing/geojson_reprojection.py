import copy
import json

from shapely.geometry import Polygon
from tqdm import tqdm
from pyproj import Transformer



if __name__ == '__main__':
    # rpg_file = '/mnt/71A36E2C77574D51/donnees/out/GEOJSON/lpis_stable_all_years.geojson'
    rpg_file = 'C:/Users/felix/OneDrive/Bureau/test/out/lpis_stable_all_years.geojson'
    print('Reading RPG . . .')
    with open(rpg_file) as f:
        data = json.load(f)
        data_reproj = copy.deepcopy(data)
        transformer = Transformer.from_crs("epsg:2154", "epsg:32631")
        for idx, f in enumerate(tqdm(data['features'])):
            for idx2, g in enumerate(f['geometry']['coordinates'][0]):
                x, y = transformer.transform(g[0], g[1])
                data_reproj['features'][idx]['geometry']['coordinates'][0][idx2][0] = x
                data_reproj['features'][idx]['geometry']['coordinates'][0][idx2][1] = y
        with open('C:/Users/felix/OneDrive/Bureau/test/out/lpis_stable_all_years_reprojected.json', 'w') as outfile:
            json.dump(data_reproj, outfile)