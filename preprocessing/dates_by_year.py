import json

dates = "/home/FQuinton/Bureau/data_pse/META/dates.json"

dt = {"2018": {},
      "2019": {},
      "2020": {}}
with open(dates) as f:
    data = json.load(f)
    for key, value in data.items():
        if value[0:4] == "2018":
            dt["2018"][key] = value[4:]
    for key, value in data.items():
        if value[0:4]== "2019":
            dt["2019"][int(key) - 36] = value[4:]
    for key, value in data.items():
        if value[0:4]== "2020":
            dt["2020"][int(key) - 63] = value[4:]

with open('/home/FQuinton/Bureau/data_pse/META/dates_by_year.json',
          'w') as file:
    file.write(json.dumps(dt, indent=4))