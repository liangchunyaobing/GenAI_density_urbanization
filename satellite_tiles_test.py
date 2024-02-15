from satellite_tiles import MyDataset

# image_dir = "/home/gridsan/qwang/satellite_images/zoom17/"
# data_dir = "/home/gridsan/qwang/JTL-transit_shared/deep_hybrid_model/data/"


image_dir = "/home/gridsan/qwang/satellite_tiles_control/satellite_tiles/"
data_dir = "/home/gridsan/qwang/satellite_tiles_control/tile_descriptions.csv"
hint_dir = "/home/gridsan/qwang/satelite_tiles_control/skeleton_16/16/"


# dataset = MyDataset(image_dir, data_dir, demo=1)
# print(len(dataset))
dataset = MyDataset(image_dir, data_dir, hint_dir)

item = dataset[1234]
jpg = item['jpg']
txt = item['txt']
hint = item['hint']

print(txt)
print(jpg.shape)
print(hint.shape)
