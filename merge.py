# Script to merge hdf5 chunk files to one and update info.json accordingly

from tqdm import tqdm
import h5py 

parser = argparse.ArgumentParser()
parser.add_argument('--name', type = str, help = "features directory name")
args = parser.parse_args()

# Format specification for features files
spec = {
	"spatial": {"features": (108079, 2048, 7, 7)}
	"objects": {"features": (108077, 100, 2048),
							(108077, 100, 4)}
}
chunkSize = 10000

# Merge hdf5 files
with h5py.File("{name}/gqa_{name}.h5".format(name = args.name)) as out:
	datasets  = {}
	for dname in spec[args.name]:
		datasets[dname] = out.create_dataset(dname, spec[args.name][dname])

	for i in tqdm(range(11)):
		with h5py.File("{name}/gqa_{name}_{index}.h5".format(name = args.name, index = i)) as chunk: 
			for dname in spec[args.name]:
				datasets[dname][(i * chunkSize):((i + 1) * chunkSize)] = chunk[dname][:]

# Update info file
with open("{name}/gqa_{name}_info.json".format(name = args.name)) as infoIn:
	info = json.load(infoIn)
	for imageId in info:
		info[imageId]["index"] = info[imageId]["file"] * chunkSize + info[imageId]["idx"]
		del info[imageId]["file"]

	with open("{name}/gqa_{name}_merged_info.json".format(name = args.name)) as infoOut:
		json.dump(info, infoMerged)