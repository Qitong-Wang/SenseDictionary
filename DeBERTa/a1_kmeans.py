import pickle
import torch
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import argparse 
from sklearn.cluster import KMeans
def statistics(filter):
    mean_value = np.mean(filter)
    max_value = np.max(filter)

    # Calculate quantiles (25%, 50%, and 75%)
    q25 = np.percentile(filter, 25)
    q50 = np.percentile(filter, 50) 
    q75 = np.percentile(filter, 75)


    print(f"Mean: {mean_value}")
    print(f"Max: {max_value}")
    print(f"25th Percentile: {q25}")
    print(f"50th Percentile (Median): {q50}")
    print(f"75th Percentile: {q75}")



def run_kmeans(key, value,args):


    if value.shape[0] < args.k:
        if args.bf16:
            return  key, torch.tensor(value).to(torch.bfloat16)
        else:
            return  key, torch.tensor(value).to(torch.float16)

    kmeans = KMeans(n_clusters=int(args.k), random_state=0,n_init="auto").fit(value)
    centers = kmeans.cluster_centers_

    if args.bf16:
        return key, torch.tensor(centers).to(torch.bfloat16)
    else:
        return key, torch.tensor(centers).to(torch.float16)

 
   



if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument('--input_file', type=str, help="Path to the input file.")
    parser.add_argument('--output_file', type=str, help="Path to the output file.")
    parser.add_argument('--k', type=float, default=10, help="K for k-means.")
    parser.add_argument('--bf16', type=bool, default=False, help="Whether to use bfloat16 precision.")

    args = parser.parse_args()
    print(args)
    print("input file:",args.input_file)
    with open(args.input_file, 'rb') as handle: 
        data = pickle.load(handle)

   
    output_data = dict()

    cluster_centroids_data = dict()
    cluster_medoids_data = dict()
    import random
    idx = 0
    keys = list(data.keys())
    random.shuffle(keys)

    for key in tqdm(keys):

        value = data[key]
        output_key, output_value = run_kmeans(key,value,args)
        output_data[output_key] = output_value


 
    print("save to:",args.output_file)
    with open(args.output_file, 'wb') as handle:
        pickle.dump(output_data, handle)
    