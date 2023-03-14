import numpy as np
def get_colors_array_with_clusters(samples, samples_with_clusters):
    colors_array = []
    cluster = -1
    for s in samples:
        for sample_with_cluster in samples_with_clusters:
            if sample_with_cluster[0] == s.all():
                cluster = sample_with_cluster[1]
                break
        print(cluster)
        #color = markers_and_colors[cluster][3]
        #colors_array.append(color)
    return colors_array


samples =np.array([np.array([1,2,3]), np.array([3,4,5]), np.array([6,7,8])])
#samples_with_clusters = [([1,2,3],0), ([3,4,5], 1), ([6,7,8],0)]
#get_colors_array_with_clusters(samples,samples_with_clusters)
#np.all(samples[0] == samples_with_clusters[0][0])
index = np.where(samples==np.array([6,7,8]))[0][0]
print(index)
print(samples[0])