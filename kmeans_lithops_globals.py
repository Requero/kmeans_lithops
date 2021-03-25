import time
import random
import numpy as np
import lithops.multiprocessing as mplths
import multiprocessing as mp
import itertools
from scipy.spatial.distance import cdist

lock = None
barrier = None
global_centroids = None
global_delta = None
worker_stats = None

def main(datapoints_per_file, dimensions,clusters,parallelism,number_of_iterations,threshold,local, localhost= False):

    manager = mplths.Manager()# if local == True else mplths.Manager()
    global lock
    lock = manager.Lock()
    global barrier
    barrier = manager.Barrier(parallelism)
    global global_centroids
    global_centroids = manager.dict()
    global global_delta
    global_delta = manager.dict({"delta": 1, "delta_c": 0, 
                                "delta_temp": 0, "delta_st": 0})
    global worker_stats
    worker_stats = manager.list()     # in seconds
    
    
    if True == False:
        # TEST Crentroids
        centroids = GlobalCentroids(2, 2)
        centroids.random_init(4)
        print(centroids.get_centroids())
        centroids.update(np.array([[1.2, 1, 1, 1], [2, 2, 2, 2]]), [2, 2])
        centroids.update(np.array([[2, 2, 2, 2.2], [1, 1, 1, 1]]), [2, 2])
        print(centroids.get_centroids())
        # TEST Delta
        delta = GlobalDelta(2)
        delta.init()
        print(delta.get_delta())
        delta.update(1, 2)
        delta.update(0, 2)
        print(delta.get_delta())
        return

    # Initialize global objects
    centroids = GlobalCentroids(clusters, parallelism)
    centroids.random_init(dimensions)
    delta = GlobalDelta(parallelism)
    delta.init()

    start_time = time.time()
    iterator = list(itertools.product(range(parallelism),
    [datapoints_per_file],[dimensions], [clusters], 
    [parallelism], [number_of_iterations],[threshold]))

    if local==True:
        pool = mp.Pool(processes=parallelism)
        pool.imap(run, iterator)
        pool.close()
        pool.join()
    else:
        if localhost == True:
            pool = mplths.Pool(processes=parallelism,initargs = ('localhost'))
            pool.imap(run, iterator)
            pool.close()
            pool.join()
            
        else:
            pool = mplths.Pool(processes=parallelism, initargs=('serverless'))
            pool.imap(run, iterator)
            pool.close()
            pool.join()
    # Parse results
    times = []
    for b in worker_stats:
        # Iterations time is second breakdown and last
        times.append(b[-1]-b[2])
    
    avg_time = sum(times) / len(times)
    print(f"Total k-means time: {time.time() - start_time} s")
    print(f"Average iterations time: {avg_time} s")
    print(global_centroids.items())
    #import matplotlib.pyplot as plt
    #for k,v in global_centroids.items():
    #    plt.scatter(v['centroids'][0], v['centroids'][1], alpha = 0.6, s=10)
    #plt.show()
    
    where = ''
    if local ==True:
        where = 'local'
    elif localhost == True:
        where ='localhost'
    else:
        where = 'serverless'

    with open('time_break_globals_'+where+'_'+str(datapoints_per_file)+'_'+str(dimensions)+'_'+str(parallelism)+'_'+str(clusters)+'_'+str(number_of_iterations)+'.txt', 'a+') as f:
        f.write(f"{time.time() - start_time}\n")
        f.write(f"{avg_time}\n")
        for item in worker_stats:
            f.write(f"{item}\n")
        
def run(arguments):
    global lock
    w_id, points_per_file, dimensions, clusters, parallelism, number_of_iterations,threshold = arguments
    worker = Worker( w_id, parallelism * points_per_file, dimensions, parallelism,clusters, number_of_iterations, threshold)
    worker_breakdown = worker.run()
    lock.acquire()
    worker_stats.append(worker_breakdown)
    lock.release()

class GlobalCentroids(object):
    def __init__(self,clusters, parallelism):
        self.num_clusters = clusters
        self.parallelism = parallelism

    @staticmethod
    def centroid_key(centroid):
        return "centroid" + str(centroid)

    def random_init(self,num_dimensions):
        global global_centroids
        global lock 
        lock.acquire()
        #temp_dict = {}
        for i in range(self.num_clusters):
            random.seed(1002+i)
            #temp_dict =  {}
            #temp_dict['centroids'] = np.random.normal(0, 1,num_dimensions)
            global_centroids[self.centroid_key(i)] = {
                'centroids':np.random.normal(0, 1,num_dimensions),
                'counters':0}
        #global_centroids = temp_dict
        lock.release()

    def update(self,coordinates, sizes):
        for k in range(self.num_clusters):
            self._update_centroid(k, coordinates[k], int(sizes[k]))

    def _update_centroid(self, cluster_id, coordinates, size):
        # Current centroid
        global global_centroids
        global lock
        centroid_k = self.centroid_key(cluster_id)
        lock.acquire()
        #print(global_centroids)
        temp_dict = global_centroids[centroid_k]#.copy()
        #print(temp_dict.items())
        #print(temp_dict)
        #lock.release()
        #barrier.wait()
        #count = temp_dict['counters']

        # First update
        #if count == 0:
        if temp_dict['counters'] == 0:
            #centroid_temp = np.zeros(len(temp_dict['centroids']))
            #sizes_temp =  0
            temp_dict['centroids_temp'] = np.zeros(len(temp_dict['centroids']))
            temp_dict['sizes_temp'] =  0
        #else:
            #centroid_temp = temp_dict['centroids_temp']
            #sizes_temp =temp_dict['sizes_temp']

        #centroid_temp =  centroid_temp + coordinates
        #sizes_temp += size
        #count += 1
        #temp_dict['centroids_temp'] = centroid_temp
        #temp_dict['sizes_temp'] = sizes_temp

        temp_dict['centroids_temp'] += coordinates
        temp_dict['sizes_temp']  += size
        temp_dict['counters'] += 1

        #if count== self.parallelism:
        if temp_dict['counters'] == self.parallelism:
            #if sizes_temp != 0:
            if temp_dict['sizes_temp'] != 0:
                #temp_dict['centroids'] = centroid_temp/sizes_temp
                temp_dict['centroids'] = temp_dict['centroids_temp']/temp_dict['sizes_temp']
            temp_dict['counters'] = 0
        #else: 
        #    temp_dict['counters'] = count
        #lock.acquire()
        #print(temp_dict.items())
        global_centroids[centroid_k] = temp_dict
        lock.release()

    def get_centroids(self):
        global lock
        lock.acquire()
        global global_centroids
        temp_dict = global_centroids#.copy()
        b = [temp_dict[self.centroid_key(k)]['centroids'] for k in range(self.num_clusters)]
        lock.release()
        return b

class GlobalDelta(object):
    def __init__(self, parallelism):
        self.parallelism = parallelism

    def init(self):
        # Hardcoded keys
        global lock
        
        lock.acquire()
        global global_delta
        temp_dict = global_delta
        temp_dict["delta"] = 1
        temp_dict["delta_c"] = 0
        temp_dict["delta_temp"] = 0
        temp_dict["delta_st"] = 0
        global_delta = temp_dict 
        #global_delta = {"delta": 1, "delta_c": 0, "delta_temp": 0, "delta_st": 0}
        lock.release()

    def get_delta(self):
        global lock
        global global_delta
        lock.acquire()
        delta = global_delta["delta"]
        lock.release()
        return delta

    def update(self,delta, num_points): 
        global lock
        global global_delta
        #global barrier
        lock.acquire()
        temp_dict = global_delta#.copy()
        #print(global_delta)
        #print(temp_dict)
        #print(temp_dict.items())
        #lock.release()
        #barrier.wait()
        #count = temp_dict["delta_c"]
        #tmp_delta = temp_dict["delta_temp"]
        #tmp_points = temp_dict["delta_st"]
        
        #count += 1
        #tmp_delta += delta
        #tmp_points += num_points
        temp_dict["delta_c"] += 1
        temp_dict["delta_temp"] += delta
        temp_dict["delta_st"] += num_points

        #if count == self.parallelism:
        if temp_dict["delta_c"] == self.parallelism:
            #temp_dict["delta"] = tmp_delta / tmp_points
            temp_dict["delta"] = temp_dict["delta_temp"] / temp_dict["delta_st"]
            temp_dict["delta_c"] = 0
            temp_dict["delta_temp"] = 0
            temp_dict["delta_st"] = 0
        #else:
            #temp_dict["delta_c"] = count
            #temp_dict["delta_temp"] = tmp_delta
            #temp_dict["delta_st"] = tmp_points
        #lock.acquire()
        #print(temp_dict)
        #print(temp_dict.items())
        #global_delta["delta"] = temp_dict["delta"]
        #global_delta["delta_c"] = temp_dict["delta_c"]
        #global_delta["delta_temp"] = temp_dict["delta_temp"]
        #global_delta["delta_st"] = temp_dict["delta_st"]
        global_delta = temp_dict#.copy()
        lock.release()

class Worker(object):

    def __init__(self, 
        worker_id, data_points, dimensions, 
        parallelism,clusters, max_iters,threshold):
        self.worker_id = worker_id
        self.num_dimensions = dimensions
        self.num_clusters = clusters
        self.max_iterations = max_iters
        self.partition_points = int(data_points / parallelism)
        self.parallelism = parallelism
        self.start_partition = self.partition_points * self.worker_id
        self.end_partition = self.partition_points * (worker_id + 1)

        self.correct_centroids = None
        self.local_partition = None
        self.local_centroids = None
        self.local_sizes = None
        self.local_membership = None

        self.global_delta = GlobalDelta( self.parallelism)
        self.global_centroids = GlobalCentroids(self.num_clusters,
                                                self.parallelism)
        self.threshold = threshold

    def run(self):
        breakdown = []
        breakdown.append(time.time())

        self.load_dataset()
        #print(self.local_partition)

        self.local_membership = np.zeros([self.local_partition.shape[0]])

        # barrier before starting iterations, to avoid different execution times
        global barrier
        barrier.wait()
        init_time = time.time()
        breakdown.append(init_time)
        iter_count = 0
        global_delta_val = 1
        while ((iter_count < self.max_iterations) and 
                (global_delta_val > self.threshold)):

            # Get local copy of global objects
            self.correct_centroids = self.global_centroids.get_centroids()
            breakdown.append(time.time())

            # Reset data structures that will be used in this iteration
            self.local_sizes = np.zeros([self.num_clusters])
            self.local_centroids = np.zeros([self.num_clusters, self.num_dimensions])

            # Compute phase, returns number of local membership modifications
            delta = self.compute_clusters()
            breakdown.append(time.time())

            # Update global objects
            self.global_delta.update(delta, self.local_partition.shape[0])
            self.global_centroids.update(self.local_centroids, self.local_sizes)

            breakdown.append(time.time())
            barrier.wait()
            breakdown.append(time.time())
            global_delta_val = self.global_delta.get_delta()
            iter_count += 1

        breakdown.append(time.time())
        iteration_time = breakdown[-1] - init_time
        return breakdown

    def load_dataset(self):
        import random
        self.local_partition = np.random.randn(self.partition_points ,self.num_dimensions)
        #self.local_partition = np.random.randn(self.end_partition -self.start_partition ,self.num_dimensions)

    def compute_clusters(self):
        # delta = 0
        # for i in range(0, self.partition_points):
        #     point = self.local_partition[i]
        #     distances = ((point - self.correct_centroids) ** 2).sum(axis=1)
        #     cluster = np.argmin(distances)
        #
        #     # add new point to local centroid
        #     self.local_centroids[cluster] += point
        #     self.local_sizes[cluster] += 1
        #
        #     # If now point is a member of a different cluster
        #     if self.local_membership[i] != cluster:
        #         delta += 1
        #         self.local_membership[i] = cluster
        #
        # return delta
        points = self.local_partition
        centroids = self.correct_centroids
        dists = cdist(points, centroids, 'sqeuclidean')
        # for each point, id of closest cluster
        min_dist_cluster_id = dists.argmin(1)  # aka memberships
        # count of points to each cluster
        self.local_sizes = np.bincount(min_dist_cluster_id,
                                       minlength=self.num_clusters)
        # sum of points to each cluster
        cluster_ids = np.unique(min_dist_cluster_id)
        for cluster_id in cluster_ids:
            points[min_dist_cluster_id == cluster_id] \
                .sum(axis=0, out=self.local_centroids[cluster_id])
        # check changes in membership
        new_memberships = min_dist_cluster_id
        delta = np.count_nonzero(
            np.add(new_memberships, - self.local_membership))
        self.local_membership = new_memberships
        return delta

if __name__ == "__main__":
    main(695866,100,25,10,10,0.00001, False, False)
    #main(100,2,2,2,10,0.00001, False, False)
