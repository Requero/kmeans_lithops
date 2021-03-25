import time
import random
import numpy as np
import lithops.multiprocessing as mplths
import multiprocessing as mp
import itertools
from scipy.spatial.distance import cdist
import ctypes

def main(datapoints_per_file, dimensions,clusters,parallelism,number_of_iterations,threshold,local, localhost= False):

    lock = mplths.Lock()
    barrier = mplths.Barrier(parallelism)

    global_centroids = mplths.Array('d', clusters*dimensions, lock=lock)
    global_counters = mplths.Array('i', clusters, lock=lock)
    global_centroids_temp = mplths.Array('d', clusters*dimensions, lock=lock)
    global_sizes_temp = mplths.Array('i', clusters, lock=lock)
    global_delta = mplths.Array('i', 4, lock=lock)
    worker_stats = mplths.Array('d', parallelism, lock=lock)     # in seconds
    
    if True == False:
        # TEST Crentroids
        centroids = GlobalCentroids(2, 2)
        centroids.random_init(4, global_centroids, global_counters,lock)
        print(centroids.get_centroids())
        centroids.update(np.array([[1.2, 1, 1, 1], [2, 2, 2, 2]]), [2, 2])
        centroids.update(np.array([[2, 2, 2, 2.2], [1, 1, 1, 1]]), [2, 2])
        print(centroids.get_centroids())
        # TEST Delta
        delta = GlobalDelta(global_delta, 2)
        delta.init()
        print(delta.get_delta())
        delta.update(1, 2)
        delta.update(0, 2)
        print(delta.get_delta())
        return

    # Initialize global objects
    centroids = GlobalCentroids(clusters, parallelism)
    centroids.random_init(dimensions, global_centroids, global_counters,lock)
    delta = GlobalDelta(parallelism)
    delta.init(global_delta)

    start_time = time.time()
    iterator = list(itertools.product(range(parallelism),
    [global_centroids], [global_counters], [global_centroids_temp], 
    [global_sizes_temp],[global_delta],[worker_stats],[lock],[barrier],
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
    for b in worker_stats[:]:
        # Iterations time is second breakdown and last
        times.append(b[-1]-b[2])
    
    avg_time = sum(times) / len(times)
    print(f"Total k-means time: {time.time() - start_time} s")
    print(f"Average iterations time: {avg_time} s")
    print(global_centroids[:])
    print(global_counters[:])
    print(global_centroids_temp[:])
    print(global_sizes_temp[:])
    where = ''
    if local ==True:
        where = 'local'
    elif localhost == True:
        where ='localhost'
    else:
        where = 'serverless'

    with open('time_break_ctypes_'+where+'_'+str(datapoints_per_file)+'_'+str(dimensions)+'_'+str(parallelism)+'_'+str(clusters)+'_'+str(number_of_iterations)+'.txt', 'w+') as f:
        f.write(f"{time.time() - start_time}\n")
        f.write(f"{avg_time}\n")
        for item in worker_stats[:]:
            f.write(f"{item}\n")
        
def run(arguments):
    w_id, global_centroids, global_counters, global_centroids_temp, global_sizes_temp, global_deltas, \
    worker_stats, lock, barrier, points_per_file, dimensions, clusters, parallelism, number_of_iterations,threshold = arguments
    worker = Worker( w_id, parallelism * points_per_file, dimensions, parallelism,clusters, number_of_iterations, threshold)
    worker_breakdown = worker.run(global_centroids, global_counters, global_centroids_temp, global_sizes_temp, global_deltas, \
    lock, barrier)
    lock.acquire()
    worker_stats[w_id]=worker_breakdown
    lock.release()

class GlobalCentroids(object):
    def __init__(self,clusters, parallelism):
        self.num_clusters = clusters
        self.parallelism = parallelism

    @staticmethod
    def centroid_key(centroid):
        return "centroid" + str(centroid)

    def random_init(self, num_dimensions, global_centroids, global_counters, lock):
        lock.acquire()
        for i in range(self.num_clusters):
            random.seed(1002+i)
            global_centroids[i*num_dimensions:(i+1)*num_dimensions-1] = np.random.normal(0, 1,num_dimensions)
            global_counters[i] = 0
        lock.release()

    def update(self,coordinates, sizes,global_centroids, global_counters, global_centroids_temp, global_sizes_temp, lock):
        for k in range(self.num_clusters):
            self._update_centroid(k, coordinates[k], int(sizes[k]),global_centroids, 
            global_counters, global_centroids_temp, global_sizes_temp, lock)

    def _update_centroid(self, cluster_id, coordinates, size, global_centroids, 
            global_counters, global_centroids_temp, global_sizes_temp, lock):
        # Current centroid
        centroid_k = cluster_id #self.centroid_key(cluster_id)
        lock.acquire()
        #temp_dict = global_centroids[centroid_k]
        n = int(len(global_centroids)/self.num_clusters)
        count = global_counters[centroid_k]

        # First update
        if count == 0:
            centroid_temp = global_centroids_temp[centroid_k*n:(centroid_k+1)*n-1]
            sizes_temp =  0
        else:
            centroid_temp = global_centroids_temp[centroid_k*n:(centroid_k+1)*n-1]
            sizes_temp = global_sizes_temp[centroid_k]

        centroid_temp =  np.array(centroid_temp) + coordinates
        sizes_temp += size
        count += 1
        global_centroids_temp[centroid_k*n:(centroid_k+1)*n-1] = centroid_temp.tolist()
        global_sizes_temp[centroid_k] = sizes_temp
        
        if count== self.parallelism :
            if sizes_temp != 0:
                global_centroids_temp[centroid_k*n:(centroid_k+1)*n-1] = (centroid_temp/sizes_temp).tolist()
            global_counters[centroid_k] = 0
        else: 
            global_counters[centroid_k] = count
        lock.release()

    def get_centroids(self, global_centroids,lock):
        num_dimensions = int(len(global_centroids)/self.num_clusters)
        lock.acquire()
        b = np.array([global_centroids[k*num_dimensions:(k+1)*num_dimensions-1]for k in range(self.num_clusters)])
        lock.release()
        return b

class GlobalDelta(object):
    def __init__(self, parallelism):
        self.parallelism = parallelism

    def init(self, global_delta):
        # Hardcoded keys
        global_delta[:] = [1,0,0,0]

    def get_delta(self,global_delta,lock):
        lock.acquire()
        delta = global_delta[0]
        lock.release()
        return delta

    def update(self,global_delta, lock, delta, num_points): 
        lock.acquire()
        delta_c,delta_temp,delta_st=  global_delta[1:3] 
        count = delta_c
        tmp_delta = delta_temp
        tmp_points = delta_st
        
        count += 1
        tmp_delta += delta
        tmp_points += num_points

        if count == self.parallelism:
            global_delta[0] = tmp_delta / tmp_points
            global_delta[1] = 0
            global_delta[2] = 0
            global_delta[3] = 0
        else:
            global_delta[1] = count
            global_delta[2] = tmp_delta
            global_delta[3] = tmp_points
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

        self.global_delta = GlobalDelta(self.parallelism)
        self.global_centroids = GlobalCentroids(self.num_clusters,
                                                self.parallelism)
        self.threshold = threshold

    def run(self, global_centroids, global_counters, global_centroids_temp, global_sizes_temp, global_deltas, 
        lock, barrier):
        breakdown = []
        breakdown.append(time.time())

        self.load_dataset()
        #print(self.local_partition)

        self.local_membership = np.zeros([self.local_partition.shape[0]])

        # barrier before starting iterations, to avoid different execution times
        barrier.wait()
        init_time = time.time()
        breakdown.append(init_time)
        iter_count = 0
        global_delta_val = 1
        while ((iter_count < self.max_iterations) and 
                (global_delta_val > self.threshold)):

            # Get local copy of global objects
            self.correct_centroids = self.global_centroids.get_centroids(global_centroids,lock)
            breakdown.append(time.time())

            # Reset data structures that will be used in this iteration
            self.local_sizes = np.zeros([self.num_clusters])
            self.local_centroids = np.zeros([self.num_clusters, self.num_dimensions])

            # Compute phase, returns number of local membership modifications
            delta = self.compute_clusters()
            breakdown.append(time.time())

            # Update global objects
            self.global_delta.update(global_deltas,lock,delta, self.local_partition.shape[0])
            self.global_centroids.update(self.local_centroids, self.local_sizes,global_centroids, global_counters, global_centroids_temp, 
                global_sizes_temp, lock)

            breakdown.append(time.time())
            barrier.wait()
            breakdown.append(time.time())
            global_delta_val = self.global_delta.get_delta(global_deltas,lock)
            iter_count += 1

        breakdown.append(time.time())
        iteration_time = breakdown[-1] - init_time
        return breakdown

    def load_dataset(self):
        import random
        self.local_partition = np.random.randn(self.partition_points ,self.num_dimensions)
        #self.local_partition = np.random.randn(self.end_partition -self.start_partition ,self.num_dimensions)

    def distance(self, point, centroid):
        """Euclidean squared distance."""
        return np.linalg.norm(point - centroid)


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
    main(1000,100,4,2,10,0.00001, False, False)
