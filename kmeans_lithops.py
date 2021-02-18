import time
import random
import numpy as np
import lithops.multiprocessing as mplths
import multiprocessing as mp
import itertools
#from test

THRESHOLD = 0.00001
DATAPOINTS_PER_FILE = 20  # dataset 100 dimensions695_866
DIMENSIONS = 3

#Global objects
global_centroids = None
global_delta = None

# Shared memory objects 
lock = None
barrier = None

# Shared objects redis
worker_stats = None

def main():

    parallelism = 2
    clusters = 3
    dimensions = DIMENSIONS
    number_of_iterations = 3

    
    # Initialize shared memory objects
    global lock
    global barrier

    local = False
    if local == True:
        manager = mp.Manager()
        lock = manager.Lock()
        barrier = manager.Barrier(parallelism)
    else:
        manager = mplths.Manager()
        lock = manager.Lock()
        barrier = manager.Barrier(parallelism)
    
    global global_centroids
    global global_delta
    global worker_stats
    global_centroids = manager.dict({
                        'centroids':manager.dict(), 
                        'counters':manager.dict(), 
                        'centroids_temp':manager.dict(), 
                        'sizes_temp':manager.dict()})
    global_delta = manager.dict()
    worker_stats = manager.list()     # in seconds
    
    # TEST Crentroids
    if True == False:
        centroids = GlobalCentroids(2, 2)
        centroids.random_init(4)
        print(centroids.get_centroids())
        centroids.update(np.array([[1.2, 1, 1, 1], [2, 2, 2, 2]]), [2, 2])
        centroids.update(np.array([[2, 2, 2, 2.2], [1, 1, 1, 1]]), [2, 2])
        print(centroids.get_centroids())
        return
    # TEST Delta
    if True == False:
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
    
    iterator = list(itertools.product(range(parallelism), [parallelism], [DATAPOINTS_PER_FILE],[dimensions], [clusters],[number_of_iterations]))
    #iterator = (range(parallelism), [parallelism], [DATAPOINTS_PER_FILE],[dimensions], [clusters],[number_of_iterations])
    #for i in iterator:
    #    print(i)
    if local==True:
        pool = mp.Pool(processes=parallelism)
        pool.imap(run, iterator)
        #run(0, parallelism, DATAPOINTS_PER_FILE, dimensions, clusters, number_of_iterations)
        #[run(i, parallelism, DATAPOINTS_PER_FILE, dimensions, clusters, number_of_iterations) for i in range(parallelism)]
        pool.close()
        pool.join()
    else:
        
        pool = mplths.Pool(processes=parallelism)
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
    for k in global_centroids.items():
        print(k)
    with open('time_break.txt', 'w') as f:
        for item in worker_stats:
            f.write(f"{item}\n")

def run(w_id, parallelism, points_per_file,
            dimensions, clusters, number_of_iterations):
        global worker_stats
        global lock
        #lock.acquire()
        worker_breakdown = train(w_id, parallelism * points_per_file, dimensions, 
                                parallelism, clusters, number_of_iterations)
        worker_stats.append(worker_breakdown)
        #lock.release()

def train(worker_id, data_points, dimensions, 
            parallelism, clusters, max_iters):
    worker = Worker(worker_id, data_points, dimensions, parallelism,
                    clusters, max_iters)
    return worker.run()

class GlobalCentroids(object):
    def __init__(self, clusters, parallelism):
        self.num_clusters = clusters
        self.parallelism = parallelism

    @staticmethod
    def centroid_key(centroid):
        return "centroid" + str(centroid)

    def random_init(self, num_dimensions):
        global global_centroids
        global lock
        lock.acquire()
        for i in range(self.num_clusters):
            random.seed(1002+i)
            numbers = [random.gauss(0, 1) for _ in range(num_dimensions)]
            # Old redis calls
            global_centroids['centroids'][self.centroid_key(i)] = numbers
            # A counter for the updates
            global_centroids['counters'][self.centroid_key(i)] = 0
        lock.release()

    def update(self, coordinates, sizes):
        for k in range(self.num_clusters):
            self._update_centroid(k, coordinates[k].tolist(), int(sizes[k]))

    def _update_centroid(self, cluster_id, coordinates, size):
        # Current centroid
        centroid_k = self.centroid_key(cluster_id)
        global global_centroids
        global lock
        lock.acquire()
        n = len(global_centroids['centroids'][centroid_k])
        count = global_centroids['counters'][centroid_k]

        # First update
        if count == 0:
            #global_centroids['centroids_temp'][centroid_k].clear()
            centroids_temp = []
            for _ in range(n):
                centroids_temp.append(0.0)
            global_centroids['centroids_temp'][centroid_k] = centroids_temp
            global_centroids['sizes_temp'][centroid_k] = 0
        
        centroids_temp = global_centroids['centroids_temp'][centroid_k]
        for i in range(n):
            centroids_temp[i] =  centroids_temp[i] + coordinates[i]
        global_centroids['centroids_temp'][centroid_k] = centroids_temp

        global_centroids['sizes_temp'][centroid_k] += size
        size = global_centroids['sizes_temp'][centroid_k]
        global_centroids['counters'][centroid_k] += 1
        count = global_centroids['counters'][centroid_k]

        if count== self.parallelism :
            if size != 0:
                global_centroids['centroids'][centroid_k].clear()
                temps = global_centroids['centroids_temp'][centroid_k]
                values = []
                for i in range(0,n):
                    values.append(temps[i]/size)
                global_centroids['centroids'][centroid_k] = values
            global_centroids['counters'][centroid_k] = 0
        lock.release()

    def get_centroids(self):
        global global_centroids
        global lock
        lock.acquire()
        b = [global_centroids['centroids'][self.centroid_key(k)] for k in range(self.num_clusters)]
        lock.release()
        return np.array(list(map(lambda point: list(map(lambda v: float(v), point)), b)))

class GlobalDelta(object):
    def __init__(self, parallelism):
        self.parallelism = parallelism

    def init(self):
        # Hardcoded keys
        global global_delta
        global lock
        lock.acquire()
        global_delta["delta"] = 1
        global_delta["delta_c"] = 0
        global_delta["delta_temp"] = 0
        global_delta["delta_st"] = 0
        lock.release()

    def get_delta(self):
        global global_delta
        global lock
        lock.acquire()
        delta =global_delta["delta"]
        lock.release()
        return delta

    def update(self, delta, num_points): 
        # Old lua script
        #Local variables
        #delta_key = "delta"
        #counter_key = "delta_c"
        #delta_temp = "delta_temp"
        #npoints_temp = "delta_st"
        global global_delta
        global lock
        lock.acquire()
        global_delta["delta_temp"] += delta
        tmp_delta = global_delta["delta_temp"]
        global_delta["delta_st"] += num_points
        tmp_points = global_delta["delta_st"]

        global_delta["delta_c"] += 1
        count = global_delta["delta_c"]

        if count == self.parallelism:
            new_delta = tmp_delta / tmp_points
            global_delta["delta"] = new_delta
            global_delta["delta_c"] = 0
            global_delta["delta_temp"] = 0
            global_delta["delta_st"] = 0
        lock.release()

class Worker(object):

    def __init__(self, worker_id, data_points, dimensions, parallelism,clusters, max_iters):
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

        self.barrier = barrier
        self.global_delta = GlobalDelta(self.parallelism)
        self.global_centroids = GlobalCentroids(self.num_clusters,
                                                self.parallelism)

    def run(self):

        self.global_centroids = GlobalCentroids(self.num_clusters,self.parallelism)
        breakdown = []
        breakdown.append(time.time())

        self.load_dataset()
        #print(self.local_partition)

        self.local_membership = np.zeros([self.local_partition.shape[0]])

        # barrier before starting iterations, to avoid different execution times
        self.barrier.wait()

        init_time = time.time()
        breakdown.append(init_time)
        iter_count = 0
        global_delta_val = 1
        while (iter_count < self.max_iterations) and (global_delta_val > THRESHOLD):

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
            self.barrier.wait()
            breakdown.append(time.time())
            global_delta_val = self.global_delta.get_delta()
            iter_count += 1

        breakdown.append(time.time())
        return breakdown

    def load_dataset(self):
        import random
        self.local_partition = np.random.randn(self.partition_points,self.num_dimensions)

    def distance(self, point, centroid):
        """Euclidean squared distance."""
        return np.linalg.norm(point - centroid)

    def find_nearest_cluster(self, point):
        cluster = 0
        min_dis = None
        for k in range(self.num_clusters):
            distance = self.distance(self.local_partition[point-self.start_partition],
                                     self.correct_centroids[k])
            if min_dis is None or distance < min_dis:
                min_dis = distance
                cluster = k

        return cluster

    def compute_clusters(self):
        delta = 0
        for i in range(0, self.end_partition-self.start_partition):
            point = self.local_partition[i]
            dists = ((point - self.correct_centroids) ** 2).sum(axis=1)
            cluster = np.argmin(dists)

            self.local_centroids[cluster] += point

            self.local_sizes[cluster] += 1

            # If now point is a member of a different cluster
            if self.local_membership[i] != cluster:
                delta += 1
                self.local_membership[i] = cluster

        return delta

if __name__ == "__main__":
    main()
