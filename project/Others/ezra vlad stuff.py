
# Vlad with power norm, l2 norm, PCA whitening
from typing import Any, Callable
from dataclasses import dataclass, field
from sklearn.cluster import KMeans

from sklearn.decomposition import PCA


def get_sift_function():
    sift = cv.SIFT_create()
    return lambda image: sift.detectAndCompute(image, None)[1]

from dataclasses import dataclass

@dataclass
class Vlad:
    rho: float
    descriptor_size: int
    get_descriptors: Callable[[Any], Any] 
    vlad_clusters: int = 6
    output_size: int = 100
    
    _clust_model: KMeans = field(init=False)
    _pca_model: PCA = field(init=False)

    def fit_transform(self, images: list[Any]):
        descriptors_list = [self.get_descriptors(i) for i in tqdm(images, desc="Vlad.fit: Extracting SIFT descriptors", total=images.shape[0])]
        all_descriptors = np.vstack([i for i in descriptors_list if i is not None]) 
        self._clust_model = KMeans(n_clusters=self.vlad_clusters, random_state=0) # centers are mu_k
        self._clust_model.fit(all_descriptors)
        self._pca_model = PCA(whiten=True, tol=0.1)
        all_vlad_vectors = []

        for descriptors in tqdm(descriptors_list, desc='Vlad.fit: Sampling all vlad_vectors'):
            all_vlad_vectors.append(self._vlad(descriptors))
        
        results = self._pca_model.fit_transform(np.array(all_vlad_vectors))
        self.output_size = min(self.output_size, self._pca_model.n_components_)
        return results[:, :self.output_size]
    
    def fit(self, images):
        self.fit_transform(images)

    def _vlad(self, descriptors):
        # 1. Each image has n descriptors of fixed size m (=128)
        # 2. Find their cluster center that each of n descriptors belongs to
        # 3. For each descriptor subtract its assigned cluster center and add it to the vlad vector for that cluster
        # vlad_vector: has k rows and m (=128) columns each row is the local aggregation of descriptors from the image

        k = self.vlad_clusters
        vlad_vector = np.zeros((k, self.descriptor_size)) # (k, 128) vector
        if descriptors is None:
            return vlad_vector.flatten()
        cluster_assignments = self._clust_model.predict(descriptors)
        vlad_vector = np.zeros((k, descriptors.shape[1]))  # (k, 128) vector

        for idx, cluster_idx in enumerate(cluster_assignments):
            vlad_vector[cluster_idx] += (descriptors[idx] - self._clust_model.cluster_centers_[cluster_idx])

        # Power normalization
        vlad_vector = (np.sign(vlad_vector) * np.abs(vlad_vector) ** self.rho).flatten()
        
        # l2 Normalization
        return vlad_vector/np.linalg.norm(vlad_vector)  # Normalize each row independently
 
    def _pca(self, vlad_vector):
        return self._pca_model.transform(vlad_vector)[:, :self.output_size]

    def transform(self, images):
        if len(images.shape) == 2:
            images = np.array([images])
            
        vecs = np.zeros((images.shape[0], self.descriptor_size*self.vlad_clusters))
        for i in tqdm(range(images.shape[0]), desc='Vlad.transform'):
            desc = self.get_descriptors(images[i])
            vecs[i] = self._vlad(desc)
        
        return self._pca(vecs)
    
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Now we use the histograms as inputs to the SVM classifier
# (nsamples, nfeatures)

X = images
y = probas

thresholds = [0, 0.33, 0.66, 1.0]
y_classes = [sum(1 for t in thresholds if value >= t) for value in y]

X_train, X_test, y_cls_train, y_cls_test, y_train, y_test = train_test_split(X, y_classes, probas, test_size=0.25, random_state=42)X_train.shape

vlad = Vlad(0.5, 128, get_sift_function(), vlad_clusters=16, output_size=1000)


X_train_vlad = vlad.fit_transform(X_train)
X_test_vlad = vlad.transform(X_test)


# Initialize and train the SVM classifier
clf = LinearSVC(dual=False, max_iter=100000)
clf.fit(X_train_vlad[:,:50], y_cls_train)

# Predict on the test set
y_pred = clf.predict(X_test_vlad[:,:50])

# Print the classification report

print(classification_report(y_cls_test, y_pred))