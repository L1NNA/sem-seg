import os
from numbers import Real
from typing import Iterable, Union, Tuple, List, Optional, Any

import numpy as np
import numpy.typing as npt
import hnswlib
from tqdm import tqdm

import logging
logging.getLogger('similarity').setLevel(logging.ERROR)

os.environ['NUMEXPR_MAX_THREADS'] = '8'

DIM = 128

# TODO: https://github.com/nmslib/hnswlib/blob/master/TESTING_RECALL.md


class HNSWIndex:
    name = 'hnsw_index'

    def __init__(self,
                 file: str = 'hnsw_index.bin',
                 dim: int = DIM,
                 max_elements: int = 10_000,
                 space: str = 'l2',
                 ef: int = 100,
                 ef_construction: int = 200,
                 M: int = 16,
                 **kwargs):
        """
        @param file: The filename where the current index may be stored
        @param dim: The dimensionality of the data to be compared
        @param max_elements: The largest number of elements that will be able to be placed in the index
        @param space: The comparison to make between elements.  Can be any of ['cosine', 'l2', 'ip']  Check nmslib/hnsw_lib for more information
        @param ef: This parameter controls the accuracy:speed tradeoff.  Higher ef is better accuracy, with slower search.  ef should always be greater than k.  This is the depth of exploration during search
        The size of the dynamic list for the nearest neighbors (used during the search). Higher ef leads to more accurate but slower search. ef cannot be set lower than the number of queried nearest neighbors k. The value ef of can be anything between k and the size of the dataset.
        @param ef_construction: This is the depth of exploration during construction
        the parameter has the same meaning as ef, but controls the index_time/index_accuracy. Bigger ef_construction leads to longer construction, but better index quality. At some point, increasing ef_construction does not improve the quality of the index. One way to check if the selection of ef_construction was ok is to measure a recall for M nearest neighbor search when ef =ef_construction: if the recall is lower than 0.9, than there is room for improvement.
        @param M: maximum outgoing connections in the graph.  M \in [4,64]
        the number of bi-directional links created for every new element during construction. Reasonable range for M is 2-100. Higher M work better on datasets with high intrinsic dimensionality and/or high recall, while low M work better for datasets with low intrinsic dimensionality and/or low recalls. The parameter also determines the algorithm's memory consumption, which is roughly M * 8-10 bytes per stored element.
        As an example for dim=4 random vectors optimal M for search is somewhere around 6, while for high dimensional datasets (word embeddings, good face descriptors), higher M are required (e.g. M=48-64) for optimal performance at high recall. The range M=12-48 is ok for the most of the use cases. When M is changed one has to update the other parameters. Nonetheless, ef and ef_construction parameters can be roughly estimated by assuming that M*ef_{construction} is a constant.

        Generate a new instance of an hnswlib's HNSW index
        """
        # super(HNSWIndex, self).__init__(file, dim, **kwargs)
        p = hnswlib.Index(space=space, dim=dim)
        if os.path.isfile(file):
            try:
                p.load_index(file)
                self._index = p
                self._file_location = file
                self._dim = p.dim
                self._max_elements = p.max_elements
                self._space = p.space
                self._ef = p.ef
            except Exception as e:
                raise e
                # File exists, but is corrupt or incorrect format.
                pass  # TODO: catch errors and handle appropriately
        else:
            p.init_index(max_elements=max_elements,
                         ef_construction=ef_construction,
                         M=M,  # maximum number of outgoing connections in the graph
                         )
            self._index = p
            self._file_location = file
            self._dim = dim
            self._max_elements = max_elements
            self._space = space
            self._ef = ef
        self.update_ef(ef)

    def update_ef(self, ef: int) -> None:
        """
        @param ef: Having a higher ef value will provide better accuracy, but take longer to search.  ef should always be greater than k.


        """
        self._index.set_ef(ef)
        self._ef = ef

    def add_element(self, element: npt.ArrayLike, id: Optional[int] = None):
        """
        @param element: A single entry to add to the index
        @param id: The label that will be used for indexing.  If the used id is already present in the index, the value will be overwritten
        return: Nil

        This adds an element.  Unfortunately, it does not return a value for the label.
        If a label is not supplied, we write to the label `n` where `n` is the number of elements added to the index
        so far.  This could have the sideffect of overwriting labels that we defined.
        """
        index = self.add_elements(element, id)
        return index

    def add_elements(self,
                     elements: npt.ArrayLike,
                     ids: Optional[Union[npt.ArrayLike, int]] = None,
                     auto_expand: bool = True):
        """
        @param elements: A series of entries that will be added to the index.  Accepts shape of (n,d) where d is the vector dimension and n is the number of elements.
        @param ids: The labels that will be used for indexing.  If the used ids are already present in the index, the values will be overwritten
        @param auto_expand: If True, expand the maximum capacity of the index until the elements provided will fit.
        return: Nil

        This adds a series of elements.  Unfortunately, it does not return values for the labels.
        If a label is not supplied, we write to the label `n+1` where `n` is the number of elements added to the index
        so far.  This could have the sideffect of overwriting labels that we defined.
        If there isn't enough space in the index, expand the index until it will fit all elements.
        """
        try:
            indices = self._index.add_items(elements, ids)
        except RuntimeError as e:
            if auto_expand:
                new_items = 1 if len(elements.shape) == 1 else len(elements)
                while self._max_elements < self._index.element_count + new_items:
                    self.expand_capacity()
            try:
                indices = self._index.add_items(elements, ids)
            except RuntimeError as e:
                # TODO: Provide feedback that increasing the size did not help.
                raise RuntimeError(e)
            # TODO: Should we provide feedback that the list has been expanded?
        # self.append_vectors(ids, elements)
        self.backup_index()
        return indices

    def train(self, train_data: npt.ArrayLike) -> None:
        """
        @param train_data: A vector of shape (n,d) where d is the dimensions of the vectors, and n is the number of elements used for training.

        This function is not used for HNSW.
        """
        pass

    def find_nearest(self, query, k: int = 7) -> Tuple[List[int], List[float]]:
        """
        @param query: The vector which we're looking for.
        @param k: The number of neighbours to return
        @return: tuple[list[int], list[float]] containing
        """
        if k > self._index.element_count:
            k = self._index.element_count
        labels, distances = self._index.knn_query(query, k=k)
        if not isinstance(labels, list):
            labels = labels.tolist()
        return labels, distances

    def backup_index(self, file: Optional[str] = None) -> None:
        """
        @param file: The location to save the index.
        """
        file = file or self._file_location
        self._index.save_index(str(file))

    def expand_capacity(self, factor: Real = 1.5) -> None:
        """
        @param factor: A factor by which to increase the maximum number of elements.  Must be greater than 1.0
        """
        new_capacity = int(self._max_elements * factor)
        self._index.resize_index(new_capacity)
        self._max_elements = new_capacity

    def load_index(self, file: Optional[str] = None) -> None:
        """
        @param file: The name of the file that stores to index to recover from.
        """
        file = file or self._file_location
        new_index = hnswlib.Index(space=self._space, dim=self._dim)
        new_index.load_index(str(file))
        del self._index
        self._index = new_index
        self._file_location = file
        self._dim = self._index.dim
        self._max_elements = self._index.max_elements
        self._space = self._index.space
        self._ef = self._index.ef

    def retrieve_all_labels(self) -> List[int]:
        """
        return: A list of all the ids currently present in the index
        """
        return self._index.get_ids_list()



if __name__ == '__main__':
    h = HNSWIndex(space='l2', dim=128)
    data = []
    ids = []
    for i in range(100):
        data.append([i-0.2]*128)
        ids.append(i)
    if not os.path.exists(h._file_location):
        h.add_elements(np.array(data), ids)
    ids, scores = h.find_nearest([1]*128)
    print(ids)
    # for i in ids[0]:
    #     print(i, h.read_vector(i))
    # # print(h.vec_db.keys())
    # print(h.vec_db.get(0))
