'''
                         --variable shorthand--
Notation:
    ch: Convex Hull
    us: Unit Sphere
    pc: Point Cloud
    s3: S3 Group
    t: Translation Group
'''

from abc import ABCMeta

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from torch_canon.Hopcroft import PartitionRefinement
from torch_canon.utilities import build_adjacency_list, check_type, direct_graph

from .align import align_pc_t, align_pc_s3
from .dfa import construct_dfa, convert_partition, traversal, construct_symmetries
from .canon import enc_us_catpc, enc_ch_pc
from .complete import check_colinear
from .qhull import get_ch_graph

from abc import ABCMeta

import numpy as np

class CanonEn(metaclass=ABCMeta):
    def __init__(self,
                 n : int = 3,
                 tol : float = 1e-4,
                 save : str = '',
     ) -> None:
        super().__init__()
        self.n = n
        self.tol = tol
        self.save = save

        self.dist_hash = None
        self.g_hash = None
        self.dist_encoding = None
        self.g_encoding = None
        self.n_encoding = None

    def _save(self, pre_data, data, frame_R, frame_t, sorted_graph, aligned_graph, dist_hash, g_hash, dist_encoding, g_encoding, n_encoding, sorted_path, aligned_path, local_list, local_mask):
        self.data = data
        self.pre_data = pre_data
        self.frame_R = frame_R
        self.frame_t = frame_t
        self.sorted_graph = sorted_graph

        if self.save in ['dist', 'all']:
            self.dist_hash = dist_hash
            self.dist_encoding = [dist_encoding[i] for i in sorted_path]
        if self.save in ['geom', 'all']:
            self.g_hash = g_hash
            self.g_encoding = [g_encoding[i] for i in sorted_path]
        if self.save in ['node', 'all']:
            self.n_encoding = [n_encoding[i] for i in sorted_path]
        symmetric_elements = self.get_symmetric_elements(aligned_graph, local_list, local_mask)
        flat_symmetric_elements = [item for sublist in symmetric_elements for item in sublist]
        assert len(flat_symmetric_elements) == data.shape[0], 'Symmetric elements do not match data size'
        self.symmetric_elements = symmetric_elements.copy()
        self.simple_asu = self.get_simple_asu(symmetric_elements,  data)
        self.symmetries = construct_symmetries(pre_data, self.symmetric_elements.copy(), self.tol)
        pass

    def get_simple_asu(self, symmetric_elements, data, asu=None):
        fresh_asu = False if asu is not None else True
        repeat_elements = set()
        while len(symmetric_elements) > 0 and () not in symmetric_elements:
            elements = list(symmetric_elements.pop())
            if not fresh_asu and len(asu) == np.linalg.matrix_rank(data):
                return asu
            elif asu is None:
                asu = [elements[0]]
                elements = np.delete(elements, 0, axis=0)
                if len(elements)>0:
                    repeat_elements.add(tuple(elements))
                continue
            dists = np.linalg.norm(data[elements][:, np.newaxis, :] - data[asu][np.newaxis, :, :],axis=2).mean(axis=1)
            argmin = np.argmin(dists)
            asu.append(elements[argmin])
            elements = tuple(np.delete(elements, argmin, axis=0))
            if len(elements)>0:
                repeat_elements.add(tuple(elements))
        if len(asu)<=np.linalg.matrix_rank(data):
            return self.get_simple_asu(repeat_elements, data, asu=asu)
        return asu

    def get_symmetric_elements(self, aligned_graph, local_list, local_mask):
        symmetric_elements = set(tuple(source) for source, _ in aligned_graph)
        symmetric_elements.update(set(tuple(target) for _, target in aligned_graph))
        # bump all values by 1 after local_mask false occurs
        if len(local_list) > 0:
            for index_idx, index_bool in enumerate(local_mask):
                if not index_bool:
                    new_symmetric_elements = set()
                    for idx, sublist in enumerate(symmetric_elements):
                        new_symmetric_elements.add(tuple([item + 1 if item >= index_idx else item for item in sublist]))

                    symmetric_elements = new_symmetric_elements

            mapping_dict = {}
            for idx, sublist in enumerate(symmetric_elements):
                for item in sublist:
                    mapping_dict[item] = idx
            result = [[] for _ in symmetric_elements]
            for sublist in local_list:
                base_element = sublist[0]  # Identify which sublist this belongs to
                mapped_index = mapping_dict.get(base_element)

                if mapped_index is not None:
                    result[mapped_index].extend(sublist[1:])  # Add elements to the correct result sublist

            for result_sublist in result:
                if len(result_sublist) > 0:
                    symmetric_elements.add(tuple(result_sublist))
        return symmetric_elements



    def get_frame(self, data, cat_data, *args, **kwargs):
        data = check_type(data) # Assert Type

        # TRANSLATION GROUP
        # -----------------
        data, frame_t = align_pc_t(data) # Translation group alignment
        cntr_data = data.clone() # TODO: Don't copy just use indexing
        dists = torch.linalg.norm(data, axis=1)
        angle_tol = max(np.arcsin(self.tol/(dists.min().item()+self.tol)), np.sqrt(self.tol))
        zero_mask = dists > self.tol
        data, cat_data = data[zero_mask], cat_data[zero_mask]
        dists = torch.linalg.norm(data, axis=1)

        # ROTATION GROUP
        # --------------
        # Unit Sphere Encoding
        dist_hash, dist_encoding, us_data, local_list, local_mask = enc_us_catpc(
                data, cat_data,
                dist_hash=self.dist_hash, dist_encoding=None,
                tol=self.tol, angle_tol=angle_tol)

        # Build Convex Hull Graph
        us_rank = torch.linalg.matrix_rank(us_data, tol=self.tol)
        us_n = us_data.shape[0]
        ch_graph = get_ch_graph(us_data, us_rank, us_n)

        bool_lst = [i in ch_graph for i in range(us_n)]
        assert all(bool_lst), 'Convex Hull is not correct'

        # Encode Convex Hull Geometry
        us_adj_dict = build_adjacency_list(ch_graph)
        dg = direct_graph(ch_graph)
        g_hash, g_encoding = enc_ch_pc(
                us_data, us_adj_dict, us_rank,
                g_hash=self.g_hash, g_encoding=None,
                tol=self.tol, angle_tol=angle_tol)

        # COMBINE ENCODINGS
        n_encoding = {}
        # for each node combine ENCODINGS
        for i in range(us_n):
            n_encoding[i] = (dist_encoding[i], g_encoding[i])

        # CONSTRUCT DFA
        dfa = construct_dfa(n_encoding, dg)
        self.hopcroft = PartitionRefinement(dfa)
        self.hopcroft.refine(dfa)
        sorted_graph, aligned_graph = convert_partition(self.hopcroft, dist_hash, g_hash, dist_encoding, g_encoding, zero_mask)
        sorted_path, aligned_path = traversal(sorted_graph, us_adj_dict, us_data, us_rank, zero_mask)
        lindep_pth = self.traverse(sorted_graph, us_adj_dict, us_data, us_rank)
        pre_data = data.clone()
        data, frame_R = align_pc_s3(cntr_data, us_data, lindep_pth)

        if self.save is False:
            return data, frame_R, frame_t
        else:
            self._save(pre_data, data, frame_R, frame_t, sorted_graph, aligned_graph, dist_hash, g_hash, dist_encoding, g_encoding, n_encoding, sorted_path, aligned_path, local_list, local_mask)
        return data, frame_R, frame_t

    def traverse(self, sorted_graph, us_adj_dict, us_data, us_rank):

        # ~~~
        # Start on the first set of symmetric edges and with the first node
        vert0_idx, dfa_node_idx = self.find_vert0(us_adj_dict, sorted_graph)
        if us_rank == 1:
            return [vert0_idx]
        vert0_vec = us_data[vert0_idx]

        # ~~~
        # Find the second node
        vert1_idx, dfa_node_idx = self.find_vert1(vert0_idx, vert0_vec, dfa_node_idx, us_adj_dict, sorted_graph, us_data)
        if us_rank == 2:
            return [vert0_idx, vert1_idx]
        vert1_vec = us_data[vert1_idx]

        v2 = self.v2_subroutine(vert0_idx, vert1_idx, dfa_node_idx, sorted_graph, us_adj_dict, us_data, us_rank)
        if v2 is None:
            v2 = self.v2_subroutine(vert1_idx, vert0_idx, dfa_node_idx, sorted_graph, us_adj_dict, us_data, us_rank)

        assert v2 is not None, f'v2 is None\n {vert0_idx},{vert1_idx}\n \n {sorted_graph}'

        return [vert0_idx, vert1_idx, v2]

    def find_vert0(self, us_adj_dict, sorted_graph):
        symmetry_group = 0
        left_edge_vertices = 0
        first_vertex = 0
        vert0_idx = sorted_graph[symmetry_group][left_edge_vertices][first_vertex]
        return vert0_idx, symmetry_group

    def find_vert1(self, vert0_idx, vert0_vec, dfa_node_idx, us_adj_dict, sorted_graph, us_data):
        return us_adj_dict[vert0_idx][0], None

    def v2_subroutine(self, vert0_idx, vert1_idx, dfa_node_idx, sorted_graph, us_adj_dict, us_data, us_rank):
        vert2_idx = None
        vert0_vec = us_data[vert0_idx]
        vert1_vec = us_data[vert1_idx]

        for idx in us_adj_dict[vert1_idx]:
          if idx != vert0_idx:
            return idx


        # Reset the dfa_node_idx v1 was received from a right hand node
        if dfa_node_idx is None:
          dfa_node_idx = 0
          for i in range(len(sorted_graph)):
            if vert1_idx in sorted_graph[i][0]:
              dfa_node_idx = i
              break

        # # Check DFA among right nodes (order 0)
        # ~~~
        while vert2_idx is None and dfa_node_idx < len(sorted_graph):

            test_vert_idxs = sorted_graph[dfa_node_idx][1] # Get connected edges in symmetry edge
            test_vert_idxs = [i for i in test_vert_idxs if (i != vert0_idx) and (i!=vert1_idx)] # Ignore if it's v0 (may happen due to symmetry BD)
            us_adj_idxs = us_adj_dict[vert1_idx]
            test_vert_idxs = [i for i in test_vert_idxs if i in us_adj_idxs] # Ignore if it's not in the ch_graph

            # testing vertices
            for idx in test_vert_idxs:
                test_vert_vec = us_data[idx]

                # ignore if co-linear
                colinear0_bool = check_colinear(vert0_vec, test_vert_vec, self.tol)
                colinear1_bool = check_colinear(vert1_vec, test_vert_vec, self.tol)
                if (not colinear0_bool) and (not colinear1_bool):
                    vert2_idx = idx
                    return vert2_idx

            # iterate dfa
            if vert2_idx is None:
              dfa_node_idx += 1
              while (dfa_node_idx < len(sorted_graph)) and (not vert1_idx in sorted_graph[dfa_node_idx][0]):
                dfa_node_idx += 1

        return vert2_idx
