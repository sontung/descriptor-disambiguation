import json
import os
from collections import defaultdict

import h5py
import numpy as np


class Sam3ObservationModel:
    uses_observation_descriptors = True

    def __init__(
        self,
        info_dir="/home/minhnxh/Documents/VinRobotic/descriptor-disambiguation/output/sam3_information",
        min_patch_count=4,
    ):
        self.info_dir = info_dir
        self.min_patch_count = min_patch_count
        self.conf = {"name": "sam3salad"}

        self.db_obs_path = os.path.join(info_dir, "observations.json")
        self.query_obs_path = os.path.join(info_dir, "query_observations.json")
        self.db_desc_path = os.path.join(info_dir, "salad_global_database.h5")
        self.query_desc_path = os.path.join(info_dir, "query_descriptors.h5")
        self.db_mask_path = os.path.join(info_dir, "sam3_results_db.h5")

        with open(self.db_obs_path, "r") as handle:
            self.db_observations = json.load(handle)
        with open(self.query_obs_path, "r") as handle:
            self.query_observations = json.load(handle)

        self.db_h5 = h5py.File(self.db_desc_path, "r")
        self.query_h5 = h5py.File(self.query_desc_path, "r")
        self.db_mask_h5 = h5py.File(self.db_mask_path, "r")
        self.db_descriptors = self.db_h5["global_descriptors"]
        self.query_descriptors = self.query_h5["query_descriptors"]
        self.query_fallback_descriptor = self._compute_fallback_descriptor(
            self.query_descriptors
        )
        self.missing_query_names = set()

        self.db_image_to_obs = defaultdict(list)
        self.query_image_to_obs = defaultdict(list)

        self._index_db_observations()
        self._index_query_observations()

    def _normalize_name(self, name):
        name = str(name)
        base = os.path.basename(name)
        stem = os.path.splitext(base)[0]
        rel = name.replace("\\", "/")
        return {name, rel, base, stem}

    def _weight_from_observation(self, obs):
        chosen = obs.get("chosen_patches", [])
        patch_count = int(sum(chosen))
        if patch_count < self.min_patch_count:
            return 0.0
        weight = float(patch_count)
        return weight if weight > 0 else 1.0

    def _db_score_for_observation(self, obs):
        image_id = obs.get("image_id")
        mask_index = obs.get("mask_index")
        if image_id is None or mask_index is None:
            return 1.0
        try:
            scores = self.db_mask_h5[str(image_id)]["architecture"]["scores"]
            if 0 <= int(mask_index) < len(scores):
                return float(scores[int(mask_index)])
        except KeyError:
            pass
        return 1.0

    def _compute_fallback_descriptor(self, descriptors):
        mean_desc = np.asarray(descriptors[:]).mean(axis=0).astype(np.float32)
        norm = float(np.linalg.norm(mean_desc))
        if norm > 0:
            mean_desc /= norm
        return mean_desc

    def _index_db_observations(self):
        for obs_idx, obs in enumerate(self.db_observations):
            weight = self._weight_from_observation(obs) * self._db_score_for_observation(
                obs
            )
            image_id = obs.get("image_id")
            stem = str(obs.get("stem"))
            if image_id is not None:
                self.db_image_to_obs[str(image_id)].append((obs_idx, weight))
            self.db_image_to_obs[stem].append((obs_idx, weight))

    def _index_query_observations(self):
        for obs_idx, obs in enumerate(self.query_observations):
            weight = self._weight_from_observation(obs)
            for key in self._normalize_name(obs.get("image_name", "")):
                self.query_image_to_obs[key].append((obs_idx, weight))
            stem = str(obs.get("stem"))
            self.query_image_to_obs[stem].append((obs_idx, weight))

    def _aggregate(self, entries, descriptors):
        if not entries:
            raise KeyError("No SAM3 observations found for image")

        indices = np.array([idx for idx, _ in entries], dtype=np.int64)
        weights = np.array([weight for _, weight in entries], dtype=np.float32)
        positive = weights > 0
        if np.any(positive):
            indices = indices[positive]
            weights = weights[positive]
        if indices.size == 0:
            raise KeyError("No SAM3 observations remaining after filtering")
        order = np.argsort(indices)
        indices = indices[order]
        weights = weights[order]

        unique_indices, inverse = np.unique(indices, return_inverse=True)
        if unique_indices.shape[0] != indices.shape[0]:
            merged_weights = np.zeros(unique_indices.shape[0], dtype=np.float32)
            np.add.at(merged_weights, inverse, weights)
            indices = unique_indices
            weights = merged_weights

        selected = descriptors[indices]

        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            aggregated = selected.mean(axis=0, dtype=np.float32)
        else:
            aggregated = np.average(selected, axis=0, weights=weights).astype(np.float32)
        norm = float(np.linalg.norm(aggregated))
        if norm > 0:
            aggregated /= norm
        return aggregated

    def process(self, name, image_id=None):
        image_keys = self._normalize_name(name)

        if image_id is not None:
            image_id = str(image_id)
            if image_id in self.db_image_to_obs:
                return self._aggregate(
                    self.db_image_to_obs[image_id], self.db_descriptors
                )
            image_keys.add(image_id)

        for key in image_keys:
            if key in self.query_image_to_obs:
                return self._aggregate(
                    self.query_image_to_obs[key], self.query_descriptors
                )

        for key in image_keys:
            if key in self.db_image_to_obs:
                return self._aggregate(self.db_image_to_obs[key], self.db_descriptors)

        key = str(name)
        if key not in self.missing_query_names:
            print(
                f"Warning: no SAM3 observation descriptors for {name} ({image_id}); "
                "using mean query descriptor fallback."
            )
            self.missing_query_names.add(key)
        return self.query_fallback_descriptor.copy()

    # def process_per_keypoint(self, name, keypoints, image_shape):
    #     import dd_utils
    #
    #     # 1. Resolve image identity
    #     image_keys = self._normalize_name(name)
    #     image_id = None
    #     for k in image_keys:
    #         if k in self.db_image_to_obs or k in self.query_image_to_obs:
    #             image_id = k
    #             break
    #
    #     if not image_id:
    #         # Fallback if image isn't found
    #         return np.tile(self.query_fallback_descriptor, (keypoints.shape[0], 1))
    #
    #     # 2. Fetch the raw masks and descriptors for this image
    #     # Note: You may need to adjust the H5 keys here depending on where
    #     # your test-set masks are actually saved (query vs db).
    #     try:
    #         all_masks = self.db_mask_h5[str(image_id)]["architecture"]["masks"][:]
    #
    #         # Reconstruct dino_patches from masks
    #         # Assuming you moved `find_dino_patch_coords` to dd_utils as well
    #         dino_patches = [dd_utils.find_dino_patch_coords(m)[1] for m in all_masks]
    #
    #         # Fetch the raw unaggregated descriptors for this image
    #         # (Requires you saved the bd_descriptors to the h5, or you fetch from self.db_descriptors)
    #         entries = self.db_image_to_obs.get(image_id, [])
    #         indices = [idx for idx, _ in entries]
    #         raw_descriptors = self.db_descriptors[indices]
    #
    #     except KeyError:
    #         # Fallback if masks aren't generated for this specific image
    #         return np.tile(self.query_fallback_descriptor, (keypoints.shape[0], 1))
    #
    #     # 3. Perform spatial routing
    #     assignments = dd_utils.mask_assignment(keypoints, image_shape, dino_patches)
    #
    #     # 4. Build the dense (K, 8192) descriptor array
    #     dense_descriptors = []
    #     fallback = self.query_fallback_descriptor
    #
    #     for mask_idx in assignments:
    #         if mask_idx != -1 and mask_idx < len(raw_descriptors):
    #             dense_descriptors.append(raw_descriptors[mask_idx])
    #         else:
    #             dense_descriptors.append(fallback)  # Background/sky gets fallback
    #
    #     return np.array(dense_descriptors, dtype=np.float32)
