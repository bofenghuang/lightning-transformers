#!/usr/bin/env python

import logging
from typing import List

import numpy as np
import scipy.stats
import torch
from torch.utils.data import Sampler

# ? or info
from pytorch_lightning.utilities.rank_zero import rank_zero_debug

# logger = logging.getLogger(__name__)

# logging.basicConfig(level=logging.INFO)


class DynamicBatchSampler(Sampler):
    def __init__(
        self,
        dataset,
        max_batch_length: int,
        num_buckets: int = None,
        length_func=lambda x: x["duration"],
        shuffle: bool = True,
        batch_ordering: str = "random",
        max_batch_ex: int = None,
        bucket_boundaries: List[int] = [],
        lengths_list: List[int] = None,
        seed: int = 42,
        epoch: int = 0,
        drop_last: bool = False,
        mode: str = "lognorm",
        fit_dataset: bool = False,
        verbose: bool = False,
    ):
        self._dataset = dataset
        self._ex_lengths = {}
        # ex_ids = self._dataset.data_ids
        self.verbose = verbose

        # We do not put a default on num_buckets to encourage users to play with this parameter
        if num_buckets is None and len(bucket_boundaries) == 0:
            raise RuntimeError(
                "Please specify either num_buckets or bucket boundaries." "Check the docs, and/or the tutorial !"
            )

        if lengths_list is not None:
            # take length of examples from this argument and bypass length_key
            # for indx in range(len(lengths_list)):
            #     self._ex_lengths[str(indx)] = lengths_list[indx]
            # todo : rm other
            self._ex_lengths = np.array(lengths_list)
        else:
            # todo
            # raise NotImplementedError()
            # use length func
            # if not isinstance(dataset, DynamicItemDataset):
            #     raise NotImplementedError("Dataset should be a Speechbrain DynamicItemDataset when using length function")
            # # for indx in range(len(self._dataset)):
            # #     self._ex_lengths[str(indx)] = length_func(self._dataset.data[ex_ids[indx]])
            # self._ex_lengths = [length_func(self._dataset.data[ex_ids[indx]]) for indx in range(len(self._dataset))]
            # self._ex_lengths = np.array(self._ex_lengths)

            lengths_list = [length_func(x) for x in self._dataset]
            self._ex_lengths = np.array(lengths_list)

        self._max_batch_length = max_batch_length
        self._shuffle_ex = shuffle
        self._batch_ordering = batch_ordering
        self._seed = seed
        self._drop_last = drop_last
        if max_batch_ex is None:
            max_batch_ex = np.inf
        self._max_batch_ex = max_batch_ex

        if len(bucket_boundaries) > 0:
            if not all([x >= 0 for x in bucket_boundaries]):
                raise ValueError("All elements in bucket boundaries should be non-negative (>= 0).")
            if not len(set(bucket_boundaries)) == len(bucket_boundaries):
                raise ValueError("Bucket_boundaries should not contain duplicates.")
            np.testing.assert_array_equal(
                np.array(bucket_boundaries),
                np.array(sorted(bucket_boundaries)),
                err_msg="The arg bucket_boundaries should be an ascending sorted list of non negative values values!",
            )
            self._bucket_boundaries = np.array(sorted(bucket_boundaries))

            self._ex_bucket_ids = self._get_bucket_ids()
        else:
            if mode == "kmeans":
                self._ex_bucket_ids = self._get_bucket_ids_by_kmeans(num_buckets=num_buckets)
                # get boudaries for debugging
                self._bucket_boundaries = np.array(self._get_boundaries_for_clusters(num_buckets=num_buckets))
            else:
                try:
                    rv = getattr(scipy.stats, mode)
                except AttributeError:
                    msg = f"Cannot import {mode} distribution from Scipy. Please use another random variable distribution like lognorm"
                    raise ImportError(msg)

                self._bucket_boundaries = np.array(
                    self._get_boundaries_through_warping(
                        rv, max_batch_length=max_batch_length, num_quantiles=num_buckets, fit_dataset=fit_dataset
                    )
                )

                self._ex_bucket_ids = self._get_bucket_ids()

        self._bucket_lens = self._get_bucket_lens(num_buckets, max_batch_length)

        self._epoch = epoch
        self._generate_batches()

    def get_durations(self, batch):
        # return [self._ex_lengths[str(idx)] for idx in batch]
        return self._ex_lengths[batch]

    def _get_bucket_ids(self):
        return np.searchsorted(self._bucket_boundaries, self._ex_lengths)

    def _get_bucket_ids_by_kmeans(self, num_buckets: int):
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            msg = "Please install sklearn to use kmeans\n"
            msg += "e.g. run: pip3 install -U scikit-learn"
            raise ImportError(msg)

        lengths = self._ex_lengths.reshape(-1, 1)
        km = KMeans(n_clusters=num_buckets, random_state=self._seed).fit(lengths)

        # sort cluster by centroid
        sorted_indices = np.argsort(km.cluster_centers_.reshape((-1,)))
        sorted_clusters = np.zeros_like(sorted_indices)
        sorted_clusters[sorted_indices] = np.arange(num_buckets)

        return sorted_clusters[km.labels_]

    def _get_boundaries_for_clusters(self, num_buckets: int, side: str = "left"):
        cluster_boundaries = []
        for bucket_id in range(num_buckets):
            len_by_cluster = self._ex_lengths[np.where(self._ex_bucket_ids == bucket_id)]
            cluster_boundaries.append([len_by_cluster.min(), len_by_cluster.max()])
        # print(cluster_boundaries)

        upper_boundaries = []
        for indx in range(num_buckets - 1):
            upper_boundaries.append(cluster_boundaries[indx][1] if side == "left" else cluster_boundaries[indx + 1][0])
        upper_boundaries.append(cluster_boundaries[-1][1])

        return upper_boundaries

    def _get_bucket_lens(self, num_buckets: int, max_batch_length: int):
        # todo: -inf
        # ? change to 0
        max_lens_by_bucket = [
            self._ex_lengths[np.where(self._ex_bucket_ids == bucket_id)].max(initial=-np.inf) for bucket_id in range(num_buckets)
        ]

        # Calculate bucket lengths - how often does one bucket boundary fit into max_batch_length?
        # self._bucket_lens = [
        #     max(1, int(max_batch_length / self._bucket_boundaries[i])) for i in range(len(self._bucket_boundaries))
        # ] + [1]

        # ? one less bucket than the other implementation
        bucket_lens = [max(1, int(max_batch_length / max_len)) for max_len in max_lens_by_bucket]

        return bucket_lens

    def _get_boundaries_through_warping(
        self,
        rv: scipy.stats.rv_continuous,
        max_batch_length: int,
        num_quantiles: int,
        fit_dataset: bool = False,
    ) -> List[int]:

        # NOTE: the following lines do not cover that there is only one example in the dataset
        # warp frames (duration) distribution of train data
        rank_zero_debug("Batch quantisation in latent space")

        if fit_dataset:
            # ? better use num_buckets
            latent_boundaries = np.linspace(
                1 / num_quantiles,
                1,
                num_quantiles,
            )
            # ? add floc=0 and fscale=1
            # RuntimeWarning: invalid value encountered in sqrt sk = 2*(b-a)*np.sqrt(a + b + 1) / (a + b + 2) / np.sqrt(a*b)
            rv_params = rv.fit(self._ex_lengths)
            # last upper boundary is always inf
            bucket_boundaries = rv.ppf(latent_boundaries, *rv_params)
            # replace inf by max length
            # bucket_boundaries[-1] = max(max(lengths), max(bucket_boundaries))

            # todo: add log

        else:
            # linspace set-up
            num_boundaries = num_quantiles + 1
            # create latent linearly equal spaced buckets
            latent_boundaries = np.linspace(
                1 / num_boundaries,
                num_quantiles / num_boundaries,
                num_quantiles,
            )
            # get quantiles using lognormal distribution
            # quantiles = lognorm.ppf(latent_boundaries, 1)
            quantiles = rv.ppf(latent_boundaries, 1)
            # scale up to to max_batch_length
            bucket_boundaries = quantiles * max_batch_length / quantiles[-1]

            # todo: add log

        # compute resulting bucket length multipliers
        length_multipliers = [bucket_boundaries[x + 1] / bucket_boundaries[x] for x in range(num_quantiles - 1)]
        # logging
        # todo: log format
        rank_zero_debug(
            "Latent bucket boundary - buckets: {} - length multipliers: {}".format(
                list(map("{:.2f}".format, bucket_boundaries)),
                list(map("{:.2f}".format, length_multipliers)),
            )
        )
        return list(sorted(bucket_boundaries))

    def _permute_batches(self):

        if self._batch_ordering == "random":
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(len(self._batches), generator=g).tolist()  # type: ignore
            tmp = []
            for idx in sampler:
                tmp.append(self._batches[idx])
            self._batches = tmp

        elif self._batch_ordering == "ascending":
            self._batches = sorted(
                self._batches,
                # key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
                key=lambda x: self._ex_lengths[x].max(),
            )
        elif self._batch_ordering == "descending":
            self._batches = sorted(
                self._batches,
                # key=lambda x: max([self._ex_lengths[str(idx)] for idx in x]),
                key=lambda x: self._ex_lengths[x].max(),
                reverse=True,
            )
        else:
            raise NotImplementedError

    def _generate_batches(self):
        rank_zero_debug("DynamicBatchSampler: Generating dynamic batches")
        if self._shuffle_ex:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self._seed + self._epoch)
            sampler = torch.randperm(len(self._dataset), generator=g).tolist()  # type: ignore
        else:
            # take examples as they are: e.g. they have been sorted
            sampler = range(len(self._dataset))  # type: ignore

        self._batches = []
        bucket_batches = [[] for i in self._bucket_lens]

        stats_tracker = [
            {"min": np.inf, "max": -np.inf, "tot": 0, "n_ex": 0, "item_lengths": [], "item_lengths_by_batch": []}
            for i in self._bucket_lens
        ]

        for idx in sampler:
            # length of pre-sampled audio
            # item_len = self._ex_lengths[str(idx)]
            item_len = self._ex_lengths[idx]

            bucket_id = self._ex_bucket_ids[idx]

            # fill audio's duration into that bucket
            bucket_batches[bucket_id].append(idx)

            # stats_tracker[bucket_id]["min"] = min(stats_tracker[bucket_id]["min"], item_len)
            # stats_tracker[bucket_id]["max"] = max(stats_tracker[bucket_id]["max"], item_len)
            # stats_tracker[bucket_id]["tot"] += item_len
            # stats_tracker[bucket_id]["n_ex"] += 1
            stats_tracker[bucket_id]["item_lengths"].append(item_len)
            # track #samples - why not duration/#frames; rounded up?
            # keep track of durations, if necessary

            # todo: change to compare 
            if (
                len(bucket_batches[bucket_id]) >= self._bucket_lens[bucket_id]
                or len(bucket_batches[bucket_id]) >= self._max_batch_ex
            ):
                self._batches.append(bucket_batches[bucket_id])
                bucket_batches[bucket_id] = []
                # keep track of durations

                stats_tracker[bucket_id]["item_lengths_by_batch"].append(stats_tracker[bucket_id]["item_lengths"])
                stats_tracker[bucket_id]["item_lengths"] = []

        # Dump remaining batches
        if not self._drop_last:
            for bucket_id, batch in enumerate(bucket_batches):
                if batch:
                    self._batches.append(batch)
                    stats_tracker[bucket_id]["item_lengths_by_batch"].append(stats_tracker[bucket_id].pop("item_lengths"))
                    # todo: save mem
                    stats_tracker[bucket_id]["item_lengths"] = []

        self._permute_batches()  # possibly reorder batches

        if self._epoch == 0:  # only log at first epoch
            # frames per batch & their padding remaining
            boundaries = [0] + self._bucket_boundaries.tolist()

            n_true_samples = 0
            n_all_samples = 0
            n_tot_batches = 0
            for bucket_indx in range(len(self._bucket_boundaries)):
                # shape: n_batchs * n_examples_per_batch
                # item_lengths_by_batch = np.array(stats_tracker[bucket_indx]["item_lengths_by_batch"])
                # num_batches = item_lengths_by_batch.shape[0]
                # max_len_by_batch = item_lengths_by_batch.max(axis=1)

                item_lengths_by_batch = stats_tracker[bucket_indx]["item_lengths_by_batch"]

                n_batches = len(item_lengths_by_batch)
                n_tot_batches += n_batches

                n_items_by_batch = [len(item_len) for item_len in item_lengths_by_batch]
                n_items = sum(n_items_by_batch)

                max_lengths_by_batch = [max(item_len) for item_len in item_lengths_by_batch]

                n_true_samples_by_bucket = sum(y for x in item_lengths_by_batch for y in x)
                n_all_samples_by_bucket = sum(n * m for n, m in zip(n_items_by_batch, max_lengths_by_batch))
                n_true_samples += n_true_samples_by_bucket
                n_all_samples += n_all_samples_by_bucket

                try:
                    pct_padding = 1 - n_true_samples_by_bucket / n_all_samples_by_bucket
                except ZeroDivisionError:
                    pct_padding = 0

                # try:
                #     n_batches = stats_tracker[bucket_indx]["tot"] // (self._max_batch_length)
                #     pad_factor = (stats_tracker[bucket_indx]["max"] - stats_tracker[bucket_indx]["min"]) / (
                #         stats_tracker[bucket_indx]["tot"] / stats_tracker[bucket_indx]["n_ex"]
                #     )
                # except ZeroDivisionError:
                #     n_batches = 0
                #     pad_factor = 0

                # todo： why full batch
                rank_zero_debug(
                    (
                        "DynamicBatchSampler: Bucket {} with boundary {:.1f}-{:.1f} and "
                        + "batch_size {}: Num Examples {}, Num Batches {}, % of padding {:.2f}%."
                    ).format(
                        bucket_indx,
                        boundaries[bucket_indx],
                        boundaries[bucket_indx + 1],
                        self._bucket_lens[bucket_indx],
                        # stats_tracker[bucket_indx]["n_ex"],
                        n_items,
                        n_batches,
                        # pad_factor * 100,
                        pct_padding * 100,
                    )
                )

            pct_true = n_true_samples / n_all_samples * 100
            rank_zero_debug("DynamicBatchSampler: % true samples {:.2f}%, % of padding {:.2f}%, #batches {}".format(pct_true, 100 - pct_true, n_tot_batches))

        # if self.verbose:
        #     batch_stats = {
        #         "tot_frames": [],
        #         "tot_pad_frames": [],
        #         "pad_%": [],
        #     }
        #     for batch in self._batches:
        #         tot_frames = sum([self._ex_lengths[str(idx)] for idx in batch])
        #         batch_stats["tot_frames"].append(tot_frames)
        #         max_frames = max([self._ex_lengths[str(idx)] for idx in batch])
        #         tot_pad = sum([max_frames - self._ex_lengths[str(idx)] for idx in batch])
        #         batch_stats["tot_pad_frames"].append(tot_pad)
        #         batch_stats["pad_%"].append(tot_pad / tot_frames * 100)

        #     padding_details = "Batch {} with {:.1f} frames with {} files - {:.1f} padding, {:.2f} (%) of total."
        #     padding_details = "DynamicBatchSampler: " + padding_details
        #     for i in range(len(self._batches)):
        #         rank_zero_debug(
        #             padding_details.format(
        #                 i,
        #                 batch_stats["tot_frames"][i],
        #                 len(self._batches[i]),
        #                 batch_stats["tot_pad_frames"][i],
        #                 batch_stats["pad_%"][i],
        #             )
        #         )

    def __iter__(self):
        for batch in self._batches:
            yield batch
        if self._shuffle_ex:  # re-generate examples if ex_ordering == "random"
            self._generate_batches()
        if self._batch_ordering == "random":
            # we randomly permute the batches only --> faster
            self._permute_batches()
        else:
            pass

    def set_epoch(self, epoch):
        """
        You can also just access self.epoch, but we maintain this interface
        to mirror torch.utils.data.distributed.DistributedSampler
        """
        self._epoch = epoch
        self._generate_batches()

    def __len__(self):
        return len(self._batches)
