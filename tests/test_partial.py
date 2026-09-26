"""Partially aggregated partition observations and their validation."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from reference import partition_evidence_by_enumeration  # noqa: E402

from himeb import (  # noqa: E402
    PartialObservation,
    PartitionBlock,
    fit_alpha_partial,
    log_evidence,
    log_evidence_partial,
    partial_evidence_breakdown,
    partial_observation,
)


class PartialEvidenceReferenceTests(unittest.TestCase):
    ALPHAS = (1e-4, 1e-2, 0.7, 1.0, 2.5, 100.0, 1e4)

    def test_full_refinement_matches_complete_count_evidence(self):
        rows = [(4, 0, 1), (2, 2, 0), (0, 0, 0), (1, 3, 2), (6, 0, 0)]
        observations = [
            partial_observation([((class_index,), count) for class_index, count in enumerate(row)])
            for row in rows
        ]
        for alpha in self.ALPHAS:
            with self.subTest(alpha=alpha):
                self.assertEqual(log_evidence_partial(alpha, observations), log_evidence(alpha, rows))

    def test_merged_evidence_equals_fine_enumeration(self):
        partitions = [
            [((0, 1), 2), ((2,), 1)],
            [((0,), 0), ((1, 2), 4)],
            [((0, 2), 3), ((1,), 2)],
            [((0,), 0), ((1,), 0), ((2,), 5)],
        ]
        for raw_blocks in partitions:
            observation = partial_observation(raw_blocks)
            for alpha in self.ALPHAS:
                with self.subTest(blocks=raw_blocks, alpha=alpha):
                    got = log_evidence_partial(alpha, [observation])
                    ref = partition_evidence_by_enumeration(raw_blocks, alpha)
                    self.assertLessEqual(abs(got - ref), 2e-12)

    def test_mixed_partitions_batch_vs_independent_enumeration(self):
        raw_observations = [
            [((0, 1), 4), ((2,), 1)],
            [((0,), 1), ((1, 2), 3)],
            [((0, 1, 2), 6)],
            [((0, 2), 1), ((1,), 0)],
        ]
        observations = [partial_observation(blocks) for blocks in raw_observations]
        for alpha in (0.001, 0.4, 2.0, 70.0, 1e4):
            with self.subTest(alpha=alpha):
                got = log_evidence_partial(alpha, observations)
                ref = math.fsum(partition_evidence_by_enumeration(blocks, alpha) for blocks in raw_observations)
                self.assertLessEqual(abs(got - ref), 2e-12)

    def test_all_partitions_and_small_aggregate_counts_by_enumeration(self):
        partitions = [
            [((0, 1, 2),)],
            [((0, 1),), ((2,),)],
            [((0, 2),), ((1,),)],
            [((1, 2),), ((0,),)],
            [((0,),), ((1,),), ((2,),)],
        ]

        def compositions(total: int, slots: int, prefix=()):
            if slots == 1:
                yield prefix + (total,)
                return
            for value in range(total + 1):
                yield from compositions(total - value, slots - 1, prefix + (value,))

        for raw_partition in partitions:
            for total in range(6):
                for aggregate_counts in compositions(total, len(raw_partition)):
                    raw_blocks = [
                        (classes, count)
                        for (classes,), count in zip(raw_partition, aggregate_counts)
                    ]
                    observation = partial_observation(raw_blocks)
                    for alpha in (1e-3, 1.0, 1e3):
                        with self.subTest(blocks=raw_blocks, alpha=alpha):
                            got = log_evidence_partial(alpha, [observation])
                            ref = partition_evidence_by_enumeration(raw_blocks, alpha)
                            self.assertLessEqual(abs(got - ref), 2e-12)

    def test_fully_merged_observation_is_identically_zero(self):
        observations = [partial_observation([((0, 1, 2), 5)])]
        for alpha in (1e-4, 0.3, 1.0, 1e4):
            self.assertEqual(log_evidence_partial(alpha, observations), 0.0)

    def test_single_draws_from_merged_blocks_are_constant(self):
        observations = [
            partial_observation([((0, 1), 1), ((2,), 0)]),
            partial_observation([((0,), 0), ((1, 2), 1)]),
        ]
        expected = 2.0 * math.log(2.0 / 3.0)
        for alpha in (1e-4, 1.0, 1e4):
            self.assertAlmostEqual(log_evidence_partial(alpha, observations), expected, places=14)

    def test_contributions_are_group_aligned(self):
        raw_observations = [
            [((0, 1), 4), ((2,), 1)],
            [((0, 1, 2), 3)],
            [((0,), 1), ((1, 2), 2)],
        ]
        observations = [partial_observation(blocks) for blocks in raw_observations]
        breakdown = partial_evidence_breakdown(0.7, observations)
        self.assertEqual(len(breakdown.contributions), 3)
        self.assertAlmostEqual(breakdown.total, math.fsum(breakdown.contributions), places=14)
        self.assertEqual(breakdown.contributions[1], 0.0)
        for index, raw_blocks in enumerate(raw_observations):
            with self.subTest(group=index):
                self.assertAlmostEqual(breakdown.contributions[index], partition_evidence_by_enumeration(raw_blocks, 0.7), places=11)


class PartialFitTests(unittest.TestCase):
    def test_shared_alpha_fit_with_mixed_partitions(self):
        raw_observations = [
            [((0, 1), 4), ((2,), 1)],
            [((0,), 1), ((1, 2), 3)],
            [((0, 1, 2), 6)],
            [((0, 2), 1), ((1,), 0)],
        ]
        observations = [partial_observation(blocks) for blocks in raw_observations]
        result = fit_alpha_partial(observations)
        self.assertNotEqual(result.fit.status, "flat")
        self.assertEqual((result.n_groups, result.n_nonempty_groups, result.n_informative_groups), (4, 4, 2))
        self.assertEqual(len(result.contributions), 4)
        self.assertEqual(result.contributions[2], 0.0)
        self.assertAlmostEqual(result.fit.log_evidence, math.fsum(result.contributions), places=13)
        self.assertAlmostEqual(result.fit.log_evidence, log_evidence_partial(result.fit.alpha, observations), places=13)

    def test_all_constant_observations_return_proved_flat(self):
        observations = [
            partial_observation([((0, 1, 2), 5)]),
            partial_observation([((0,), 0), ((1, 2), 0)]),
            partial_observation([((0, 1), 0), ((2,), 1)]),
        ]
        result = fit_alpha_partial(observations)
        self.assertEqual(result.fit.status, "flat")
        self.assertEqual(result.fit.alpha, 1.0)
        self.assertTrue(result.fit.converged)
        self.assertFalse(result.fit.budget_exhausted)
        self.assertEqual(result.fit.max_iterations_used, 0)
        self.assertEqual(result.n_informative_groups, 0)
        self.assertEqual(len(result.fit.candidates), 2)
        for alpha in (1e-4, 1.0, 1e4):
            self.assertAlmostEqual(log_evidence_partial(alpha, observations), result.fit.log_evidence, places=14)

    def test_flat_evidence_uses_block_prior_mass(self):
        observations = [partial_observation([((0, 1), 1), ((2,), 0)])]
        result = fit_alpha_partial(observations)
        self.assertEqual(result.fit.status, "flat")
        self.assertAlmostEqual(result.fit.log_evidence, math.log(2.0 / 3.0), places=14)


class PartialValidationTests(unittest.TestCase):
    def test_reject_repeated_class_inside_block(self):
        with self.assertRaisesRegex(ValueError, "repeated"):
            partial_observation([((0, 0, 1), 2), ((2,), 1)])

    def test_reject_overlapping_blocks(self):
        with self.assertRaisesRegex(ValueError, "repeated"):
            partial_observation([((0, 1), 2), ((1, 2), 1)])

    def test_reject_omitted_class(self):
        with self.assertRaisesRegex(ValueError, "omits class"):
            partial_observation([((0,), 2), ((2,), 1)])

    def test_reject_empty_block(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            partial_observation([((), 2), ((1, 2), 1), ((0,), 3)])

    def test_reject_negative_and_noninteger_counts(self):
        with self.assertRaises(ValueError):
            partial_observation([((0, 1), -1), ((2,), 1)])
        with self.assertRaises(TypeError):
            partial_observation([((0, 1), True), ((2,), 1)])
        with self.assertRaises(TypeError):
            partial_observation([((0, 1), 2.5), ((2,), 1)])

    def test_reject_inconsistent_k_across_samples(self):
        observations = [
            partial_observation([((0, 1), 2), ((2,), 1)]),
            partial_observation([((0,), 1), ((1,), 2)]),
        ]
        with self.assertRaisesRegex(ValueError, "same original classes"):
            log_evidence_partial(1.0, observations)

    def test_direct_invalid_dataclass_is_revalidated(self):
        bad = PartialObservation(blocks=(PartitionBlock((0, 0), 2),), k=1, n=2)
        with self.assertRaises(ValueError):
            log_evidence_partial(1.0, [bad])


if __name__ == "__main__":
    unittest.main()
