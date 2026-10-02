from __future__ import annotations

from tools.focus_first.candidate_scoring import (
    event_ranking_metrics,
    leave_one_group_out_splits,
    permute_columns_within_groups,
)


def test_event_ranking_metrics_are_event_level():
    result = event_ranking_metrics(
        ['a', 'a', 'a', 'b', 'b'],
        [0, 1, 0, 1, 0],
        [0.9, 0.8, 0.1, 0.7, 0.2],
    )
    assert result['events'] == 2
    assert result['rankable_events'] == 2
    assert result['top1_recall'] == 0.5
    assert result['top3_recall'] == 1.0
    assert result['mean_reciprocal_rank'] == 0.75


def test_missing_positive_event_is_not_silently_scored():
    result = event_ranking_metrics(['a', 'a', 'b'], [0, 0, 1], [0.9, 0.1, 0.5])
    assert result['events'] == 2
    assert result['rankable_events'] == 1
    assert result['missing_positive_events'] == 1


def test_ties_use_average_rank_not_input_order():
    first = event_ranking_metrics(['a', 'a'], [1, 0], [0.5, 0.5])
    second = event_ranking_metrics(['a', 'a'], [0, 1], [0.5, 0.5])
    assert first['median_rank'] == second['median_rank'] == 1.5
    assert first['top1_recall'] == second['top1_recall'] == 0.0


def test_mask_permutation_preserves_nonmask_fields_and_group_values():
    rows = [
        {'video': 'a', 'id': 1, 'mask': 10, 'coverage': 0.1},
        {'video': 'a', 'id': 2, 'mask': 20, 'coverage': 0.2},
        {'video': 'b', 'id': 3, 'mask': 30, 'coverage': 0.3},
    ]
    out = permute_columns_within_groups(
        rows, columns=['mask', 'coverage'], group_key='video', seed=7,
    )
    assert [row['id'] for row in out] == [1, 2, 3]
    assert {(row['mask'], row['coverage']) for row in out if row['video'] == 'a'} == {
        (10, 0.1), (20, 0.2),
    }
    assert out[2]['mask'] == 30 and out[2]['coverage'] == 0.3


def test_leave_one_video_out_never_leaks_video():
    groups = ['a', 'a', 'b', 'b', 'c']
    splits = leave_one_group_out_splits(groups)
    assert len(splits) == 3
    for training, validation in splits:
        assert set(groups[index] for index in training).isdisjoint(
            groups[index] for index in validation
        )


if __name__ == '__main__':
    tests = [
        value for name, value in sorted(globals().items())
        if name.startswith('test_') and callable(value)
    ]
    for test in tests:
        test()
        print(f'PASS {test.__name__}')
    print(f'{len(tests)} tests passed')
