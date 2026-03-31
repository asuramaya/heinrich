from heinrich.bundle.scoring import parse_markdown_table, score_against_leaderboard

SAMPLE_TABLE = """
| Run | Score | Author |
|-----|------:|--------|
| ModelA | 1.11 | alice |
| ModelB | 1.15 | bob |
| ModelC | 1.20 | carol |
"""

def test_parse_markdown_table():
    rows = parse_markdown_table(SAMPLE_TABLE)
    assert len(rows) == 3
    assert rows[0]["Run"] == "ModelA"
    assert rows[0]["Score"] == "1.11"

def test_parse_empty():
    assert parse_markdown_table("") == []
    assert parse_markdown_table("no table here") == []

def test_score_better_than_all():
    rows = parse_markdown_table(SAMPLE_TABLE)
    signals = score_against_leaderboard(rows, 1.05)
    rank = [s for s in signals if s.kind == "leaderboard_rank"][0]
    assert rank.value == 1.0

def test_score_worst():
    rows = parse_markdown_table(SAMPLE_TABLE)
    signals = score_against_leaderboard(rows, 1.25)
    rank = [s for s in signals if s.kind == "leaderboard_rank"][0]
    assert rank.value == 4.0  # behind all 3

def test_score_gap():
    rows = parse_markdown_table(SAMPLE_TABLE)
    signals = score_against_leaderboard(rows, 1.15)
    gap = [s for s in signals if s.kind == "leaderboard_gap"][0]
    assert abs(gap.value - 0.04) < 0.01  # 1.15 - 1.11

def test_empty_leaderboard():
    assert score_against_leaderboard([], 1.0) == []
