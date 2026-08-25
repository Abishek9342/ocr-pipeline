"""Character/Word Error Rate — the standard OCR/ASR accuracy metrics,
both just normalized Levenshtein edit distance at different granularities.
"""


def _levenshtein(a: list, b: list) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


def cer(prediction: str, ground_truth: str) -> float:
    """Character Error Rate: edit distance / reference length. 0.0 = perfect."""
    truth_chars = list(ground_truth)
    if not truth_chars:
        return 0.0 if not prediction else 1.0
    return _levenshtein(list(prediction), truth_chars) / len(truth_chars)


def wer(prediction: str, ground_truth: str) -> float:
    """Word Error Rate: edit distance over whitespace-split tokens / reference word count."""
    truth_words = ground_truth.split()
    if not truth_words:
        return 0.0 if not prediction.strip() else 1.0
    return _levenshtein(prediction.split(), truth_words) / len(truth_words)
