import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward.reward import compute_reward, SCORE_ALL_PASS, SCORE_ERROR, SCORE_TIMEOUT, SCORE_PARTIAL_MAX

# ── known-correct solutions ──────────────────────────────────────────────────
CORRECT_CASES = [
    (
        "def add(a, b):\n    return a + b",
        ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]
    ),
    (
        "def is_even(n):\n    return n % 2 == 0",
        ["assert is_even(2) == True", "assert is_even(3) == False"]
    ),
    (
        "def reverse_string(s):\n    return s[::-1]",
        ["assert reverse_string('hello') == 'olleh'", "assert reverse_string('') == ''"]
    ),
    (
        "def factorial(n):\n    if n == 0: return 1\n    return n * factorial(n-1)",
        ["assert factorial(0) == 1", "assert factorial(5) == 120"]
    ),
    (
        "def max_list(lst):\n    return max(lst)",
        ["assert max_list([1,2,3]) == 3", "assert max_list([-1,-2,-3]) == -1"]
    ),
    (
        "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiouAEIOU')",
        ["assert count_vowels('hello') == 2", "assert count_vowels('xyz') == 0"]
    ),
    (
        "def flatten(lst):\n    return [x for sublist in lst for x in sublist]",
        ["assert flatten([[1,2],[3,4]]) == [1,2,3,4]"]
    ),
    (
        "def is_palindrome(s):\n    return s == s[::-1]",
        ["assert is_palindrome('racecar') == True", "assert is_palindrome('hello') == False"]
    ),
    (
        "def sum_list(lst):\n    return sum(lst)",
        ["assert sum_list([1,2,3]) == 6", "assert sum_list([]) == 0"]
    ),
    (
        "def capitalize_words(s):\n    return ' '.join(w.capitalize() for w in s.split())",
        ["assert capitalize_words('hello world') == 'Hello World'"]
    ),
]

# ── known-incorrect solutions ────────────────────────────────────────────────
INCORRECT_CASES = [
    (
        "def add(a, b):\n    return a - b",                          # wrong op
        ["assert add(1, 2) == 3"]
    ),
    (
        "def is_even(n):\n    return n % 2 == 1",                    # inverted
        ["assert is_even(2) == True"]
    ),
    (
        "def reverse_string(s):\n    return s",                      # no-op
        ["assert reverse_string('hello') == 'olleh'"]
    ),
    (
        "def factorial(n):\n    return n * factorial(n-1)",          # missing base case → RecursionError
        ["assert factorial(5) == 120"]
    ),
    (
        "def max_list(lst):\n    return min(lst)",                   # wrong fn
        ["assert max_list([1,2,3]) == 3"]
    ),
    (
        "def count_vowels(s):\n    return len(s)",                   # counts all chars
        ["assert count_vowels('hello') == 2"]
    ),
    (
        "def flatten(lst):\n    return lst",                         # no-op
        ["assert flatten([[1,2],[3,4]]) == [1,2,3,4]"]
    ),
    (
        "def is_palindrome(s):\n    return True",                    # always True
        ["assert is_palindrome('hello') == False"]
    ),
    (
        "import time\ndef sum_list(lst):\n    time.sleep(10)\n    return sum(lst)",  # timeout
        ["assert sum_list([1,2,3]) == 6"]
    ),
    (
        "def capitalize_words(s):\n    return s.upper()",            # wrong behavior
        ["assert capitalize_words('hello world') == 'Hello World'"]
    ),
]

# ── test runner ──────────────────────────────────────────────────────────────
def run_tests():
    print("=== Correct cases (expect score > 0) ===")
    correct_passed = 0
    for i, (code, tests) in enumerate(CORRECT_CASES):
        score, reason = compute_reward(code, tests)
        ok = score == SCORE_ALL_PASS
        correct_passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] case {i+1:02d}: score={score:.2f} | {reason}")

    print(f"\n=== Incorrect cases (expect score <= 0) ===")
    incorrect_passed = 0
    for i, (code, tests) in enumerate(INCORRECT_CASES):
        score, reason = compute_reward(code, tests)
        ok = score <= 0
        incorrect_passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] case {i+1:02d}: score={score:.2f} | {reason}")

    total = len(CORRECT_CASES) + len(INCORRECT_CASES)
    passed = correct_passed + incorrect_passed
    print(f"\n{passed}/{total} sandbox tests passed")
    if passed < total:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()