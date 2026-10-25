"""A few Jev-style decisions across the three task formats (choice, noul, score)."""

import sys

from jev import DEFAULT_MODEL, Jev

CASES = [
    ("choice",
     "User wants to change their shipping address.",
     ["order_status", "change_shipping", "cancel_order"]),
    ("noul",  # binary verification: two options spelling out the criteria
     "Claim: The Eiffel Tower is in Berlin.",
     ["yes - the claim is factually correct", "no - the claim is factually incorrect"]),
    ("score",  # ordinal grading
     "Rate the politeness of: 'Give me my refund now, you idiots.'",
     ["1 - very rude", "2 - rude", "3 - neutral", "4 - polite", "5 - very polite"]),
    ("choice, none-of-the-above",
     "Which of these is a prime number?",
     ["21", "33", "49", "none of the above"]),
    ("choice, K=12 (option 10 is correct)",
     "Which month has the fewest days?",
     ["January", "March", "April", "May", "June", "July", "August", "September",
      "October", "February", "November", "December"]),
]


def main():
    jev = Jev(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL)
    for kind, state, options in CASES:
        d = jev.score(state, options)
        print(f"\n[{kind}] {state}   ({d.prompt_tokens} prompt tok, {d.seconds * 1e3:.0f} ms)")
        for i, (o, p, q) in enumerate(zip(options, d.probs, d.first_probs), 1):
            mark = "<-" if i - 1 == d.best else "  "
            print(f"  {mark} [{i:2d}] {p:7.2%}   (first-token {q:7.2%})  {o}")


if __name__ == "__main__":
    main()
