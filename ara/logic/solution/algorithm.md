# Algorithm

For each candidate pair, F1 computes task preservation, complete-block micro gain, and macro gain over five equally weighted families. Within each family, three development variants are summarized by median relative gain and checked against a fixed non-degeneration tolerance.

Eligible candidates are ranked lexicographically by structured-pruning non-degeneration, number of non-degraded families, worst-family median gain, structured-pruning median gain, macro relative gain, micro relative gain, and stable prompt ID. Selection first fills a quota of three unique-source rows per category, then fills to 30 using the same ordering.

If any category quota or total size fails, the algorithm returns development No-Go and does not execute confirmation.
