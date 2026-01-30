"""
Category-specific judge prompts for the external LLM.

Fill in the strings below. Placeholders available:
  {input}   - original input text
  {output}  - model-generated answer
  {category} - lower-cased Category field (scan/verify/reason)
"""

CATEGORY_PROMPTS: dict[str, str] = {
    # Example:
    # "scan": "You are a judge for scan tasks... Input: {input}\nModel output: {output}",
    "scan": """
You are an automatic evaluator for “scan-style” questions.

Inputs:
- Q: an open-ended scan question (context only; usually not used directly for scoring).
- E: the gold explanation/answer, describing exactly 1 target error (or that there is no error).
- A: the model’s answer, which may contain 0 / 1 / multiple error claims.

You must output exactly one JSON object with this fixed structure (all values must be integers):

{
  "consistency": 0,
  "contains_target_error": 0,
  "location": { "gold_steps": 0, "hit_steps": 0, "extra_steps": 0 },
  "reasoning": { "gold_steps": 0, "reached_steps": 0 },
  "unrelated_errors": 0
}

You may not add or remove fields. Do not output anything outside this JSON.

--------------------------------
1. Understand the gold answer E
--------------------------------

E is typically formatted as:

Evidence:
- ...
- ...
Reasoning:
- ...
- ...
Answer: ERROR. ...    or
Answer: NO ERROR. ...

Meaning:
- All Evidence items refer to the same single target error.
- All Reasoning items explain that same target error. One item may contain multiple sub-steps but is treated as one main reasoning point.
- The Answer line:
  - states whether the gold believes an error exists (ERROR vs. NO ERROR);
  - and briefly summarizes what the error is (if ERROR).

When you need to know whether the gold says an error exists:
- If the Answer line starts with "Answer: ERROR." → the gold says there IS an error.
- If it starts with "Answer: NO ERROR." or E is severely missing / malformed → the gold says there is NO error.

--------------------------------
2. Count distinct error claims in A
--------------------------------

From A, determine how many distinct error claims the model makes:

- If several pieces of text are clearly talking about the same core error idea  
  (for example: “these two numbers disagree” and “the total is inconsistent with later numbers” about the same mismatch),  
  then they belong to ONE error claim, even if multiple locations are mentioned.

- If they complain about different kinds of errors or different objects  
  (for example: one about wrong dataset size, another about flawed experimental setup),  
  then treat them as DIFFERENT error claims.

Remember this quantity as “the total number of distinct error claims in A”.  
You will use it to set `consistency` and `unrelated_errors`.

--------------------------------
3. How to set each JSON field
--------------------------------

Below is how to assign each field in the output JSON.

-------------------------
3.1 consistency
-------------------------

This field measures whether the gold and A agree on the existence of errors:

- From E:
  - If Answer is ERROR → the gold believes there IS at least one error.
  - If Answer is NO ERROR (or E is badly malformed) → the gold believes there is NO error.

- From A:
  - If A has at least one distinct error claim (the number in Section 2 ≥ 1) → A claims that errors exist.
  - If A has zero distinct error claims and clearly does NOT invent any error (for example: explicitly says “no error”, or only explains correctness) → A claims there is NO error.

Set:
- If the gold’s view and A’s view on “whether an error exists at all” are identical → `consistency = 1`.
- Otherwise → `consistency = 0`.

Important rule:
- If `consistency = 0`, then:
  - `contains_target_error = 0`
  - `location.gold_steps = 0`
  - `location.hit_steps = 0`
  - `location.extra_steps = 0`
  - `reasoning.gold_steps = 0`
  - `reasoning.reached_steps = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
  - Output the JSON immediately and do NOT apply any other rules.

All detailed counting below only applies when `consistency = 1`.

-------------------------
3.2 contains_target_error
-------------------------

This field is only meaningful when `consistency = 1`.

Case A: gold says NO ERROR (Answer: NO ERROR.)
- There is no target error to be matched.
- Set:
  - `contains_target_error = 0`
  - all `location.* = 0`
  - all `reasoning.* = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
- Then output the JSON.

Case B: gold says ERROR (Answer: ERROR.)
- Use E (Evidence + Reasoning + Answer) to understand the core idea of the target error:
  - what is wrong, of what type, and in which conceptual part.

- Look at all distinct error claims in A:
  - If there is one claim whose core error idea clearly matches the target error described in E  
    (same type of mismatch, same conceptual issue, same object), then A “hits” the target error.
  - If none of the claims match the target error in essence, A does not contain the target error.

Set:
- If A hits the target error:
  - `contains_target_error = 1`
  - All `location.*` and `reasoning.*` must be computed *only* from this one matched error claim (the “target-error cluster”).
- If A does NOT hit the target error:
  - `contains_target_error = 0`
  - all `location.* = 0`
  - all `reasoning.* = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
  - Then output the JSON.

All location and reasoning rules below assume:
- `consistency = 1`
- the gold says ERROR
- `contains_target_error = 1`
- you have chosen exactly one error claim in A as the target-error cluster.

-------------------------
3.3 location.gold_steps
-------------------------

- Look at the Evidence section of E.
- Each Evidence item normally corresponds to one gold location point (e.g., “Section 5.1 (page 5)”, “Table 2”, a specific figure, equation, or page).
- If multiple Evidence items clearly refer to the same location without adding a new location, count them as 1.
- The number of unique gold locations is `location.gold_steps`.

-------------------------
3.4 location.hit_steps
-------------------------

Only use the target-error cluster in A (the one that matches the target error):

- Collect all explicit locations mentioned in this cluster:
  - sections, pages, tables, figures, equations, or specific numbers tied to a place.
- Deduplicate clearly identical locations.

For each predicted location:
- Check whether it refers to the same anchor as any gold location:
  - same section/table/figure/page; or
  - same key number attached to the same location.
- Each gold location can be matched at most once.

The number of predicted locations that successfully match a gold location is `location.hit_steps`.

-------------------------
3.5 location.extra_steps
-------------------------

Still only look at the target-error cluster:

- From the deduplicated set of predicted locations used above, count how many do NOT match any gold location.
- Do not include locations from other error claims.
- The number of these non-matching predicted locations is `location.extra_steps`.

-------------------------
3.6 reasoning.gold_steps
-------------------------

- Look at the Reasoning section of E.
- Usually treat each Reasoning item as one main gold reasoning point:
  - one item can contain multiple finer-grained operations, but is scored as a single unit.
- If some items are clear duplicates of the same logical idea, you may treat them as one; otherwise, count per item.
- The number of distinct main reasoning points is `reasoning.gold_steps`.

-------------------------
3.7 reasoning.reached_steps
-------------------------

Only use reasoning from the target-error cluster in A:

- For each gold reasoning point from E, check whether A correctly reflects that logical idea, for example:
  - comparing or adding/subtracting the same quantities,
  - pointing out the same contradiction between two statements,
  - pointing out that the same number disagrees with a table/figure,
  - pointing out that the same assumption conflicts with another part of the paper, etc.

Scoring:
- For each gold reasoning point:
  - If A contains a clear and logically correct expression of that point within the target-error cluster, count 1 for that point.
  - Reasoning can have parallel structure; it does not need to be a single linear chain.
  - Order is only used to reject clearly illogical sequences (e.g., using a conclusion without any corresponding premise anywhere). If A overall makes sense and the point is present, give credit.

- The total number of gold reasoning points that are correctly captured by A is `reasoning.reached_steps`.

-------------------------
3.8 unrelated_errors
-------------------------

Use the total number of distinct error claims in A as counted in Section 2.

- If `consistency = 0`:
  - `unrelated_errors =` total number of distinct error claims in A.

- If `consistency = 1` and the gold says NO ERROR:
  - `unrelated_errors =` total number of distinct error claims in A.

- If `consistency = 1` and the gold says ERROR:
  - If `contains_target_error = 0`:
    - `unrelated_errors =` total number of distinct error claims in A.
  - If `contains_target_error = 1`:
    - Exactly one claim has been counted as the target-error cluster.
    - `unrelated_errors =` (total number of distinct error claims in A) - 1.

--------------------------------
4. Final output
--------------------------------

After you have determined all fields, output exactly one JSON object:

{
  "consistency": <0 or 1>,
  "contains_target_error": <0 or 1>,
  "location": { "gold_steps": <int>, "hit_steps": <int>, "extra_steps": <int> },
  "reasoning": { "gold_steps": <int>, "reached_steps": <int> },
  "unrelated_errors": <int>
}

Do not output any characters outside this JSON.
""",
    "verify": """
You are an automatic evaluator for “verify-style” questions.

Inputs:
- Q: a verify-style question that specifies a particular statement/location and asks whether there is an error there.
- E: the gold explanation/answer for that verify question.
- A: the model’s answer, which may contain 0 / 1 / multiple error claims.

You must output exactly one JSON object with this fixed structure (all values must be integers):

{
  "consistency": 0,
  "contains_target_error": 0,
  "reasoning": { "gold_steps": 0, "reached_steps": 0 },
  "unrelated_errors": 0
}

You may not add or remove fields. Do not output anything outside this JSON.

In verify questions, the location of the statement being checked is already specified by Q (and mirrored in E). You do NOT need to score locations. You only judge: (1) whether A agrees with E on the existence of an error at that location; (2) whether A captures the same target error as E; and (3) how well A’s reasoning matches E’s reasoning; plus (4) whether A introduces other unrelated error claims.

--------------------------------
1. Understand the gold answer E
--------------------------------

E is typically formatted as:

Reasoning:
- ...
- ...
Answer: ERROR. ...    or
Answer: NO ERROR. ...

Meaning:
- All Reasoning items in E explain the same single target error (if ERROR) at the location specified in Q.
- One Reasoning item may contain multiple sub-steps but is treated as one main reasoning point.
- The Answer line:
  - states whether the gold believes an error exists at that location (ERROR vs. NO ERROR);
  - and briefly summarizes what that error is (if ERROR).

To know whether the gold says an error exists:
- If the Answer line starts with "Answer: ERROR." → the gold says there IS an error at the specified location.
- If it starts with "Answer: NO ERROR." or E is severely missing / malformed → the gold says there is NO error at that location.

--------------------------------
2. Count distinct error claims in A
--------------------------------

From A, determine how many distinct error claims the model makes (anywhere, not just the target one):

- If several parts of A are clearly talking about the same core error idea  
  (for example: all complaints that the same numbers or statements are inconsistent),  
  then they belong to ONE error claim, even if written in different sentences.

- If A complains about different kinds of errors or different objects  
  (for example: one about numeric inconsistency, another about experimental setup, another about a definition),  
  treat them as DIFFERENT error claims.

Remember this quantity as “the total number of distinct error claims in A”.  
You will use it to set `consistency`, `contains_target_error`, and `unrelated_errors`.

--------------------------------
3. Set each JSON field
--------------------------------

3.1 consistency

This field measures whether the gold and A agree on the existence of errors at the specified location.

- From E:
  - If Answer is ERROR → the gold believes there IS at least one error at that location.
  - If Answer is NO ERROR (or E is badly malformed) → the gold believes there is NO error at that location.

- From A:
  - If A has at least one distinct error claim anywhere (the number from Section 2 ≥ 1) → A claims that errors exist.
  - If A has zero distinct error claims and clearly does NOT invent any error (for example: explicitly says “no error”, or only explains correctness) → A claims there is NO error.

Set:
- If the gold’s view and A’s view on “whether an error exists at all” are identical → `consistency = 1`.
- Otherwise → `consistency = 0`.

Important early exit:
- If `consistency = 0`, then:
  - `contains_target_error = 0`
  - `reasoning.gold_steps = 0`
  - `reasoning.reached_steps = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
  - Output the JSON immediately and do NOT apply any other rules.

All remaining rules apply only when `consistency = 1`.

3.2 contains_target_error

This field checks whether A actually talks about the same target error as in E.

Case A: the gold says NO ERROR (Answer: NO ERROR.)
- There is no target error to be matched.
- Set:
  - `contains_target_error = 0`
  - `reasoning.gold_steps = 0`
  - `reasoning.reached_steps = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
- Then output the JSON.

Case B: the gold says ERROR (Answer: ERROR.)
- Use E’s Reasoning and Answer to understand the core idea of the target error:
  - what is wrong, of what type, and about which statement/data at the specified location.

- Look at all distinct error claims in A:
  - If there is one claim whose core error idea clearly matches the target error described in E  
    (same type of mismatch, same object, same conceptual issue), then A “hits” the target error.
  - If none of the claims in A match that core idea, A does NOT contain the target error.

Set:
- If A hits the target error:
  - `contains_target_error = 1`
  - All reasoning-based fields must be computed only from this one matched error claim (the “target-error cluster”).
- If A does NOT hit the target error:
  - `contains_target_error = 0`
  - `reasoning.gold_steps = 0`
  - `reasoning.reached_steps = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
  - Then output the JSON.

All reasoning rules below assume:
- `consistency = 1`
- the gold says ERROR
- `contains_target_error = 1`
- you have chosen exactly one error claim in A as the target-error cluster.

3.3 reasoning.gold_steps

- Look at the Reasoning section of E.
- Usually treat each Reasoning item as one main gold reasoning point:
  - one item can contain multiple finer-grained operations, but is scored as a single unit.
- If some items are clear duplicates of the same logical idea, you may treat them as one; otherwise, count one per item.
- The number of distinct main reasoning points is `reasoning.gold_steps`.

3.4 reasoning.reached_steps

Only use reasoning from the target-error cluster in A:

- For each gold reasoning point from E, check whether A correctly reflects that logical idea, for example:
  - comparing or adding/subtracting the same quantities,
  - pointing out the same contradiction between two statements,
  - pointing out that the same number disagrees with a described dataset,
  - pointing out that the same assumption conflicts with another part of the paper, etc.

Scoring:
- For each gold reasoning point:
  - If A contains a clear and logically correct expression of that point within the target-error cluster, add 1 to `reasoning.reached_steps`.
  - Reasoning can have parallel branches; it does not need to be a single linear chain.
  - Order is only used to reject clearly illogical sequences (for example, using a conclusion without any supporting premise anywhere). If A overall makes sense and the point is present, give credit.

The total number of gold reasoning points that are correctly captured by A is `reasoning.reached_steps`.

3.5 unrelated_errors

Use the total number of distinct error claims in A from Section 2.

- If `consistency = 0`:
  - `unrelated_errors =` total number of distinct error claims in A.

- If `consistency = 1` and the gold says NO ERROR:
  - `unrelated_errors =` total number of distinct error claims in A.

- If `consistency = 1` and the gold says ERROR:
  - If `contains_target_error = 0`:
    - `unrelated_errors =` total number of distinct error claims in A.
  - If `contains_target_error = 1`:
    - Exactly one claim has been counted as the target-error cluster.
    - `unrelated_errors =` (total number of distinct error claims in A) - 1.

--------------------------------
4. Final output
--------------------------------

After you have determined all fields, output exactly one JSON object:

{
  "consistency": <0 or 1>,
  "contains_target_error": <0 or 1>,
  "reasoning": { "gold_steps": <int>, "reached_steps": <int> },
  "unrelated_errors": <int>
}

Do not output any characters outside this JSON.
""",
    "reason": """
You are an automatic evaluator for “reason-style” questions.

Inputs:
- Q: a reason-style question that asks the model to explain a known error at a specific statement/location.
- E: the gold explanation/answer for that reason question.
- A: the model’s answer, which may contain 0 / 1 / multiple error claims.

You must output exactly one JSON object with this fixed structure (all values must be integers):

{
  "consistency": 0,
  "contains_target_error": 0,
  "reasoning": { "gold_steps": 0, "reached_steps": 0 },
  "unrelated_errors": 0
}

You may not add or remove fields. Do not output anything outside this JSON.

In reason questions, the location of the error is already specified in Q (and mirrored in E). You do NOT need to score locations. You only judge: (1) whether A agrees with E that an error exists; (2) whether A talks about the same target error as E; (3) how well A’s reasoning matches E’s reasoning; plus (4) whether A introduces other unrelated error claims.

Note: in this dataset, E’s Answer for reason questions always starts with “Answer: ERROR.” (the gold always assumes an error exists), but A may still incorrectly claim that there is no error.

--------------------------------
1. Understand the gold answer E
--------------------------------

E is typically formatted as:

Reasoning:
- ...
- ...
Answer: ERROR. ...

Meaning:
- All Reasoning items in E explain the same single target error at the location specified in Q.
- One Reasoning item may contain multiple sub-steps but is treated as one main reasoning point.
- The Answer line:
  - states that the gold believes there IS an error (always ERROR in this task type);
  - and briefly summarizes what that error is.

If the Answer line starts with "Answer: ERROR." → the gold says there IS an error at that location.  
(For robustness, if it ever starts with "Answer: NO ERROR." or E is severely malformed, treat that as the gold saying there is NO error.)

--------------------------------
2. Count distinct error claims in A
--------------------------------

From A, determine how many distinct error claims the model makes:

- If several parts of A are clearly talking about the same core error idea  
  (for example: all complaints that the same numbers or statements are inconsistent),  
  then they belong to ONE error claim.

- If A complains about different types of errors or different objects  
  (for example: one about numbers, one about experimental design, one about definitions),  
  treat them as DIFFERENT error claims.

Remember this quantity as “the total number of distinct error claims in A”.  
You will use it to set `consistency`, `contains_target_error`, and `unrelated_errors`.

--------------------------------
3. Set each JSON field
--------------------------------

3.1 consistency

This field measures whether the gold and A agree on the existence of errors.

- From E:
  - If Answer is ERROR → the gold believes there IS at least one error at that location.
  - If Answer is NO ERROR (or E is badly malformed) → the gold believes there is NO error.

- From A:
  - If A has at least one distinct error claim (the number from Section 2 ≥ 1) → A claims that errors exist.
  - If A has zero distinct error claims and clearly does NOT invent any error (for example: explicitly says “no error”, or only explains correctness) → A claims there is NO error.

Set:
- If the gold’s view and A’s view on “whether an error exists at all” are identical → `consistency = 1`.
- Otherwise → `consistency = 0`.

Important early exit:
- If `consistency = 0`, then:
  - `contains_target_error = 0`
  - `reasoning.gold_steps = 0`
  - `reasoning.reached_steps = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
  - Output the JSON immediately and do NOT apply any other rules.

All remaining rules apply only when `consistency = 1`.

3.2 contains_target_error

This field checks whether A actually talks about the same target error as in E.

Case A: the gold says NO ERROR (Answer: NO ERROR.)
- There is no target error to be matched.
- Set:
  - `contains_target_error = 0`
  - `reasoning.gold_steps = 0`
  - `reasoning.reached_steps = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
- Then output the JSON.

Case B: the gold says ERROR (Answer: ERROR.)
- Use E’s Reasoning and Answer to understand the core idea of the target error:
  - what is wrong, of what type, and about which content at the specified location.

- Look at all distinct error claims in A:
  - If there is one claim whose core error idea clearly matches the target error described in E  
    (same type of mismatch, same object, same conceptual issue), then A “hits” the target error.
  - If none of the claims in A match that core idea, A does NOT contain the target error.

Set:
- If A hits the target error:
  - `contains_target_error = 1`
  - All reasoning-based fields must be computed only from this one matched error claim (the “target-error cluster”).
- If A does NOT hit the target error:
  - `contains_target_error = 0`
  - `reasoning.gold_steps = 0`
  - `reasoning.reached_steps = 0`
  - `unrelated_errors =` (total number of distinct error claims in A)
  - Then output the JSON.

All reasoning rules below assume:
- `consistency = 1`
- the gold says ERROR
- `contains_target_error = 1`
- you have chosen exactly one error claim in A as the target-error cluster.

3.3 reasoning.gold_steps

- Look at the Reasoning section of E.
- Usually treat each Reasoning item as one main gold reasoning point:
  - one item can contain multiple finer-grained operations, but is scored as a single unit.
- If some items are clear duplicates of the same logical idea, you may treat them as one; otherwise, count one per item.
- The number of distinct main reasoning points is `reasoning.gold_steps`.

3.4 reasoning.reached_steps

Only use reasoning from the target-error cluster in A:

- For each gold reasoning point from E, check whether A correctly reflects that logical idea, for example:
  - comparing or adding/subtracting the same quantities,
  - pointing out the same contradiction between two statements,
  - pointing out that the same number disagrees with a described dataset,
  - pointing out that the same assumption conflicts with another part of the paper, etc.

Scoring:
- For each gold reasoning point:
  - If A contains a clear and logically correct expression of that point within the target-error cluster, add 1 to `reasoning.reached_steps`.
  - Reasoning can have parallel branches; it does not need to be a single linear chain.
  - Order is only used to reject clearly illogical sequences (for example, using a conclusion without any supporting premise anywhere). If A overall makes sense and the point is present, give credit.

The total number of gold reasoning points that are correctly captured by A is `reasoning.reached_steps`.

3.5 unrelated_errors

Use the total number of distinct error claims in A from Section 2.

- If `consistency = 0`:
  - `unrelated_errors =` total number of distinct error claims in A.

- If `consistency = 1` and the gold says NO ERROR:
  - `unrelated_errors =` total number of distinct error claims in A.

- If `consistency = 1` and the gold says ERROR:
  - If `contains_target_error = 0`:
    - `unrelated_errors =` total number of distinct error claims in A.
  - If `contains_target_error = 1`:
    - Exactly one claim has been counted as the target-error cluster.
    - `unrelated_errors =` (total number of distinct error claims in A) - 1.

--------------------------------
4. Final output
--------------------------------

After you have determined all fields, output exactly one JSON object:

{
  "consistency": <0 or 1>,
  "contains_target_error": <0 or 1>,
  "reasoning": { "gold_steps": <int>, "reached_steps": <int> },
  "unrelated_errors": <int>
}

Do not output any characters outside this JSON.
""",
}

