# TakeMeter

> ### 👋 Start here
>
> **New to this repo? Read [RUNNING.md](RUNNING.md) first** — setup, the
> notebook, the baseline, and what to do when something breaks.
>
> Once `python test.py` passes:
>
> ```bash
> head -5 data/practice_labels.csv     # the shape your labels.csv needs
> ```
>
> Then open `takemeter.ipynb` **in this folder** — in VS Code, or with
> `jupyter notebook` if you prefer. Pick the kernel: the `.venv` inside this
> project. Run section 1, which reports the hardware you'll be training on.
> Everything else waits until you have data.
>
> Nothing to upload, nothing to connect, no accounts and no keys. The notebook
> runs on your machine and writes next to your code.
>
> **The rest of this file is your submission.** Fill it in as you go.

---

<!-- ─────────────────────────────────────────────────────────────────────────
     HOW TO USE THIS FILE

     Unit 5 asks for the first five sections. Unit 6 adds the five below them.

     Everything is pasted as TEXT. No screenshots, no images.

     ⚠️ The confusion matrix especially. The notebook prints one as a markdown
     table, ready to copy. A screenshot of a matrix earns nothing. Paste the
     table.
     ───────────────────────────────────────────────────────────────────────── -->

<!-- ═══════════════════════ UNIT 5 — THE BUILD ═══════════════════════ -->

## What This Does

My community is r/anime, a public subreddit for anime discussion. My classifier sorts each post into one of three labels based on what the post is doing: `question` (asks for a specific answer or recommendation for the author's own situation), `news` (reports an event or fact), or `discussion` (shares a view or opens a general conversation). I chose these because they match how people actually use the subreddit, and because each boundary can be stated in a sentence.

Distinctions I noticed while reading posts:
- Many posts ask for something, like a recommendation ("Should I watch AOT?") or an ID for an anime someone half remembers.
- Some posts just report news with no opinion, like a voice actor's death announcement or a production announcement.
- Some posts share a view or invite everyone's opinions, like "Batch releases vs weekly."
- Some posts mix two types, like "Opinion on Eighty Six?", which shares an opinion but also asks what others think.

---

## Label Taxonomy

### `discussion`

**Definition:** The post shares the author's view or opens a general conversation about a show, character, or topic.

**Example 1:**
> Opinion on Eighty Six? I picked up simply because I saw an advertisement for it at the back of a light novel... Though what do others think of this anime?

**Example 2:**
> How did they manage to put out FMAB and HxH Weekly? Random thought I had, but how did they manage to put out FMAB and especially HxH 2011 weekly? Everyone bashes MAPPA for working their employees to the bone and yet I've never heard the same be said about Bones or Madhouse... Is there some context I'm missing like studios schedules being less crowded back then?

### `news`

**Definition:** The post reports an event or fact, with no opinion attached.

**Example 1:**
> Jun Maeda (writer of Angel Beats, Charlotte, original author of Clannad, Kanon and AIR) has been diagnosed with throat cancer.

**Example 2:**
> Cyberpunk: Edgerunners season 2 director searched Cyberpunk 2077 with a "virtual camera" to find the best Night City spots to use in the anime

### `question`

**Definition:** The post asks for a specific answer or recommendation for the author's own situation.

**Example 1:**
> Should I actually watch aot. All my friends tell me to watch aot, I tried the first 10 episodes and got bored quickly, should I keep watching, note that aot is my first ever anime I usually watch tv shows, do you guys recommend anything else?

**Example 2:**
> Is there an anime where it's an Isekai to a fantasy world where the one transfered/reincarnated is like senku from Dr. Stone? I want to see a science guy go into a fantasy world, and test everything with the mind of a scientist... I wondered if such anime/manga exist

### The hardest boundary

**Which two labels:** question and discussion

**The decision rule I used every time:** If the post wants a specific answer for the author's own situation, like a recommendation or an ID, it's a question. If it shares a view or opens a general conversation, it's discussion, even if it ends with "thoughts?". A question that polls everyone's experiences ("What anime have you...?") is discussion. For mixed posts, I check the title: if the title is the request, it's a question, and if the title shares a view or feeling, it's discussion. The subreddit's own flair never decides the label.

---

## The Dataset

**Where the posts came from:** Public posts from r/anime, collected in October 2026. I gathered them by script (an Apify Reddit scraper, and a script that queried the Arctic Shift archive for posts on random days from the past year) plus some copied by hand. I removed rewatch and episode threads, bot-made threads, spam, and posts that were mainly about minors, self-harm, abuse allegations, or explicit content. I also removed usernames and personal profile links.

**How I labelled them:** I labelled 20 posts cold, before showing any label to an AI tool, and marked them `cold` in the note column. They were labelled across my working session, not in one block, and three of them (Jun Maeda, the Edgerunners director, and the HxH weekly post) were also my README examples. I used Claude to pre-label 221 more rows (note: `prelabeled by Claude, checked`), and I labelled or settled the other 9 with its help in chat. My first batch was mostly questions and had very little news (about 7%), so I collected a second batch aimed at announcements and opinion posts. To stay under the 70% cap I also removed 17 random `question` rows from the first batch.

**Counts per label:**

| Label | Count | Share |
|---|---|---|
| question | 133 | 53.2% |
| discussion | 74 | 29.6% |
| news | 43 | 17.2% |
| **Total** | 250 | 100% |

**Three hard cases**

**1.**
> *The post:* "Opinion on Eighty Six? I picked up simply because I saw an advertisement for it at the back of a light novel... Though what do others think of this anime?"
>
> *Could have been:* question or discussion
>
> *I chose, because:* discussion. The author has already started the show and shares an opinion (hard to follow, some parts feel cliche), then asks what others think. That is a general conversation, not a request for a specific answer or recommendation for their own situation.

**2.**
> *The post:* "Is there an anime where it's an Isekai to a fantasy world where the one transfered/reincarnated is like senku from Dr. Stone? I want to see a science guy go into a fantasy world..."
>
> *Could have been:* question or discussion
>
> *I chose, because:* question. The subreddit flair says Discussion, but the post asks for a specific recommendation that fits the author's own taste, which is what my rule calls a question. The flair did not decide the label.

**3.**
> *The post:* "What Anime is hard to watch again because you know the ending already? I was trying to rewatch Remake our Life!... What anime have you found that you have no interesting going back to...?"
>
> *Could have been:* question or discussion
>
> *I chose, because:* discussion. The title is phrased as a question, but it polls everyone's own experiences and does not ask for anything for the author's situation, so it opens a general conversation.

---

## The Training Run

**Base model:** distilbert-base-uncased

**Settings:** 3 epochs, learning rate 2e-5, batch size 16, 128 tokens per post, seed 42. Trained on an Apple Silicon GPU (mps) with PyTorch 2.14.1.

**Anything I changed from the defaults, and why:** I changed no training settings. The only edit I made in the notebook was setting `LABELS = ["question", "discussion", "news"]` to match my CSV.

**Split sizes:** train 175 / validation 37 / test 38. Test split per label: question 20, discussion 11, news 7.

---

## How I Used AI

**Moment 1**

- *What I asked for:* I asked Claude to help me design labels for r/anime after I read posts, and I pasted real posts to check which label each one fit.
- *What came back:* Claude suggested three labels: question, news, and opinion. It also flagged posts that sat between two labels, like a post that asks a question but also shares a view.
- *What I changed:* I noticed that nearly every opinion post invites replies, so opinion and discussion could not be separated, and I merged them into one `discussion` label. I also added to my decision rule that a question polling everyone's experiences is discussion, and that for mixed posts I check whether the title is the request.

**Moment 2**

- *What I asked for:* I asked Claude how to collect about 200 posts without copying each one by hand, after Reddit's public pages stopped working for scrapers.
- *What came back:* Claude wrote scripts, including one that samples random days from the Arctic Shift archive, and suggested an Apify scraper. It warned that Reddit's terms restrict using posts to train models and told me to check with my instructors.
- *What I changed:* I ran the Apify scraper and the archive script, dropped posts about minors, self-harm, and abuse allegations, and removed usernames and profile links. 

**Pre-labelling disclosure:** Claude pre-labelled 221 of my 250 rows from posts I collected by script. Those rows say `prelabeled by Claude, checked` in the note column. Nine more rows were labelled with Claude's help in chat, and 20 are cold labels I wrote before seeing any label from Claude.

<!-- ═══════════════════════ UNIT 6 — THE TEST ═══════════════════════

     Don't fill these in during unit 5.
     ═══════════════════════════════════════════════════════════════════ -->

---

## Baseline vs. Trained

<!-- Both models on the same posts. `python baseline.py --trained results.json`
     prints this table for you. -->

| Measure | Baseline | Trained | Difference |
|---|---|---|---|
| Overall accuracy |  |  |  |
| Macro F1 |  |  |  |
| F1 — `label_one` |  |  |  |
| F1 — `label_two` |  |  |  |

**What I predicted before I looked:**
<!-- Milestone 1 asks you to write this BEFORE seeing the trained numbers. A
     prediction made afterwards isn't one. -->

**What the gap actually means:**
<!-- If the baseline matched your trained model, your fine-tuning added
     nothing — and that is a real finding, not a failure. Say it plainly. -->



---

## Run Log — Before

<!-- Five criteria across three seeds. The notebook's section 6 prints the
     spread table; the Target and Verdict columns are yours. -->

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

### Confusion matrix

<!-- ⚠️ TYPED AS A MARKDOWN TABLE. The notebook prints one ready to paste.
     An image of a matrix earns nothing. -->

| true \ predicted |  |  |  |
|---|---|---|---|
| **** |  |  |  |
| **** |  |  |  |
| **** |  |  |  |

**My biggest off-diagonal number, and what it means:**
<!-- Not "the model made mistakes" — WHICH boundary it didn't learn, and which
     direction. "7 real analysis posts were called hot_take and only 3 went the
     other way" is a direction, not just an error rate. -->



---

## Verdicts and Diagnoses

<!-- MET or MISSED against LAST UNIT's target. The target has to hold across
     all three seeds, not turn up sometimes. -->

| # | Criterion | Target | Verdict | How I decided |
|---|---|---|---|---|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |
| 4 |  |  |  |  |
| 5 |  |  |  |  |

**Diagnoses**

<!-- For each miss: the cause, and how you know. The four common causes are:
     too few examples for a label, a boundary you applied inconsistently, a
     genuinely hard label pair, and a task the model can't reach from this
     much data.

     ⚠️ Use your agreement report as evidence. It is the only instrument you
     have that can tell a LABELLING problem from a MODEL problem, and this
     section is graded on whether you used it that way. -->



---

## Agreement Report

<!-- Your rate against the staff set, and every disagreement adjudicated.

     Remember you labelled these 30 under the STAFF taxonomy in
     data/staff_taxonomy.md, not your own — so every argument below is made
     from those definitions and those decision rules. -->

**Agreement rate:** ___ / 30 = ___%

<!-- Nobody grades this number. A 60% who argues every disagreement from the
     stated rules beats a 95% who wrote "staff was right" nine times. Several
     of the 30 were chosen because they're genuinely ambiguous — you should be
     winning some of these. -->

**Disagreements**

<!-- Three lines each: the post, both labels, and who you think is right and
     why — grounded in the staff definitions you were both applying.

     Then sort each into one of three piles:
       (a) the rule covered it and I applied it loosely → a consistency problem
       (b) the rule genuinely doesn't say               → a gap in the definitions
       (c) the rule is ambiguous here and my reading is defensible → argue it.
           This is a legitimate win.

     Pile (a) is the one that matters most for your diagnosis: if you applied a
     written rule two different ways on 30 posts, that is direct evidence about
     what you did across your own 200. -->

**1.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**2.**
> *The post:*
>
> *Staff said / I said:*
>
> *My call, and why:*
>
> *Which pile:*

**What the pattern in my disagreements tells me:**



---

## The Improvement

**What I changed:**

**Which diagnosis pointed at it:**

### Run Log — After

| Criterion | Target | Seed 42 | Seed 7 | Seed 2024 | Verdict |
|---|---|---|---|---|---|
| 1.  |  |  |  |  |  |
| 2.  |  |  |  |  |  |
| 3.  |  |  |  |  |  |
| 4.  |  |  |  |  |  |
| 5.  |  |  |  |  |  |

**Did it help, and how do I know:**

<!-- If it didn't, say so. Relabelling that didn't help is a genuinely
     interesting result and earns full credit. -->



---

## What's Still Broken

<!-- For each criterion still missed: what you'd do, and why you stopped. -->



**The gap between what I meant my labels to capture and what the model
learned:**
<!-- Two sentences. Your confusion matrix is the evidence. -->



<!-- ═════════════════════════════════════════════════════════════════════

     SUBMISSION CHECKLIST — unit 5

       [ ] criteria.md has five numbered criteria, each naming a NUMBER
       [ ] Each has a reason underneath tied to your data or taxonomy
       [ ] labels.csv: at least 150 rows, text/label/note, ONE file not split
       [ ] No label above 70%
       [ ] All five unit 5 sections have real content
       [ ] Label Taxonomy includes the decision rule for your hardest boundary
       [ ] The Dataset includes three hard cases
       [ ] results.json and test_split.csv committed (the notebook does this)
       [ ] At least four new commits
       [ ] Repository URL submitted — WRITE IT DOWN

     SUBMISSION CHECKLIST — unit 6

       [ ] Baseline vs. Trained table, with your prediction written beforehand
       [ ] Run Log — Before, five criteria across three seeds
       [ ] Confusion matrix TYPED AS A MARKDOWN TABLE
       [ ] A verdict on every criterion
       [ ] A diagnosis for every miss, using the agreement report as evidence
       [ ] Agreement Report with every disagreement adjudicated
       [ ] One improvement, with Run Log — After
       [ ] What's Still Broken
       [ ] results_three_seeds_before.json, results_three_seeds_after.json,
           baseline_results.json and
           agreement_results.json committed
       [ ] At least four new commits
       [ ] The SAME repository URL as last unit

     Do not delete and recreate this repository.
     ═════════════════════════════════════════════════════════════════════ -->

---

📖 **How to run this project: [RUNNING.md](RUNNING.md)**