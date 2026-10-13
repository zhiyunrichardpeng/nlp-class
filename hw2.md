---
layout: default
img: bratconll2k
caption: Explore phrasal chunking interactively using Brat
title: Homework 2 | BERT Finetuning for Robust Phrasal Chunking
active_tab: homework
---

# Homework 2

<span class="text-info">Start on {{ site.hwdates[2].startdate }}</span> |
<span class="text-warning">Due on {{ site.hwdates[2].deadline }}</span>

## Conceptual question

<span class="text-info">Out on {{ site.hwdates[2].startdate }}</span> 
{% if site.hwdates[2].hwc-url %}
<span>Posted on [{{site.hwc}}]({{ site.hwdates[2].hwc-url }}).</span> 
{% endif %}

Please submit your conceptual question to Gradescope and your programming question to Coursys.

# Programming Homework 2: BERT Finetuning for Robust Phrasal Chunking

## Getting Started

Use the public starter repository below and your group's existing SFU GitHub repository. Replace `USER` and `GROUP` with the repository owner's SFU username and your group name. Run these commands from a parent directory that will contain both repositories.

For a fresh starter checkout:

    git clone https://github.com/zhiyunrichardpeng/nlp-class-hw.git

To update an existing starter checkout instead, first save any local work, then run from its parent directory:

    git -C nlp-class-hw pull --ff-only origin master

Clone your group repository if necessary:

    git clone git@github.sfu.ca:USER/nlpclass-{{ site.semcode }}-g-GROUP.git

Copy the **contents** of `nlp-class-hw/bertchunker/` into `nlpclass-{{ site.semcode }}-g-GROUP/hw2/`. The `hw2` directory should directly contain `answer/`, `data/`, `requirements.txt`, `zipout.py`, `zipsrc.py`, and `check.py`; do not add another nested `bertchunker` directory. Preserve any work already in your group's `hw2` directory. Run all remaining commands with `hw2` as the current directory.

Use Python 3.10 or later. On Linux/macOS, create and activate the environment:

    python3.10 -m venv venv
    source venv/bin/activate
    python -m pip install -r requirements.txt

If Python 3.10 is not your executable's name, substitute your installed Python 3.10-or-later executable. On later sessions, activate the existing environment again.

On Windows PowerShell, you can use the environment directly:

    py -3 -m venv venv
    .\venv\Scripts\python.exe --version
    .\venv\Scripts\python.exe -m pip install -r requirements.txt

Confirm the version is at least 3.10. For each later `python3` command, use `.\venv\Scripts\python.exe` on Windows. In addition, pass the Windows interpreter to `zipout.py`:

    .\venv\Scripts\python.exe zipout.py -x .\venv\Scripts\python.exe

`zipout.py` otherwise looks for the Unix path `venv/bin/python3`. Keep the environment and model checkpoints out of your Git commits.

## Background

The syntax of a natural language, similar to the syntax of a programming language involves
the arrangement of tokens into meaningful groups. Phrasal chunking is the task of finding 
non-recursive syntactic groups of words. For example, the sentence:

> He reckons the current account deficit will narrow to only # 1.8 billion in September .

can be divided into phrasal chunks as follows[^1]:

> [NP <span style="color: DarkBlue">He</span>] 
[VP <span style="color: BlueViolet">reckons</span>] 
[NP <span style="color: DarkBlue">the current account deficit</span>] 
[VP <span style="color: BlueViolet">will narrow</span>] 
[PP <span style="color: red">to</span>] 
[NP <span style="color: DarkBlue">only # 1.8 billion</span>] 
[PP <span style="color: red">in</span>] 
[NP <span style="color: DarkBlue">September</span>] .

[^1]: *Caveat*: If you have a linguistic background, you might find the verb phrases `VP` and prepositional phrases `PP` are different from what you might be used to. In this task, the `VP` is a verb and verb modifiers like auxiliaries (`were`) or modals (`might`), and the `PP` simply contains the preposition. This difference is because of the fact that the chunks are non-recursive (cannot contain other phrases) -- we need trees for full syntax.

## Data set

The training data has three columns separated by whitespace. The supplied development and test input files have two columns (word and part-of-speech tag); the development chunk labels are provided separately in `data/reference/dev.out`.
Each word has been put on a separate line and there is an empty
line after each sentence.

In the training data, the first column contains the current word, the second column is
the part-of-speech tag for that word, and the third column is
the chunk tag.

Here is an example of the file format:

    He        PRP  B-NP
    reckons   VBZ  B-VP
    the       DT   B-NP
    current   JJ   I-NP
    account   NN   I-NP
    deficit   NN   I-NP
    will      MD   B-VP
    narrow    VB   I-VP
    to        TO   B-PP
    only      RB   B-NP
    #         #    I-NP
    1.8       CD   I-NP
    billion   CD   I-NP
    in        IN   B-PP
    September NNP  B-NP
    .         .    O

The chunk tags contain the name of the chunk type, for example I-NP
for noun phrase words and I-VP for verb phrase words.  Most chunk
types have two types of chunk tags, B-CHUNK for the first word of
the chunk and I-CHUNK for each other word in the chunk. The full set of tags
for this task is in the file `data/tagset.txt`.

The sequence of labels, `B-NP`, ..., `I-NP` represents a single
phrasal chunk. For instance, the following sequence of labels:

    the       DT   B-NP
    current   JJ   I-NP
    account   NN   I-NP
    deficit   NN   I-NP

gives us the NP phrase:

> [NP <span style="color: DarkBlue">the current account deficit</span>] 

The O chunk tag is used for tokens which are not part of any chunk.

The data set comes from the Conference on Natural Language Learning:
[CoNLL 2000 shared task](http://www.cnts.ua.ac.be/conll2000/chunking/)[^2].

[^2]: [Introduction to the CoNLL-2000 Shared Task: Chunking](https://www.aclweb.org/anthology/W00-0726/)

There is a helpful program `count_sentences.py` which allows you
to count how many sentences are in a CoNLL formatted file.

This homework is not just about phrasal chunking but **robust**
phrasal chunking. The input data in dev and test files have been
infected with noise so the input to your chunker will look like
this:

    Rqckwell NNP
    , ,
    based VBN
    in IN
    El NNP
    Segundo NNP
    , ,
    Calief. NNP
    , ,
    is VBZ
    an DT
    aerospace NN
    , ,
    electronics NNS
    , ,
    automotive JJ
    and CC
    graphics NNS
    concern VBP
    . .

As you see the words have been infected with noise so
that it contains several spelling mistakes, e.g. `Rockwell` 
is now `Rqckwell`. The training data is clean, so these spelling
variations may be unfamiliar to the model. The subword tokenizer
can split a noisy word into pieces instead of treating every noisy
word as an unknown token.

The input files do not have the output chunk labels
which appear in `data/reference/dev.out` for input `data/input/dev.txt`.

## Data files

The data files provided are:

* `data/train.txt.gz` -- the training data used to train the `answer/default.py` model
* `data/input` -- input files `dev.txt` and `test.txt` infected with noise
* `data/reference/dev.out` -- the reference output for the `dev.txt` input file

## Default solution

The default solution is provided in `answer/default.py`. From `hw2`, copy it
as the starting point for your solution:

    cp answer/default.py answer/bertchunker.py
    cp answer/default.ipynb answer/bertchunker.ipynb

Keep any existing solution and notebook before copying. If you have a
checkpoint from an earlier starter, regenerate it with the corrected
starter before running `zipout.py`:

    python3 answer/default.py --force > output.txt

This trains a fresh baseline and replaces `data/chunker.pt` and the matching
epoch checkpoints. Back up any previous experiment checkpoints first.
On Windows, train without shell redirection, then collect and check the
baseline predictions with the supplied scripts:

    .\venv\Scripts\python.exe answer/default.py --force
    .\venv\Scripts\python.exe zipout.py -r default.py -x .\venv\Scripts\python.exe -o baseline-output -z baseline-output
    .\venv\Scripts\python.exe check.py -z baseline-output.zip

Check `baseline-output/dev.ret` and `baseline-output/test.ret` for `0`.
This avoids the different output encoding used by older PowerShell versions
for `>` redirection. The `output.txt` / `conlleval.py` examples below are for
Linux/macOS; use the archive-based commands above on Windows.

After changing your own model or training settings, use
`python3 answer/bertchunker.py --force > output.txt` to regenerate its checkpoint.
On Windows, use `.\venv\Scripts\python.exe answer/bertchunker.py --force`
without redirection, followed by the Windows `zipout.py` command above under
Getting Started. Keep your baseline and method results separately for comparison.
The short form of `--force` is `-f`.

Then generate and check the outputs:

    python3 zipout.py
    python3 check.py

The default solution will look for the file `chunker.pt`
in the data directory. If it does not find this file it
will start training on `data/train.txt.gz`. Training time depends on the device and configuration; allow substantially more time on a CPU. The program also downloads the base model and tokenizer when they are not cached.

The original [`chunker.pt` model
file](https://drive.google.com/file/d/1OY_NKo0vxS2ndttdfpqGyMo0PubbhIK3/view?usp=sharing)
was trained using an earlier starter. It is retained as a historical
reference; the corrected starter rejects legacy checkpoints that do not
record compliance with the frozen encoder policy. Train a fresh baseline
with the command above for your comparisons and report the measured results.

Please do not commit the file into your git repository as it is
moderately large and you can go over your disk quota. 

Once you have trained the corrected baseline at `data/chunker.pt`, you can run:

    python3 answer/default.py > output.txt

And then you can check the score on the dev output file called `output.txt` by running:

    python3 conlleval.py -o output.txt

For historical context, the earlier instructions reported the following detailed evaluation. This is a historical reference result, not a measured score for a newly trained model with a frozen encoder:

    processed 23663 tokens with 11896 phrases; found: 11847 phrases; correct: 10764.
    accuracy:  94.01%; (non-O)
    accuracy:  94.37%; precision:  90.86%; recall:  90.48%; FB1:  90.67
                 ADJP: precision:  79.29%; recall:  69.47%; FB1:  74.06  198
                 ADVP: precision:  74.25%; recall:  74.62%; FB1:  74.44  400
                CONJP: precision:  66.67%; recall:  85.71%; FB1:  75.00  9
                 INTJ: precision: 100.00%; recall: 100.00%; FB1: 100.00  1
                   NP: precision:  90.22%; recall:  91.57%; FB1:  90.89  6330
                   PP: precision:  96.80%; recall:  94.31%; FB1:  95.54  2378
                  PRT: precision:  80.56%; recall:  64.44%; FB1:  71.60  36
                 SBAR: precision:  92.27%; recall:  75.53%; FB1:  83.06  194
                   VP: precision:  90.48%; recall:  90.36%; FB1:  90.42  2301
    (90.85844517599392, 90.48419636852724, 90.67093459124794)

For this homework we will be scoring your solution based on the FB1 score
which is described in detail in the Check your performance section below. However the FB1
score is not the only focus. You can focus on efficiency, model size, 
experimental comparison with other approaches and many other choices.

Make sure that the command line options are kept as they are in
`answer/default.py`. You can add to them but you must not delete any
command line options that exist in `answer/default.py`.

Submitting the default solution without modification will get you
zero marks.

### The default model

The model uses the `distilbert-base-uncased` Transformer encoder and a
classification head that predicts a chunk tag for each subword token.
For this homework, keep the pretrained encoder weights frozen and train
the classification head (and any added components) using the ground-truth
chunk tags in `data/train.txt.gz`.

After training the corrected baseline at `data/chunker.pt`, inspect its model structure from the `hw2` directory:

    python3 answer/model.py

This prints out the model:

    TransformerModel(
      (encoder): DistilBertModel(
        (embeddings): Embeddings(
          (word_embeddings): Embedding(30522, 768, padding_idx=0)
          (position_embeddings): Embedding(512, 768)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (layer): ModuleList(
            (0-5): 6 x TransformerBlock(
              (attention): MultiHeadSelfAttention(
                (dropout): Dropout(p=0.1, inplace=False)
                (q_lin): Linear(in_features=768, out_features=768, bias=True)
                (k_lin): Linear(in_features=768, out_features=768, bias=True)
                (v_lin): Linear(in_features=768, out_features=768, bias=True)
                (out_lin): Linear(in_features=768, out_features=768, bias=True)
              )
              (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (ffn): FFN(
                (dropout): Dropout(p=0.1, inplace=False)
                (lin1): Linear(in_features=768, out_features=3072, bias=True)
                (lin2): Linear(in_features=3072, out_features=768, bias=True)
                (activation): GELUActivation()
              )
              (output_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            )
          )
        )
      )
      (classification_head): Linear(in_features=768, out_features=22, bias=True)
    )

Use PyTorch to optimize the trainable head and any added components on
the training data. The pretrained encoder parameters must remain frozen.

### Hyperparameters

Use `distilbert-base-uncased` as the fixed base model. Freeze all of its
pretrained encoder parameters throughout training; do not replace the
base model or update its weights. You may change and train the layers
on top of the encoder, including the classification head and any new
components you add.

You may tune training hyperparameters, including the learning rate,
number of epochs, and batch size. You may also tune hyperparameters of
your added components. Record your settings and comparison results in
`answer/bertchunker.ipynb`.

### Pytorch

You will need to use some Pytorch API calls to solve this homework.
We do not expect you to already know Pytorch in great detail.
The following links will help you get started but you can learn
a lot of the Pytorch basics by understanding `default.py` and
the process of solving this homework.

Some useful links if you feel lost at the beginning:

* [60 mins intro to Pytorch](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
* [Introduction to the transformers library](https://huggingface.co/docs/transformers/notebooks)

Read the source code in `answer/default.py` in detail.

## The Challenge

Your task is to _improve robust phrasal chunking while keeping the
`distilbert-base-uncased` encoder fixed and frozen_. You may tune training
hyperparameters and train or improve the layers on top, as described above.
The score is explained in detail in the Check your performance
section below. With substantial computational resources and using
large pre-trained models (which are beyond the scope of this homework)
the [state of the art accuracy on this
dataset](https://nlpprogress.com/english/shallow_syntax.html) has
reached an F1-score above 97 percent (typically by using more 
information than just the training data available for this
task).

However, the numbers from that leaderboard do not apply to the dev
and test data in this homework. The dev and test data used here are
much more challenging than the standard CoNLL 2000 chunking task
because several typos have been introduced into the data (as explained
above).

## Improving the model

Here are some specific things you can try to improve the accuracy
of the fine-tuned model:

1. Deal with misspellings in the dev and test data using adversarial training (more details below).
1. Use more than the last layer of the Transformer since lower layers of a pre-trained LLM tend to reflect "syntax" while higher levels tend to reflect "semantics" (waving hands profusely).
1. Experiment with optimizers and learning rates for the trainable classification head or other added components. Keep the pretrained encoder weights frozen.
1. Improve the classification head using either:
    1. multi-layer perceptron (MLP)
    1. CRF
    1. mini-Transformer.

Choose one or two improvement ideas for your model; these may include
MLP or CRF. Separately, your notebook must compare the fixed set of
baseline (linear head), MLP, and CRF models, plus any other methods you
attempt. Use the same fixed, frozen base encoder for these comparisons.
Report the dev scores you actually obtain; no improvement idea guarantees
a particular score.

### Dealing with Misspellings

A very simple idea for dealing with the misspellings in the dev and
test data is to realize that the training data is not similarly
noisy. Augmenting the training data with additional noisy examples
can help the model handle misspellings at inference time.

For more advanced approach, look into adversarial training as
explained in the following paper:

> [Combating Adversarial Misspellings with Robust Word Recognition](https://www.aclweb.org/anthology/P19-1561/). Danish Pruthi, Bhuwan Dhingra, Zachary C. Lipton. ACL 2019.

### Improving the classification head

The classification head in `default.py` is a single linear layer. You could experiment with more expressive feed-forward networks like the following model. 

        FFN(
            (dropout): Dropout(p=0.1, inplace=False)
            (lin1): Linear(in_features=768, out_features=3072, bias=True)
            (lin2): Linear(in_features=3072, out_features=768, bias=True)
            (classification_head): Linear(in_features=768, out_features=22, bias=True)
        )

You can experiment with `lin1` having more parameters than the input or fewer parameters.

### CRF Layer

A linear-chain Conditional Random Field (CRF) scores a complete label
sequence using emission scores from the classification head and learned
transition scores between labels.

For a sentence of length $$n$$, let $$E_{i,k}$$ be the emission score for
tag $$k$$ at token $$i$$ and $$A_{j,k}$$ a transition score from tag $$j$$
to tag $$k$$. With separate start and end scores, define

<p>$$s(\mathbf{x},\mathbf{y}) = A_{\mathrm{START},y_1}+\sum_{i=1}^{n}E_{i,y_i}+\sum_{i=2}^{n}A_{y_{i-1},y_i}+A_{y_n,\mathrm{END}}.$$</p>

The conditional log probability is

<p>$$\log p(\mathbf{y}\mid\mathbf{x})=s(\mathbf{x},\mathbf{y})-\log\sum_{\mathbf{y}'\in\mathcal{Y}(\mathbf{x})}\exp s(\mathbf{x},\mathbf{y}').$$</p>

Here, $$\mathcal{Y}(\mathbf{x})$$ is the set of candidate label sequences
for the input. The normalization ranges over label sequences, not input
tokens. Train by minimizing the negative log probability of the gold
sequence, using the forward algorithm to compute the normalization, and
decode the highest-scoring sequence with Viterbi. Independent per-token
`argmax` is not CRF decoding. Transition and emission values are scores,
not probabilities to add after separate softmax operations.

In the starter, `tag_space` has shape `[batch, length, number_of_tags]`;
the tag dimension is `size(-1)` (or `size(2)`), not `size(3)`. Keep padding
out of the sequence score and loss with a valid-token mask. Use explicit
start/end states rather than connecting the first token to the last.
Preserve the mapping between subwords and output words. BIO constraints,
if imposed, should allow both `B-X` followed by `I-X` and `I-X` followed
by `I-X` within a span.

A reference for neural sequence labeling with a CRF layer is:

> [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360). Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami, Chris Dyer. NAACL 2016.

Based on the original CRF paper (which you don't really need to dive into but linked here for completeness):

> [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://dl.acm.org/doi/10.5555/645530.655813). John Lafferty, Andrew McCallum, Fernando Pereira. ICML 2001.

Implementing a CRF layer is a useful PyTorch exercise. Compare its measured
performance and errors against the baseline and MLP models in your notebook.
You may also investigate a head with self-attention as an additional method.

## Required files

You must create the following files:

* `answer/bertchunker.py` -- your solution, starting from `answer/default.py` as explained above.
* `answer/bertchunker.ipynb` -- your notebook report, including documentation and analysis. No separate `report.pdf` is required for HW2-P.

## Run your solution on the data files

To create the `output.zip` file for upload to {{ site.hwsubmit.name }} do:

    python3 zipout.py

For more options:

    python3 zipout.py -h

## Check your performance

To check your performance on the dev set:

    python3 check.py

The output score is the $F_{\beta=1}$ score or [FB1 score](https://en.wikipedia.org/wiki/F1_score)
which is the harmonic mean of the precision and recall
computed over all the output phrasal chunks.

    python3 check.py -h

In particular use the log file to check your output evaluation:

    python3 check.py -l log

The performance on `data/input/test.txt` will not be shown.  We will
evaluate your output on the test input after the submission deadline.


## Preparing your python notebook to clearly document what you have done

Prepare a clear summary of what you did in `answer/bertchunker.ipynb`, which serves as your report. Organize the documentation and analysis into clear sections, with full sentences. Use figures, graphs, and tables to compare experimental results.

The documentation and analysis in your python notebook should include the following:
<!-- * Group name with names of group members -->
* A short **description of the task** (e.g. the problem you are solving) in your own words.  Make sure to indicate what is the expected input and output.   
* A short description of your method, the layers you trained, and the hyperparameter settings used in each experiment.
* Results (both quantitative and qualitative) comparing the required models and your method. Include illustrative examples of their predictions and errors. Use tables or figures to compare chunk-level precision, recall, and FB1 scores on the dev set.
* Discussion of alternative methods you tried and how well they worked (or didn't work).  
<!-- * Breakdown of contributions by each group member -->

Compare all three required models: the baseline with a linear classification head, an MLP head, and a CRF model. Include any other methods you attempted. Keep the base encoder fixed and frozen in all comparisons, document differences in training settings, and discuss which approach worked best and what errors the approaches made. Your one or two chosen improvement ideas may use MLP or CRF, but do not replace the required comparison set.

## Submit your homework on {{ site.hwsubmit.name }}

Once you are done with your homework submit all the relevant materials
to {{ site.hwsubmit.name }} for evaluation.

### Create output.zip

Once you have a working solution in `answer/bertchunker.py` create
the `output.zip` for upload to Coursys using:

    python3 zipout.py

Before uploading, check that `output/dev.ret` and `output/test.ret` both contain `0`. Inspect `output/dev.err` and `output/test.err` if a run failed. The presence of `output.zip` alone does not establish that both runs succeeded.

### Create source.zip

To create the `source.zip` file for upload to {{ site.hwsubmit.name }} do:

    python3 zipsrc.py

You must have the following files or `zipsrc.py` will complain about it:

* `answer/bertchunker.py` -- your solution, starting from `answer/default.py` as explained above.
* `answer/bertchunker.ipynb` -- your notebook report, including documentation and analysis.

In addition, each group member should write a short description of their contribution in `answer/README.username`, replacing `username` with their own SFU ID. `zipsrc.py` checks for the Python and notebook filenames but does not check that every member has a contribution file; check these yourself.

`source.zip` contains the contents of `answer/` at its archive root. Include any helper Python files your solution imports and an `answer/requirements.txt` if your dependencies differ from the starter. Inspect the archive before uploading to ensure these files and all contribution statements are included.

### Upload to {{ site.hwsubmit.name }}

Go to `Programming Homework 2` on {{ site.hwsubmit.name }} and do a group submission:

* Upload `output.zip` and `source.zip` to [{{ site.hwsubmit.name }}]({{ site.hwsubmit.url }}/+hwp2/)
* Make sure you have documented your approach and analysis in `answer/bertchunker.ipynb` (which serves as your report).
* Check if each group member has a `answer/README.username`.
* Update your group GitHub repository with the same source and notebook files that you submit to Coursys. After uploading, verify that the submission lists both archives and that your group is correct.

## Grading

The grading is split up into the following components:

* dev scores (see Table below)
* test scores (see Table below)
* Documentation and analysis quality in the notebook
* Code content and quality
   * Make sure that you are not using any external data sources in your solution.
   * Make sure you have implemented the fine-tuning model improvements yourself without using external libraries.
* Check if each group member has a `answer/README.username`.

Your F-score should be equal to or greater than the score listed for the corresponding marks.

| **Score(dev)** | **Score(test)** | **Marks** | **Grade** |
| Nan  | Nan  | 0   | F  |
| 90.5 | 82   | 55  | D  |
| 91   | 83   | 60  | C- |
| 91.5 | 84   | 65  | C  |
| 92   | 85   | 70  | C+ |
| 92.5 | 86   | 75  | B- |
| 93   | 87   | 80  | B  |
| 93.5 | 88   | 85  | B+ |
| 94   | 90   | 90  | A- |
| 94.5 | 92   | 95  | A  |
| 96   | 95   | 100 | A+ |
{: .table}

The score will be normalized to the marks on {{ site.hwsubmit.name }} for the dev and test scores.

