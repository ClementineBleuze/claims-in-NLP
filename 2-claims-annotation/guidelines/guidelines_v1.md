# Guidelines for claims classification

In this project, your task is to **label sentences** according to the **type of claims they contain**. Some sentences **do not contain any**, so you must assign them the label <span class="colour" style="color:darkred">*no claim* (NC). Otherwise, whenever at least one claim is identified, it must be categorized according to these four categories (which will be defined just below):

* <span class="colour" style="color:green">*positive* </span>claims <span class="colour" style="color:green">(POS)</span>
* <span class="colour" style="color:red">*negative* </span>claims <span class="colour" style="color:red">(NEG)</span>
* <span class="colour" style="color:blue">*factual* </span>claims <span class="colour" style="color:blue">(FACT)</span>
* <span class="colour" style="color:magenta">*prospective* </span>claims <span class="colour" style="color:magenta">(PROSP)</span>

> You are allowed to use **multiple labels**, if and only if the sentence actually contains **multiple claims** (e.g two clauses separated by "and") from different categories. If you identify **one claim** but hesitate on its category, please choose only one that seems the most appropriate (see **Ambiguous cases** at the very end below for examples)

As a **general methodology**, we recommend that you:

1. Decompose the sentence to annotate into simpler clauses (if relevant)
2. For each clause, decide if it is a claim or not. If no claims are found in the sentence, use the label <span class="colour" style="color:darkred">(NC)
3. Else, for each claim, identify its category and assign the corresponding label to the sentence.
4. If you hesitate, please refer to these guidelines (definition of a claim, categories, and ambiguous cases). Also note that **you have access to the previous and next sentence** (in the paper's full text) **of the one you are annotating** as well as the **name of the paper section** from which it was extracted in the metadata pannel (bottom right) of the annotation page, which can allow you to better contextualize it.

## What is / isn't a claim

A **claim** is a statement (at the phrase-level) found in a sentence. Independently of its category, it must meet following requirements:

1. It is a **statement** emitted by the paper's authors **on the basis of their work / findings / reflexion** and NOT on previous knowledge or related works

> <span class="colour" style="color:green">**claim:** *we [prove/ suggest/ find] that Y*
> <span class="colour" style="color:red">**not a claim:** *In work X, authors [prove/ suggest/ find] that Y*
> <span class="colour" style="color:red">**not a claim:** *[It is assumed in general that] Y*

2. It captures a **valuable result, contribution or conclusion** that the authors put forward as bringing a **new knowledge or tool** to the community. More precisely, it can be:

* an explicit statement of a **contribution**
* an **evaluation result**, a **result analysis**, a **conclusion**, an **observation**
* a **reflexion** about the work itself, a **prediction** or **anticipation** of what it can bring to the community

> <span class="colour" style="color:green">**claim:** *we successfully trained a LLM to perform multilingual translation using few-shot learning*

But it is NOT:

* an introduction to the field, a background statement, an in-depth explanation
* a technical description, a methodology account, data or model details
* an outline of the structure of the paper, a caption to a figure or table

> <span class="colour" style="color:red">**not a claim:** *extractive summarization techniques extract key sentences from a text to produce a summary*
> <span class="colour" style="color:red">**not a claim:** *we trained for 50 epochs and used a learning rate of 0.001*
> <span class="colour" style="color:red">**not a claim:** *we used dataset X from source Y*
> <span class="colour" style="color:red">**not a claim:** *Table 1 shows our main results*

3. It implies (more or less of) the **subjectivity** of the authors, so it can be presented as only possible or plausible (*this could indicate that Y*, *we hypothesize that Y*)
4. It must fit in **one or more** of the following categories. If it doesn't, it shouldn't considered as a claim.

**Please note that if a sentence contains a claim BUT ALSO statements that are not claims, it should be labelled with the category of the identified claim. The <span class="colour" style="color:darkred">*no claim* (NC) category is reserved to sentences containing 0 claim.**

## Claim categories

### <span class="colour" style="color:green">1. *Positive* claims <span class="colour" style="color:green">(POS)

These are claims announcing **main results, findings** and **analyses / conclusions** that derive directly from them, or **original working hypotheses** proposed by the authors as the basis of their work. They are "positive" in the sense that they contribute to the establishment of **new knowledge** for the scientific community. They anwer to the question: *What do the authors [maybe] show / establish ?*

> "it shows that our system achieves the performance of 84.8 % / 66.7 % / 74.7 in precision / recall / fmeasure on relation detection ."
> "Our findings highlight the importance of equipping dialogue systems with the ability to assess their own uncertainty and exploit in interaction."

### <span class="colour" style="color:red">2. *Negative* claims <span class="colour" style="color:red">(NEG)

These are claims by which the authors **acknowledge some [potential] limitations** of their work or findings (often in order to nuance some <span class="colour" style="color:green">*positive* claims, hence the "negative" label). They answer to the question: *What [are / could be] some limitations of the authors' [work / findings] ?*

> "the results do not necessarily apply to other encoder-decoder models or autoregressive models such as GPT series”

### <span class="colour" style="color:blue">3. *Factual* claims <span class="colour" style="color:blue">(FACT)

These are claims by which the authors announce the **nature of their contributions** in terms of what they have **actually realised or produced** (a model, a survey, a corpus, a method, etc.). They answer to the question: *What kind of contribution did the authors make in this work ?*

> "We present the first challenge set and evaluation protocol for the analysis of gender bias in machine translation (MT)"
> "We propose the novel task of automatic source sentence detection and create SourceSum [...]"

### <span class="colour" style="color:magenta">4. *Prospective* claims <span class="colour" style="color:magenta">(PROSP)

These are claims that **anticipate** possible **consequences / impact** of the presented work or **suggestions** of **future continuations / directions**. They often imply a higher degree of **subjectivity from the authors** than the other categories of claims. They answer to the question: *What [will / could] this work [become / provoke / evolve into]?*

> "The proposed method may be an important module for future applications related to time ."
> "We believe the isarcasm dataset , with its novel method of sampling sarcasm as intend by its author , shall revolutionise research in sarcasm detection in the future"

## Ambiguous cases

### Multiple labels

Multiple labels are to be used only when **a sentence actually contains more than one claim**. In the example below, two claims are identified: a <span class="colour" style="color:blue">*factual* claim <span class="colour" style="color:blue">(FACT) and a <span class="colour" style="color:green">*positive* claim <span class="colour" style="color:green">(POS). So, we should assign **both labels** <span class="colour" style="color:blue">(FACT) and <span class="colour" style="color:green">(POS).

> <span class="colour" style="color:blue">*We created a new model for task X* *and <span class="colour" style="color:green">achieved an accuracy of Z on dataset Y.*

Multiple labels shouldn't be used in case of an **hesitation on the category to choose** for **one single claim**: then, please read carefully the guidelines and following ambiguous cases to decide and choose the most appropriate labe.

### Wrong sentence segmentation

In case of a poor sentence segmentation of the paper, you may encounter "sentences" that are actually incomplete because they were split, e.g

> (1) *We created a corpus based on the work of (X et al.*
> (2) *2020) and introduce a novel method for MT*

If you find that the overall sentence contains some claims, **please annotate all the sub-sentences accordingly** in the same manner. If you encounter sub-sentences but do not find their other parts in the surrounding documents to be annotated, and **you can't make sense of them**, please assign them the <span class="colour" style="color:darkred">(NC) label.

### Other ambiguous cases

To be completed.
