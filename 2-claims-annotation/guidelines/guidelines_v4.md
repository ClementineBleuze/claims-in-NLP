# Annotation of sentences in NLP research papers
## Overview
What you need to annotate:

* the **abstract**<span style="color:darkblue">*</span>
* the **introduction**<span style="color:darkblue">*</span> 
* parts where **main results** are presented and discussed (**analysis, interpretation, discussion, etc.**) 
* parts where **limitations, ethical statements, future directions** are discussed 
* the **conclusion**<span style="color:darkblue">*</span> 

<span style="color:darkblue">*Please note that labels **context-AIC**, **outline-AIC** and **contribution-AIC** are to be considered only in the **Abstract**, **Introduction** and **Concluding part(s) (please note that for some papers, the concluding part may be *Discussion* instead of *Conclusion*)** . You do not need to pay attention to these three labels when annotating result sections</span>.

>  If you encounter sentences which have been **poorly segmented** (e.g split in the middle of a reference), or **bad PDF-XML conversion clues** that are disturbing for a good comprehension (e.g footnote text being inserted in the middle of a paragraph), please use the additional tag <span style="color:red">**error**</span> (only when the sentence was relevant to the annotation, no need to spot all errors !).

Please ignore:

* **methodology, experimental setup** parts
* **data presentation**, **model details**
* **related works** (if separate from the introduction)

## The labels

### <span style="color:darkred">**0. context-AIC**</span>

 <span style="color:darkblue">[Only in Abstract, Introduction or Conclusion]</span>
 Sentences providing **context** / **explanations** to the reader on the discussed task/issue. Typically: to provide background in Abstract/Introduction, or to widen the discussion in Conclusion. Can contain references to related works, in which case, please also add the <span style="color:cyan">rw</span> tag.

### <span style="color:darkred">**1. outline-AIC**</span>

<span style="color:darkblue">[Only in Abstract, Introduction or Conclusion]</span>
Sentences presenting the **outline** of the paper (*In Section n, we discuss X*, *We will conclude this paper with a discussion on X*). Can at the same time contain other types of claims, e.g <span style="color:blue">contribution</span> or <span style="color:green">result</span> (*In Section n, we prove that [result]*).

###  <span style="color:blue">**2. contribution-AIC**</span> 

<span style="color:darkblue">[Only in Abstract, Introduction or Conclusion]</span>

Description of the main **contributions** of the work, that is, **everything** in Abstract, Introduction or Conclusion, that has to do with:
- **the final outcome** (<span style="color:darkblue">*What has been done? What does this work bring ?*</span> *A model, a survey, experiments ...*).
- **its function / objective / research questions it provides answer to** (<span style="color:darkblue">*What for ?*</span> *For topic labelling, Our objective is X ...*)
- **main features and qualities of the work** (<span style="color:darkblue">*Why is this valuable ?*</span>*It's the first study on [subject]*, *We provide the code at [adress]*)
- **additionnal operating details / methodology adopted** (<span style="color:darkblue">*How does it work ? How did the authors do ?*</span> *It uses a Transformer architecture, We annotated data from corpus X, ...*) 

This does NOT include **justifications** of methodology choices (*We adopted this architecture **because** it has property X *)
###  <span style="color:green">**3. result**</span>

Any kind of **result** reported by the authors:
- **experimental**: evaluation, performance assessment, measure, observation...
- **non-experimental**: arguments, new knowledge, recommendation ...

Also **interpretations** and **discussions** about these results:
- explanations or opinion about a result
- comparison with results from other works


### <span style="color:red">**4. limitation**</span>

Declaration or description of a **limitation** of the present work<span style="color:darkblue">* </span> : 
- difficulties encountered (*Our budget was not sufficient to gather more data*)
-  limitations on the interpretation of some results, anticipation of contexts in which the results/performance could change (*The accuracy may vary on less-specialized datasets*, *We only experimented with data in english*)
- undesirable behaviours, things that do not function as expected (or as good as expected) (*The model produces hallucinations*)

<span style="color:darkblue">* See *Ambigous cases* for a discussion on the difference between <span style="color:red">limitation</span> and <span style="color:magenta">direction</span>.</span>
### <span style="color:magenta">**5. impact**</span>

Statements about the **impact** of the work on a group of people or on an area of research:
- the impact can be positive, negative or neutral 
- impacted people can be certain groups, people who participated in the work, the scientific community, the society as a whole ... or the impact can be on an area of research

This includes:
- statements which focus on the **importance / interest** of the work itself, or of the domain it belongs to (ie *What effect will it have on to the community ?*)
- some **ethical considerations**  (*We declare that our work raises no ethical issues*, *We ensured that our annotators were given appropriate working conditions*), although not all sentences in *Ethics statements* part do necessarily belong to this category (e.g you could encounter some <span style="color:red">limitation</span>, or some sentences without a label)

### <span style="color:magenta">**6. directions**</span>

Discussions about **concrete future directions** for this work<span style="color:darkblue">* </span>(either suggested, planned, under development, considered, etc.). 
<span style="color:darkblue">* See *Ambigous cases* for a discussion on the difference between <span style="color:red">limitation</span> and <span style="color:magenta">direction</span>.</span>

## The additionnal tags
### <span style="color:cyan">**7. rw**</span>

Explicit references (work is cited / a phrase like *Recent works* is used) to findings or contributions of **related works**, <span style="color:red">to be used only as an additional tag when combined with another label (e.g <span style="color:darkred">background</span>, <span style="color:green">discussion</span>, etc.)</span>

###  <span style="color:#72544E">**8. error**</span>
For sentences which have been **poorly segmented** (e.g split in the middle of a reference), or which contin **bad PDF-XML conversion clues** that are disturbing for a good comprehension (e.g some footnote text is inserted in the middle of a sentence, or a number alone constitutes a sentence).

## Ambiguous cases
In general, keep in mind that what we annotate is not **our interpretation of the sentences**, but **our perception of how the authors wrote them**.
###  <span style="color:blue">contribution-AIC</span> vs.  <span style="color:green">result</span> in technical papers
Especially in papers where authors present a system they have created, it can be confusing to make a difference between <span style="color:blue">contribution-AIC</span> and <span style="color:green">result</span>, because both can talk about system features. Please consider following sentences:

1. *We created an innovative and easy-to-use system which translates english poetry to german.*
2. *Our innovative, easy-to-use system was able to translate english poetry to german*.

They both talk about the same elements (an innovative and easy-to-use model, an english to german poetry translation feature), but **they present these elements in different manners**. We are actually interested in knowing whether the authors **wrote it like** a contribution, or a result. Please keep in mind that:
- <span style="color:blue">contribution-AIC</span> sentences have a function of **presentation** of the conducted work (outcome nature, features, essential details). This is in general quite **factual**, but there can be subjectivity in the choice of words, and qualifiers in particular (*a model* or *an **efficient** model* ?). This is what we have in (1): the sentence emphasises the type of work conducted (*an innovative and easy-to-use system*) and its function, ie *what does it do ?* (*[it]  translates english poetry to german*). When we read it, we don't understand that it was successful in this task in a particular experimental context, but understand that it translates english poetry to german *in general*. Maybe there was actually an experiment, but this is not self-evident when reading the sentence, so we understand it in the general meaning.

- <span style="color:green">result</span> sentences emphasise on **what the system did / how it performed during an experiment or test phase** which corresponds more to sentence (2), because of ***was able to** translate [...]*. In this sentence, we understand that the authors report a performance established in a particular testing context. 

So, finally:
1. *We created an innovative and easy-to-use system [outcome + subjectivity] which translates english poetry to german [system function + subjectivity].* -->  <span style="color:blue">contribution-AIC</span>
2. *Our innovative, easy-to-use systemwas able to translate english poetry to german [experimental performance assessment]*. --> <span style="color:green">result</span>

###  <span style="color:red">limitation</span> vs.  <span style="color:magenta">direction</span>
Let's consider following sentences:

1.*The precision of the model should still be improved.*
2. *In future work, we should focus on improving the model's precision.*
3. *The issue of low model precision is left as future work.*

All of them talk about the same elements (an unsatisfying model precision, needed improvement), but **they present these elements in different manners**. We are actually interested in knowing whether the authors **wrote it like** a limitation, or a direction, or both:

* In (1), the emphasis is on the declaration of a <span style="color:red">limitation</span> of the model. The phrase *should still be improved* is actually a paraphrase for *is not satisfying*, but does not express a concrete research direction. (1) roughly says *There is a problem with the model's precision*. 
* In (2), the emphasis is on the research <span style="color:magenta">direction</span>. Even if we can infer that a need to improve the precision means that it is not satisfying, this is not the way that it is expressed directly in the text. (2) roughly says *We propose a future direction (improving model's precision)*
* In (3), the emphasis is both on the declaration of a <span style="color:red">limitation</span>  (*issue of low model precision*) and on the proposition of a research <span style="color:magenta">direction</span>. (3) roughly says *There is a problem with the model's precision* AND *We propose a future direction (improving model's precision)*

So finally:

1.*The precision of the model should still be improved.* -->  <span style="color:red">limitation</span>
2. *In future work, we should focus on improving the model's precision.* -->  <span style="color:magenta">direction</span>
3. *The issue of low model precision is left as future work.*  -->  <span style="color:red">limitation</span> +  <span style="color:magenta">direction</span>