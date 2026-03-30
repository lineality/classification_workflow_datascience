#### classification_workflow_datascience
# Understanding Classification Workflow in Data Science
A Non-Technical Overview of Traditional Known-Class
& Future "Classification" Modeling
##### 2025.05-09 G.G.Ashbrook

## Contents:
1. Navigating Jargon & Pragmatic Definitions of 'AI'
2. Outlining 'Classification' as 'known-class classification' in the jargon meaning
3. 'Discovery'/'identification': Possible new types of technology for things similar to common-language 'classification'

Here the focus will be "classification" tasks using Machine-Learning technologies (or Artificial Intelligence, AI). The examples involve Natural Language Processing (NLP), but machine learning can use any inputs.

The goal here is to explain what 'classification modeling' in data-science is in a straightforward way, to help you to identify whether a need-and-goal is a classification task or something else. Part of this will be understanding how 'classification' is defined in data science and machine learning, which may differ from how the term 'classify' may be used more broadly in everyday language.


# 1. Navigating Jargon & Pragmatic Definitions of 'AI':

Keep in mind that many terms that might sound like casual-language may be technical jargon terms with meanings different from casual-language. Terms such as 'descriptive' vs. 'predictive' modeling, or 'explain/explanation/explainable' or 'generalize' or 'parameter' or 'complex/complexity', or 'classification' may be undefined or strangely defined in ways very different from common language meanings. Here we will be going into some of the details of what 'classification' can mean technically.

"Classification" as in 'training a classification model' usually is a very specific jargon reference to specific workflows and technologies that will be described below. But there is also a more general meaning of 'classification tasks' that may be related to deep learning technologies that (as of 2025) are still young and not fully understood. So there may be two very different questions involved in what might seem like one question: "Is this [something we need to do] a classification task?" 

The term AI does not need to be vague and arm-wavy, though sometimes the term is used without specifying what is meant. 'AI' can be defined clearly enough in a few standard ways:
1. The 1956 (original) definition: Automating a task that is previously, or currently, done only by a human-person.

2. Synonymous with STEM, as a general collection all interlocking Science, Statistics, Technology, Engineering, Maths, Medicine, etc. areas. 

3. A specific, or very specific, technology or set of technologies, such as decision trees (such as XG-Boost trees) or 'statistical learning' more broadly, regression or logistic regression modeling, "classification modeling", artificial neural networks and deep learning, or specifically Generative Foundation Models, or Transformer models, or GPT models (like 'Chat GPT'). Any of these specific (or very specific) technologies could be called 'AI.'


# 2. Outlining 'Classification' as 'known-class classification' in the jargon meaning

Machine Learning has to do with automating what you already know how to do and have done many, many, times. If you cannot define a task, and or if you cannot do that task, then 'machine learning' or 'ai' (in this context) are the solution you are looking for. 

## Classic Example: Sorting Departmental Mail (a stock example in data science)
For traditional statistical-learning, we can think of Data-Science 'classification' as being like assigning a newly hired assistant to read and route incoming customer emails to the right department in a company (or a municipal institution). There are a few things we need to prepare first in order for our new-hire-task to be a success at their mail-sorting job.

Before this new-hire can really be effective at sorting, we will need:

1. Known-Classes: 
A clear selection of which departments you care about (these categories are your "classes", which is an important point in the traditional definition of the term "classification" ('class'-ification)). Without known-classes, the process is not 'classification.'

2. Labeled Examples 
You will need examples (a specific strict type of examples) of what needs to go to each department (your "labeled" "training data"), many examples (and explanations would be helpful too, for communication purposes). This, saying that something is an example of a "class" to be sorted into, is called a 'label.' A data-table of examples including the 'labels' for how they should be sorted is called the labeled training data set (or just 'training data'), whether you are training a person or training an automated system. A human-person might get a clear idea of their task from a small number of solid examples (maybe in the order of ten examples), but machine learning (or statistical learning, or AI) should have as many examples as possible, ideally millions, thousands are probably good, hundreds may be enough. Getting and curating your labeled examples is traditionally 90% of the time and effort of the whole project.

The training data are sometimes called either testing/training data or even testing-training-validation data (depending on the workflow details) as these data are both used to train the 'AI' and used to test that the system is really working after it has been trained.

To recap, if you want emails about billing-issues to go to Finance, product-questions to go to Support, and partnership-inquiries to go to Business Development, you will need to:

1. Tell the assistant that these (Finance, Support, and Business Development) are the three departments that handle relevant emails. [Your 'classes']

2. Show the assistant many examples of which emails go to which department.

Only if these steps are done effectively can the assistant correctly route new emails.

The assistant (your classification model) does not decide which departments should exist or what types of emails matter to each department, or whether they should be routing emails or writing emails. The assistant only "learns" to associate examples (or generalized classes of examples) with the labels you provide. 

Clear Labels: Data labels must be clear and exclusive. Arbitrary labels that could be one thing or could be something else will not work.

Terminology note about Testing a Model:
"Generalization" (among a variety of meanings) can mean that the assistant can 'generalize' what they learned from studying the examples to new data (new emails in this case) that they have not seen before. The more labeled-examples you have, the better the chance that the assistant has at being able to learn a process that generalizes well to enough new data. If the model does well with training data but not on testing data it has not seen before, then it is said that the model cannot 'generalize' its understanding to the test-set (and often said that the model is 'over-fit' on the training data that it describes well but cannot deviate from). 

## Common Misconceptions
People who have not used machine-learning solutions and workflow may think, "I will just feed all this raw data into a black-box AI and somehow the AI will figure out what the categories should be and then which categories everything belongs to." This does not match the process of traditional classification. A computer (not unlike a new untrained person who knows nothing about your needs) cannot know the needed details about your needs and goals unless you:

1. Explicitly define the categories (or 'classes')
2. Provide many labeled examples of what goes into each category (class).


If someone says, "Just use the algorithm to classify our documents," but they cannot tell you:
1. What specific categories they want to classify into.
2. Where to find at least 500-5,000+ labeled examples of each category

then they are asking for something that is not compatible with the definition of a known-class "classification" process, no matter how much emphasis, enthusiasm, or coercion may be applied.

If someone has data (perhaps data that they have not looked at yet) and wants to understand what it is so that in the future they can form clear classification targets, then they do have real tasks to do (identifying, discovering, analyzing, and understanding the data) but these are not known-class 'classification' tasks.


## Different Types of "Models," Architectures, For Traditional Classification

Not all 'machine learning' methods have exactly the same workflow, for example there are some methods that are called "unsupervised learning." It is possible (if unlikely in this email NLP example) that you would "train" your unsupervised model by adjustments (rather than the statistics of training examples). However a given tool works, you still need to be able to build it to work for known-classes and verify that it does work on correctly labeled train/test dataset. You still need defined targets and a labeled testing/training dataset.


If you seek out a more technical guide to data science and machine learning it will probably cover a common spectrum of 'types' of 'models':
- Regression
- Logistic Probability
- Decision Trees
- Support Vector Machines (SVM)
- Bayesian
- Latent Dirichlet Allocation (LDA/LDiA)
- Clustering (unsupervised)
- Dimensionality-Reduction (unsupervised)
- Deep-Learning/Artificial Neural Networks (before 2023)
and some less common areas such as genetic algorithms (note: deep learning was 'uncommon' before 2012). 

The details of exactly how to structure the data and 'train' and 'test' vary but the overall workflow is the same:
1. pick your target classes
2. make a huge labeled dataset
3. train and test
4. 'predict' the known 'class' (predict y based on X)

Especially in Natural Language Processing (NLP), whole builds, applications, and pipelines, usually combine many tools among which are a mix of older and newer technologies (the good old tech says good). This can be confusing where the term 'model' might be used by different people (at different times) to refer to quite a variety of completely different things: one piece, the whole, one calculation, one variable, etc. Sometimes you will have a single 'end to end' 'model,' at other times you will have myriad steps and pieces in spread-out workflows and pipelines in a massive diverse architecture.


# 3. 'Discovery'/'identification': Possible new types of technology for things similar to common-language 'classification'

The ability to open-endedly generate language output from a general-foundation model is relatively new to ~2023 technology. Books written up to 2023 (such as Melanie Mitchell, Michael Wooldridge, Francois Chollet, see below for specifics) went out of their way to explain that all AI models are simple tools for a specific single purpose, totally lacking any general world knowledge such as people keep wanting to imagine the 'AI' to have. That somewhat changed around 2023 (it is not yet clear exactly how), but for our purposes here it means that models can now answer much more general questions than before, such as broadly what topic an article is about, not merely narrowly as trained only for specific options. To a non-technical user this may seem like no big deal, but this is a big technical difference. For example, while it is not traditional technically-defined 'known-class- classification' you can now build a system to perform 'unknown-class' identification/discovery.


## Timeline: From Traditional Classification to Possible Generative Foundation Models and Ensemble-System-Architectures:

To understand how ~'classification' might be approached in slightly different ways after 2023, which could possibly change the future use of the term 'classification, let's look at a technology timeline:

1. 1956-1996, GOFAI: Good Old Fashioned AI (MYCIN & ELIZA)
2. 1996-2012, Statistical Learning (From 'ESL' and 'ISL')
3. 2012-2022, Deep Learning: (From Imagenet & Hinton)
4. 2023 -> ~,  General Foundation Models (From ChatGPT)


1. human-authored rules-based AI
In the first epoch of AI, classification systems were manually created by people: human 'experts' would program an automated 'expert system' based on decisions at meetings. MYCIN is a classic example of a blood-test classification system designed to automate and standardize how expert medical technicians assign classes and categories based on data from test results. This AI did not 'learn' on its own in any way.

2. statistical-rules based AI
Maturing in the 1990's people started automating the use of statistical calculations instead of jumping ahead to the final conclusions of human experts. Experts still needed to pick the classes, label the examples, and fine tune the system, but in this epoch the AI 'model' would 'learn' by doing a statistical analysis of the examples and classes, and/or by making 'decision trees' to arrive at a result.
- linear models & polynomial models
- decision trees

Note: Linear 'parametric' models and decision trees made of rules and features are also two ways that people have become accustomed to saying that a process is 'explained.' But these technical uses of the term 'explain' should not be too confused with more general common meanings of 'explain.' 



3. In the one-task deep-learning era: 2012-2022
Fuzzier tasks that in the past only humans could make judgement-calls about could be learned by deep learning models: again, single tasks in isolation.
A still unresolved topic here is that while a model based on a linear regression equation could be said to be 'explained' or a decision tree could be said to be 'explained' by the choice logic, deep learning was (while still technically well-defined) less simple for the human mind to feel like it was 'understandable,' while 'explain' and 'understand' were never clearly defined by people at any point.


4. Models that can recommend a schema and labels (identify something that no person knew might be there in an open-ended way), but with significant caveats:
- big
- slow
- expensive
- not reliable enough to be fully autonomous
- severely limited in difficulty and 'quantity of steps'
- significantly limited when bridging structured-analytical and unstructured-gestimation, but somewhat possible.

Here for the first time AI is able to extend into areas such as 'general world knowledge' and sets of inter-related concepts (beyond one single concept). 


## Meaning-and-Context vs. No-Contextal-Meaning
One way that 'sub-symbolic' 'deep learning' 'artificial neural network' models are different, is that they can (not all models are designed to do this) model the contextual meaning of the language. This is not based on extremely simple rules or based on measurements of data, so this is generally not called 'explainable.'

If you have a problem that can be solved without using context and meaning, it is often simpler, faster, and cheaper to do so. But if you can use a larger system and contextual-meaning helps the results, then deep learning can often produce the highest quality results (if at a higher cost and if you have lots of data). 

A generative model can be used in a classification pipeline. As in this article, though the details are not specified,
https://www.economist.com/finance-and-economics/2025/05/29/how-might-china-win-the-future-ask-googles-ai, likely a google gemini API was used in a pipeline-architecture to process, label, and record the results of 'classification' using a large generative model. The number of documents and their being public are noteworthy. Sending private data to a third party cloud model is not always a good option (or an option at all). And the number of documents also can represent a tradeoff or barrier: millions of api-hits can become expensive.


## Automated Discovery vs. Automated Classification
Though they may seem similar or blurred together in loose descriptions, 'classifying into known classes' and 'finding out what unknown classes are' are two different sets of needs-goals-tasks. 

For the most part, automated 'class/category' discovery using machine learning was not a standard possible branch of data-science before 2022-2023, when 'general-foundation-models' became mature enough to perform a general concept discovery task.

As of 2025, while Automated-Discovery is possible, it is not very mature or optimized for performance: it is relatively slow, with a big trade-off between being expensive or inexact (and/or slow).


## Interfacing Unstructured and Structured (e.g. Tabular) Data:
While Automated Discovery can be done, the process is done by a whole pipeline-architecture. The whole process of going from raw input through Discovery and producing tabular results is not performed by a single black-box or 'end-to-end' untrained 'model' that mysteriously produces a structured table of results: the system-architecture is purpose built by your team and the general 'model' (if only one) is just one piece of that overall automation system (architecture). There is a temptation to think that because a generative model can be helpful for a person who does a task that the entire task process can easily be automated with that 'model,' but full (or even partial) automation is usually significantly more involved and difficult than 'assistance.'


## "Descriptive Modeling" & "Predictive Modeling"
Machine learning predicts y based on X; the same mechanics of predicting y based on X can be socially and culturally called 'descriptive' or 'predictive' but those are "conventions" and not technical terms describing two separate worlds of machine learning. There are not two separate worlds or types of linear regression, logistic "regression," decision trees, or machine learning more broadly. The terms ('descriptive' & 'predictive'), if meaningful at all, may describe possibly-different human psychological orientations to using exactly the same math, predicting y based on X. Different academic disciplines sometimes unavoidably take different paths to and around the same core math and mechanics, often having their own 'conventions' of style. These stylistic conventions are non-technical. Someone might feel extremely strongly about the DNA of their company and their policies and their values and their visions and their passions and their brands, but that does not mean that there are somehow two different maths for predicting y based on X. 

While people might poetically describe some machine-learning classification cases as feeling to them like something 'descriptive' flavoured, and other cases as 'predictive' in artistic essence, they are not talking about two different things in reality. 'Descriptive' projects use light switches and computers. 'Predictive' projects use light switches and computers. The political brand does not affect the light switches and the math of the computers. This is one of many cases of confusion over terminology and jargon, where people cannot resist projecting their fantasies on to reality, and where the mirages of imagined but not-real distinctions cause trouble when the rubber meets the road.



# Appendix 1: Recommended Reading

(concepts, non-technical)
Artificial Intelligence: A Guide for Thinking Humans (Topic: History & Future of AI)
by Melanie Mitchell  Pelican (October 15, 2019) https://www.amazon.com/Artificial-Intelligence-Guide-Thinking-Humans/dp/0241404827/

(concepts, non-technical)
A Brief History of Artificial Intelligence: What It Is, Where We Are, and Where We Are Going
by Michael Wooldridge, Glen McCready, et al.
https://www.amazon.com/Brief-History-Artificial-Intelligence-Where/dp/B088MMPZ49/ 

(concepts and technical)
"Natural Language Processing in Action: Understanding, Analyzing, and Generating Text with Python", by Hobson Lane, Hannes Hapke, et al. (1st Edition)
https://www.amazon.com/Natural-Language-Processing-Action-Understanding/dp/B07X37578L/ 

See more recommended books: https://docs.google.com/document/d/11DFQtsNjrqHENS0D7UpuZhOhcqCKK39JfmEBc8O8NHI/ 


