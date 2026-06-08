#### classification_workflow_datascience
# Understanding Classification Workflow in Data Science
A Non-Technical Overview of Traditional Known-Class
& Future "Classification" Modeling
##### 2025.05-09 G.G.Ashbrook

## Contents:
1. Navigating Jargon 
2. Pragmatic Definitions of 'AI'
3. Outlining 'Classification' as 'known-class classification' in the jargon meaning
4. 'Discovery'/'identification': Possible new types of technology using foundation models for unstructured-tasks similar to common-language 'classification'


### Introduction

Here the focus will be "classification" tasks using Machine-Learning technologies (or Artificial Intelligence, AI). The examples here will focus on Natural Language Processing (NLP), but this should also apply to "classification" more broadly (such as images, audio, sensor-data, IoT data, etc.). 

The goal here is to explain what 'classification modeling' in Data-Science is in a straightforward way, for example, to help you to identify whether a need-and-goal is a "classification" task or something else. Part of this will be understanding how "classification" is defined in Data-Science and machine learning, which can differ from how the term 'classify' is used more broadly in everyday language.

Another goal here is to try to help with preparing to understand and use newer Data Science technologies that are not the same as traditional "classification," and to help clarify the likely confusing terminology and overlap of concepts. At time of writing (2025-2026), the introduction of 'foundation models' (or machine-learning models that are not focused on only one narrow task) are expanding the repertoire of what software and Data Science tools can do. Speaking optimistically, this is a practical and empowering step forward allowing us to be able to do more things, but a combination of confusions between newer and older technology and confusions of language and concepts overall will likely lead to a few years of 'early days' failures and muddle during the likely decades it make take to establish new best practice norms. I will use terms such as "discovery" to try to describe the new processes that are not the same as traditional "classification," but as yet there are no established names for what is happening and how.


# 1. Navigating Jargon:

Keep in mind that many terms that might sound like casual-language may be technical jargon terms with meanings different from casual-language. Terms such as 'descriptive modeling' vs. 'predictive modeling', or 'explain/explanation/explainable' or 'generalize' or 'parameter' or 'complex/complexity', or 'classification' may lack any real definition for your use-case or they may be defined in strangely technical ways very different from common language meanings. In some senses there are overlapping sub-languages of description: there is the casual and brainstorming language used when informally outlining a non-technical idea, and on the other hand there is formal and technical language. When these two differ that can cause problems, such as where you lay out what appears to be a very clear goal but somehow the point of it gets lost or mis-translated into something significantly different as the project moves from planning through development to hands-on deployment and use.

Here we will be going into some of the details of what "classification" can mean technically.

"Classification" as in 'training a classification model' in Data Science usually is a very specific jargon reference to specific workflows and technologies that will be described below. This often differs from the use and meaning of "classify" in everyday life and human-tasks. And there is also a broader meaning of 'classification tasks' that can include tasks done by newer Deep Learning and foundation-model technologies that (as of 2025) are still young and not fully understood. So there may be three or more very different questions involved in what might seem like one question: "Is this [thing we need to do] a classification task?" 

# 2. Pragmatic Definitions of 'AI'

The term AI does not need to be vague and arm-wavy, though often the term is used without specifying what is meant. 'AI' can be defined clearly enough in a few standard ways:
1. The 1956 (original) definition: Automating a task that is previously, or currently, done only by a human-person.

2. Synonymous with STEM: A general collection all interlocking Science, Statistics, Technology, Engineering, Maths, Medicine, etc. areas. A deployed system 'in production' (being used in the real world) often has various parts overlapping with many areas of STEM-science and institutional organization, though it may be branded as 'a tool' for users.

3. A specific, or very specific technology, or set of technologies: hand-build expert-systems, decision trees (such as XG-Boost trees) or 'statistical learning' more broadly, regression or logistic regression modeling, "classification modeling", unsupervised learning or clustering or dimensionality-reduction, or specifically reinforcement learning, or artificial neural networks and deep learning, or specifically General Foundation Models, or pretrained Transformer models (like 'Chat GPT'). Any of these specific (or very specific) technologies could be referred to as 'AI.'


# 3. Outlining 'Classification' as 'Known-Class Classification' in the Jargon-Meaning

Machine Learning has to do with automating what you already know how to do and have done many, many, times. I will say that again, because while it may sound simple and boring, the details of this seemingly simple and perhaps seemingly obvious and redundant statement are very important and often hard to catch the first time: 

Machine Learning has to do with automating a process that you already know how to (thoroughly and completely) do and have 100% done many, many, times.

If you cannot define a task, and / or if you cannot do that task, then 'machine learning' or 'ai' (in this context) is not the solution that you are looking for. 

Let's look at an illustration of this. And in going through this example we will dive into how the term "class" is more technically defined, which is very important for being able to correctly describe a future project as being a "known-class classification" task or not.




## Classic Example: Sorting Departmental Mail (a stock example in Data Science)

(Here we will be looking at the traditional statistical-learning type of "classification".) 

We can think of "classification" as being like training a newly hired human employee. This new-hire is being assigned to read and route incoming customer emails to the right department in a company (or a municipal institution). There are a few things we need to do first in order for this new-hire to be good at their mail-sorting job.

Before this new-hire can be effective at sorting, we will need:

1. Known-Classes: 
We need to define a clear list of which departments we care about. These departments (these categories) are our "classes." This is an essential (necessary) part of the traditional definition of the term "classification" ('class'-ification)). Without a complete, fixed, list of known-classes, the process is not 'classification.' The task may be casually described as being like ~classification, but (again) if you do not have a complete and un-changing list of classes then the task is not a traditional Machine-Learning "classification" task.

2. Labeled Examples 
You will need examples (a specific strict type of examples) of what needs to go to each department (your "labeled" "training data"), many examples (and explanations would be helpful too, for communication purposes). Documenting very clear examples, saying that something is an example of a "class" to be sorted into, is called a "label." 

A data-table of examples including the "labels" for how they should be sorted is called the "labeled training data set" (or just "training data"). 

How many labeled examples do you need to document? A human-person might get a clear idea of their task from a small number of solid examples (maybe in the order of ten examples), but machine learning (or statistical learning, or AI) should have as many examples as possible, ideally millions, thousands are probably good, hundreds may be enough if the pattern is simple and clear. 

Getting and curating your labeled examples is traditionally 90% of the time and effort of the whole project. In other words, when you are scoping out how long your project will take, you should take into account both the size and proportion of data-labeling tasks. This is an area that can be overlooked by people more familiar with the workflow of backend and frontend technical projects. Data-Science workflow contains parts and proportions that cannot be simplistically mapped onto a standard workflow of software developer stories and tickets. 

As a terminology note: The training data are sometimes called either testing/training data or even testing-training-validation data (depending on the workflow details) as these data are both used to train the 'AI' and used to test that the system is really working after it has been trained.

To recap, if you want emails about billing-issues to go to Finance, product-questions to go to Support, and partnership-inquiries to go to Business Development, you will need to:

1. Tell the new-hire that these (Finance, Support, and Business Development) are the three departments that handle relevant emails. (These are your "classes" to classify into.)

2. Show the new-hire many examples of which emails go to which department.

Clear and clean Labels: Data labels must be accurate and exclusive. There are many details and edge-cases around labeling data, from 'balancing' proportions of labels in the overall data set, to positive and negative examples, to exclusive vs. overlapping labels. These and other details, along with any requirements you might have about 'explainability,' can greatly affect the course and duration of development, as well as the quality-testing options. 

Only if these steps are done effectively can the new-hire correctly route new emails.


Terminology note about Testing a Model:
"Generalization" (among a variety of meanings) can mean that the assistant can 'generalize' what they learned from studying the examples to new data (new emails, in this case) that they have not seen before. The more labeled-examples you have, the better the chance that the assistant has at being able to learn a process that generalizes well to enough new data. If the model does well with training data but not on testing data it has not seen before, then it is said that the model cannot 'generalize' its ~understanding to the test-set. (Another term you may come across is that a model may have 'over-fit' on training data that it does describe well, and cannot carry that learning over to new cases; the learning was too specific to the training data). 


## Common Misconceptions
People who have not used machine-learning solutions and workflow may think, "I will just feed all this raw data into a black-box AI and somehow the AI will figure out what the categories should be and then which categories everything belongs to." This does not match the process of traditional classification. A computer (not unlike a new untrained person who knows nothing about your needs) cannot know the essential details about your needs and goals until you:

1. Explicitly define the categories (or 'classes')
2. Provide many labeled examples of what goes into each category (class).


If someone says, "Just use the algorithm to classify our documents," but they cannot tell you:
1. What specific categories they want to classify into.
2. Where to find at least 500-5,000+ labeled examples of each category

then they are asking for something that is not compatible with the definition of a "known-class classification" process, no matter how much emphasis, enthusiasm, or coercion may be applied.

If someone has data (perhaps data that they have not looked at yet) and wants to understand what it is so that in the future they can form clear classification targets, then they do have real tasks to do (identifying, discovering, analyzing, and understanding the data) but these are not known-class "classification" tasks in technical language.


## Different Types of "Models," Architectures, For Traditional Classification

Not all 'machine learning' methods have exactly the same workflow, for example there are some methods that are called "unsupervised learning." It is possible (if unlikely in this email NLP example) that you would "train" your unsupervised model by other adjustments or dimensionality-reduction techniques (rather than the statistics of training examples). However a given tool works, you still need to be able to build it to work for known-classes and verify that it does work on a correctly labeled train/test dataset: You still need defined targets and a labeled testing/training dataset.


If you seek out a more technical guide to data science and machine learning it will probably cover a common spectrum of 'types' of 'models':
- Regression
- Logistic Probability
- Decision Trees
- Support Vector Machines (SVM)
- Bayesian (a whole realm of tools)
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

Especially in Natural Language Processing (NLP), whole builds, applications, and pipelines, usually combine many tools among which are a mix of older and newer technologies (the good old tech stays good). This can be confusing where the term 'model' might be used by different people (at different times) to refer to quite a variety of completely different things: the whole, one piece, one calculation, one variable, etc. Sometimes you will have a single 'end to end' 'model,' at other times you will have myriad steps and pieces in spread-out workflows and pipelines in a massive diverse architecture.


# 4. 'Discovery'/'identification': Possible New Types of Technology for Task Similar to Common-Language 'Classification'

The ability to open-endedly generate language output from a general-foundation model is relatively new to ~2023 technology. Books written up to 2023 (such as by Melanie Mitchell, Michael Wooldridge, Francois Chollet, see below for specifics) went out of their way to explain that all AI models are simple tools for one specific single purpose, totally lacking any general world knowledge such as people keep wanting to imagine the 'AI' to have. That somewhat changed around 2023 (it is not yet clear exactly how), but for our purposes here it means that models can now answer much more general questions than before, such as broadly what topic an article is about, not merely narrowly as trained only for specific options. To a non-technical user this may seem like no big deal, but this is a big technical difference. For example, while it is not traditional technically-defined 'known-class- classification' you can now build a system to perform 'unknown-class' identification/discovery.


With important differences in how they can be used, there are also similar models and technologies (similar to talking generative models) that do not speak in words, but that output raw vectors in the concept-space of the model. These (opaquely-named) "Embedding Vectors" can also be used to identify specific content (and attributes such as sentiment) in 'unstructured' data without having been trained for that one task. Both 'talking' generative models and 'raw' embedding-vector models are 'pre-trained' models that can sometimes be further fine-tuned but for various reasons are usually used as-is. 

All of these new factors (pretrained models, generative models, raw-vector models, possible fine tuning, possible custom training) open up a new set of tasks and workflows that are not the same as traditional "known-class classification" machine learning.



## Timeline: From Traditional Classification to Generative Foundation Models and Ensemble-System-Architectures:

To understand how ~'classification' might be approached in slightly different ways after 2023, which might change the future use of the term "classification," let's look at a technology timeline:

1. 1956-1996, GOFAI: Good Old Fashioned AI (MYCIN & ELIZA)
2. 1996-2012, Statistical Learning (From 'ESL' and 'ISL')
3. 2012-2022, Deep Learning: (From Imagenet & Hinton)
4. 2023 -> ~, Pretrained General Foundation Models (ChatGPT, Claude, Mistral, etc.)


1. human-authored rules-based AI
In the first epoch of AI, classification systems were manually created by people: human 'experts' would program an automated 'expert system' based on human decisions. MYCIN is a classic example of a blood-test classification system designed to automate and standardize how expert medical technicians assign classes and categories based on data from test results. This AI did not 'learn' on its own in any way.

2. statistical-rules based AI
Maturing in the 1990's, people started automating the use of statistical calculations instead of jumping ahead to the final conclusions of human experts. Experts still needed to pick the classes, label the examples, often "engineer" "features" in the data by hand, and fine-tune the system, but in this epoch the AI 'model' would 'learn' by doing a statistical analysis of the examples and classes, and/or by making 'decision trees' to arrive at a result.
These include:
- linear models & polynomial models
- decision trees

Note: Linear 'parametric' models and decision trees made of rules and features are also two ways that people have become accustomed to saying that a process is 'explained.' But these technical uses of the term 'explain' should not be confused with more general common meanings of 'explain.' 


3. In the one-task deep-learning era: 2012-2022
Fuzzier tasks that in the past only humans could make judgement-calls about could be learned by deep learning models: again, single tasks in isolation.
The 'artificial neutral network' does the feature-engineering by itself, which is both a strength and a weakness. The artificial neural network can do a better job of feature-engineering than a human, but it is not easy to unpack and "explain" what these "features" are. 
An important example of this is where the training data are flawed or incomplete, leading the model to do a good job at learning the wrong thing. This problem is not unique to deep-learning, any model trained on bad data can be difficult to debug. But a deep-learning model can be even more difficult.

A still unresolved topic here is that while a model based on a linear regression equation could be said to be 'explained' or a decision tree could be said to be 'explained' by the choice logic, deep learning was (while still technically well-defined) less simple for the human mind to feel like it was 'understandable,' while the terms 'explain' and 'understand' were never clearly defined by people at any point. This is an important area to navigate where non-technical discussions are planning and evaluating Data Science workflow and outcomes.


4. Possible New Type of Tool: Models that can recommend a schema and labels (identify something that no person knew might be there in an open-ended way), but with significant caveats:
- big
- slow
- expensive
- not reliable enough to be fully autonomous
- severely limited in difficulty and 'quantity of steps'
- significantly limited when bridging structured-analytical and unstructured-gestimation, but somewhat possible.

Here for the first time AI is able to extend into areas such as 'general world knowledge' and sets of inter-related concepts (beyond one single concept). 


## Meaning-and-Context vs. No-Contextal-Meaning
One way that 'sub-symbolic' 'deep learning' 'artificial neural network' models are different, is that they can model the contextual meaning of language (not all deep-learning models are designed to do this). This is not based on extremely simple rules or based on measurements of data, so this is generally not called 'explainable.'

If you have a problem that can be solved without using context and meaning, it is often simpler, faster, and cheaper to do so. But if you can use a larger system and contextual-meaning helps the results, then deep learning can often produce the highest quality results (if at a higher cost and if you have lots of data). 

A generative model can be used in a classification pipeline. As in this article, though the details are not specified,
https://www.economist.com/finance-and-economics/2025/05/29/how-might-china-win-the-future-ask-googles-ai, likely a google gemini API was used in a pipeline-architecture to process, label, and record the results of 'classification' using a large generative model. The number of documents and their being public are noteworthy. Sending private data to a third-party cloud model is not always a good option (or an option at all). And the number of documents also can represent a tradeoff or barrier: millions of api-hits can become too expensive.


## Automated Discovery vs. Automated Classification
Though 'classification' and 'discovery' may seem similar or blurred together in loose descriptions, 'classifying into known classes' and 'finding out what unknown classes are' are two very different sets of needs-goals-tasks. 

For the most part, automated 'class/category' discovery using machine learning was not a standard possible branch of Data-Science before 2022-2023, when 'general-foundation-models' became mature enough to perform a general concept discovery task.

As of 2025, while Automated-Discovery is possible, it is not very mature or optimized for performance: it is relatively slow, with a big trade-off between being expensive or inexact (and/or slow).


## Interfacing Unstructured Data and Structured (e.g. Tabular) Data:
While Automated-Discovery can now be done, the process is done by a larger whole pipeline-architecture. The process of going from raw input through Discovery and producing tabular results is not performed by a single black-box or 'end-to-end' untrained 'model.' The system-architecture is purpose-built by your team and the general 'model' (if only one) is just one piece of that overall automation system (architecture). 

There is a temptation to think that because a generative model can be helpful for a person who does a task that the entire task-set can easily be automated with that 'model,' but full (or even partial) automation is usually significantly more involved and difficult than partial assistance. When planning it is important to clearly distinguish between a small 'help' scope vs. a much larger 'full autonomy' scope.

There can also be multiple phases of discovery, which should make sense intuitively. At first you might want only a simple rough overview because you do not know what to expect. But as you learn more, you will start (implicitly or explicitly) to create a more structured set of questions. For example (to continue with the email example above) if the main two categories of emails turn out to be pets and cooking, then you will naturally want more depth in those known areas. This is part of why Automated-Discovery is often a pipeline of processes and not a single simple model. As you learn more your questions and queries multiply, though there remain open-ended parts of the task that only a foundation model can do on the fly (without a completely known structure). 

This follows the overall pattern of expecting there to be project-scope in interfacing structured and unstructured forms.



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




