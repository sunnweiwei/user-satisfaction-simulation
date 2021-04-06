# Simulating User Satisfaction for the Evaluation of Task-oriented Dialogue Systems

Evaluation is crucial in the development process of task-oriented dialogue systems. As an evaluation method, user simulation allows us to tackle issues such as scalability and cost-efficiency, making it a viable choice for large-scale automatic evaluation. To help build a human-like user simulator that can measure the quality of a dialogue, we propose the following task: simulating user satisfaction for the evaluation of task-oriented dialogue systems. The purpose of the task is to increase the evaluation power of user simulations and to make the simulation more human-like. To overcome a lack of annotated data, we propose a user satisfaction annotation data set, User Satisfaction Simulation (USS), that includes 6,800 dialogues sampled from multiple domains, spanning real-world e-commerce dialogues, task-oriented dialogues constructed through Wizard-of-Oz experiments, and movie recommendation dialogues. All user utterances in those dialogues, as well as the dialogues themselves, have been labeled based on a 5-level satisfaction scale. We also share three baseline methods for user satisfaction prediction and action prediction tasks. Experiments conducted on the USS dataset suggest that distributed representations outperform feature-based methods. A model based on hierarchical GRUs achieves the best performance in in-domain user satisfaction prediction, while a BERT-based model has better cross-domain generalization ability.

## Dataset

| Domain      |    JDDC |     SGD | MultiWOZ |  ReDial |    CCPE |
| ----------- | ------: | ------: | -------: | ------: | ------: |
| Language    | Chinese | English |  English | English | English |
| #Dialogues  |   3,300 |   1,000 |    1,000 |   1,000 |     500 |
| Avg# Turns  |    32.3 |    26.7 |     23.1 |    22.5 |    24.9 |
| #Utterances |  54,517 |  13,833 |   12,553 |  11,806 |   6,860 |
| Rating 1    |     120 |       5 |       12 |      20 |      10 |
| Rating 2    |   4,820 |     769 |      725 |     720 |   1,472 |
| Rating 3    |  45,005 |  11,515 |   11,141 |   9,623 |   5,315 |
| Rating 4    |   4,151 |   1,494 |      669 |   1,490 |      59 |
| Rating 5    |     421 |      50 |        6 |      34 |       4 |

### Data preparation

The USS dataset is based on five benchmark task-oriented dialogue datasets: JDDC, Schema Guided Dialogue (SGD), MultiWOZ 2.1, Recommendation Dialogues (ReDial), and Coached Conversational Preference Elicitation (CCPE). 

1. JDDC is a large-scale, real-world Chinese e-commerce conversation corpus with over 1 million multi-turn dialogues. We first classify the conversation into 11 types according to the type of transaction, e.g., delivery, return, invoice, etc. Then, we sample 300 dialogue sessions from each type, for a total of 3,300 conversations. The JDDC data set provides the action of each user utterances, including 234 categories. We compress them into 12 categories based on a manually defined classification method. 
2. SGD is a dataset consisting of over 20K annotated task-oriented conversations between a human and a virtual assistant spanning 16 domains. MultiWOZ 2.1 is a multi-domain dialogue dataset spanning 7 distinct domains and containing over 10K dialogues. We sample 1,000 conversations from the two datasets. We directly use the action annotation from the original datasets. The SGD has 12 actions, and MultiWOZ has 21 actions. 
3. ReDial is an annotated dataset consisting of over 10K conversations, where users recommend movies to each other. We sample 1,000 dialogues. Since the original dataset does not provide actions, we use the action annotation provided by IARD. 
4. CCPE is a dataset consisting of 502 dialogues with 12K annotated utterances between a user and an assistant discussing movie preferences. We sample 300 dialogues from the CCPE dataset and used the actions provided by the original dataset.

### User satisfaction assessment

We hired 40 annotators to annotate exchange-level and dialogue-level user satisfaction levels of each conversation with five levels (1–5). We first show a dialogue between user and system in text form to the annotators and ask the annotators to label the user satisfaction of each user sentence at the exchange-level. We require annotators to rate user satisfaction based on past conversations, so the satisfaction is assessed before the user’s sentence, not after writing the sentence. In this regard, we are different from previous annotation work

The scale we asked annotators to follow was:

- 1 = Very dissatisfied (the system fails to understand and fulfill user’s request); 
- 2 = Dissatisfied (understands the request but fails to satisfy it in any way); 
- 3 = Normal (understands users request and either partially satisfies the request or provides information on how the request can be fulfilled); 
- 4 = Satisfied (understands and satisfies the user request, but provides more information than what the user requested or takes extra turns before meeting the request); and 
- 5 = Very satisfied (understands and satisfies the user request completely and efficiently).

Using a 5 point scale over a binary scale provides an option for the annotators to factor in their subjective interpretation of the extent of success or failure of a system’s response to satisfy a user’s request. In addition, we ask the annotators to rate the dialogue-level satisfaction to capture the overall satisfaction of a user’s interaction with the system. 

We divide the data into two groups based on language, JDDC (Chinese) and Others (English). In each group, we randomly assign data to annotators to ensure that the different types of conversations in the group are evaluated according to a consistent standard. For the JDDC group, we also ask annotators to give a textual explanation for the rating.

###  Measures and disclaimers

To guarantee annotation quality, we ask at least three annotators to repeatedly label the data. If there is a discrepancy among the three annotators (i.e., three annotators give three different ratings.), we ask a fourth annotator to recheck it. We removed the results of annotators that were inconsistent with others. Finally, expert ratings are highly correlated with a Fleiss Kappa score of 0.574. 

In all the provided instruction materials, we described the purpose of this data construction effort and pointed out that the data will only be used for research. We did not record any information about the annotators and warned the annotators not to divulge any of their private information.

## Baselines

![Performance for user satisfaction prediction. Bold face indicates the best result in terms of the corresponding metric. Underline indicates comparable results to the best one.](https://github.com/sunnweiwei/user-satisfaction-simulation/blob/master/imgs/satisfaction-prediction.png)

![ Performance for user action prediction. Bold face indicates the best result in terms of the corresponding metric. Underline indicates comparable results to the best one.](https://github.com/sunnweiwei/user-satisfaction-simulation/blob/master/imgs/action-prediction.png)

## Utilization of this resource

We have developed resources that are meant to help answer the question of what is a good dialogue. Our annotations and prediction task offer a better characterization of what is a good dialogue than existing datasets. Exchange-level user satisfaction and action prediction can reflect what kind of system behavior will bring positive user satisfaction and what behavior will harm the user experience, which makes our method applicable to many related fields.

### Building human-like user simulation

In most prior work, user simulations mechanically give the slots, and thus measure very limited aspects of a dialogue. Building a human-like user simulation remains an open challenge. In this study, we propose the task of user satisfaction simulation and release a dataset for the task. Inspired by previous work on similar tasks, we provide a series of baselines. However, due to the challenging nature of the task, there is plenty of room to improve user satisfaction, and to explore how user satisfaction prediction can be combined with action prediction. Response generation based on user satisfaction (i.e., reflect user satisfaction in a generated utterance) is still an open problem. Previous work on open-domain dialogue may serve as a reference. In addition to user satisfaction, how to ground user simulator by introducing external knowledge and persona to establish a more human-like user simulator has not yet been studied.

### Future applications

The USS dataset cannot only be used for user simulation but also for other conversational information access tasks. As a user satisfaction annotation dataset which exceeds existing ones in scale, our data can facilitate research on user satisfaction modeling and POMDP-based dialogue system. Moreover, the USS dataset can also facilitate research into dialogue breakdown detection, and human-machine hand-off prediction, etc. In the JDDC domain, we provide annotators’ explanations on user satisfaction annotations, which includes a total of 9,900 explanation texts. This information can be applied to user studies of user satisfaction, and interpretability studies of evaluations.