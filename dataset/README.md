## User Satisfaction Simulation

In *JDDC.txt*, *SGD.txt*, *MWOZ.txt*, *ReDial.txt*, and *CCPE.txt*, each line is separated by "\t": 

- speaker role (USER or SYSTEM), 
- text, 
- action, 
- satisfaction (repeated annotation are separated by ","), 
- explanation text (only for JDDC at dialogue level, and repeated annotation are separated by ";")

And sessions are separated by blank lines.

Since the original dataset does not provide actions, we use the action annotation provided by [IARD](https://github.com/wanlingcai1997/umap_2020_IARD) and included it in *ReDial-action.txt*.

The JDDC data set provides the action of each user utterances, including 234 categories. We compress them into 12 categories based on a manually defined classification method (see *JDDC-ActionList.txt*).







