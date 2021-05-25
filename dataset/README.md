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



Example from SGD


```
Role \t Text \t Action \t Sat1,Sat2,Sat3
USER	I would like to find some Oneway Flights for my upcoming trip.	INFORM_INTENT	2,3,3
SYSTEM	Sure, Where are planning to make a trip, please mention the destination and departure points? When do you plan to leave?	REQUEST	
USER	I am leaving form Washington to Mexico city on the 10th.	INFORM	3,3,3
SYSTEM	There is search results for your requirement, American Airlines outbound flight is leaves at 10:15 am and it has 1 stop. The price of the ticket is $243.	OFFER	
```

Example from JDDC

```
Role \t Text \t Action \t Sat1,Sat2,Sat3 \t Exp1;Exp2;Exp3(only for dialogue-level)

SYSTEM	你是商品有问题要换货吗			
USER	人家都说可以换只是问我要自己去换还是要上门来给我换	保修返修及退换货政策	2,2,3	
USER	我只是要问售后点在哪里而已	保修返修及退换货政策	1,1,3	
USER	你不懂就找个懂的过来	保修返修及退换货政策	1,1,1	
USER	我没时间浪费	保修返修及退换货政策	1,1,3	
SYSTEM	好吧			
SYSTEM	很抱歉			
SYSTEM	我问过了您只能通过网上处理			
SYSTEM	收货点我们真的查不到			
SYSTEM	售后点我们真的查不到			
USER	无语，查不到就说查不到还要我处理什么	OTHER	2,1,2	
SYSTEM	抱歉			
SYSTEM	请问还有其他还可以帮到您的吗?			
SYSTEM	抱歉了祝您生活愉快再见			
SYSTEM	您要是实在找不到的话，就去查地图吧，这是我最后可以做的了			
USER	OVERALL		1,1,1	system不能为用户解决问题 不能理解用户的意思;system完全没有解决用户的问题，也没有理解客户意图，导致客户体验很差;system未能理解用户的需求，用户体验差
```