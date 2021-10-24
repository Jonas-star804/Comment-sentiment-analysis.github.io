## Background
When shopping offline, people have become accustomed to checking the merchant's reviews on websites such as Meituan, Koubei and Ctrip to determine whether they are buying at the store.
Each merchant has a large amount of comment information on these apps, some positive and some negative. A smart APP automatically classifies user reviews.
This project uses the hotel evaluation data of Ctrip APP and combines natural language processing technology and machine learning methods to establish a natural language sentiment analysis model.
The model can automatically analyze and judge the emotion of user comments and determine whether the comments are positive or negative.

## Tools
- Text preprocessing;
- BiLstm modeling;
- TorchText tool is used;
- Flask framework deploys the model and opens the interface;
- Queue calls interfaces to categorize emotions;
