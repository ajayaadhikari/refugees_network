# Policy Change regarding refugees

## Introduction
In the recent years, the number of refugees globally has increased substantially, e.g. 21.3 million in 2015 [1]. This project tries to analyse the policy change of the countries that receive these refugees. The source dataset contains the number of asylum application per destination and origin country for each month (1999-2016) [2]. This dataset does not contain the policy change information directly. A novel method is applied to leverage this dataset and infer policy change. The results are evaluated against ground truths and analysed in different granularity.

## Policy Graph Creation
The original graph is as follows. Each node represents a country. The weights and the direction of the edges show the number of asylum application from a origin country to a destination country. This graph is created per specific month (1999-2016).
Two pre-processing steps are applied. First, outgoing links from countries that provide less than 200 refugees are removed. Second, all weights are normalized to mitigate the seasons bias. Because there was an overall tendency of increase of outflow during the summer.
An obvious solution to infer policy change in a destination country, is to look at the increase of inflow per origin country. But this a bad idea as there could be an overall increase of outflow from the origin country. To counter this bias, first, for each specific month and per destination country an expected distribution to destination countries is computed, using the graphs of the six previous months. Second, the real distribution of that month is measured. Our hypothesis is that the difference between these distribution indicates policy change.  The weights on the resulting policy graphs contain the difference between the expected distribution and the real distribution and are in the range from -1 to 1.

## Evaluation Policy Graphs
To make sure that the method used is justified, the policy graphs are evaluated against ground truths (known policy changes).

![Figure 1](/images/local_policy_change.PNG)

*Figure 1: Significant Policy change in Germany and Hungary, while stable policy change in the Netherlands (2015 and 2016)*
- In September 2015 the headline of The independent reads “Berlin says all Syrian asylum-seekers are welcome to remain”. [3]
- Another big policy change that echoed in the media was in September 2015, when Hungary closed its borders to refugees. A Headline from the BBC states “Migrant crisis: Hungary's closed border leaves many stranded” [5]
- In November 2016 Germany had a chance of heart, and does not encourage refugees to come. Headline of Daily Express states “'DON'T COME HERE!' Merkel migrant U-turn as Germany orders EU to SEND BACK boats to Africa”[4]

The above  graph shows the estimated change of policy in Germany, Hungary and the Netherlands for refugees coming from Syria. All the mentioned known policy changes can be seen in the graph. To the best of our knowledge we did not find any policy changes in The Netherlands from 2015 to 2016 in the media. And even thought there was an increase of number of applications in this period to the Netherlands, our method correctly reports no change in policy.

[1] Refugees Facts (http://www.unhcr.org/figures-at-a-glance.html)

[2] Source Dataset (http://popstats.unhcr.org/en/asylum_seekers_monthly)

[3] Germany welcome Syrian asylum seekers. (http://www.independent.co.uk/news/world/europe/germany-opens-its-gates-berlin-says-all-syrian-asylum-seekers-are-welcome-to-remain-as-britain-is-10470062.html)

[4] Germany makes an U-turn (http://www.express.co.uk/news/world/729402/Migrant-crisis-Angela-Merkel-refugee-Germany-tougher-asylum-smugglers-Mediterranean)

[5] Hungary closes borders (http://www.bbc.com/news/world-europe-34260071)
