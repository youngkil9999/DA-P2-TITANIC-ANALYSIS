Once you have finished analyzing the data, create a report that shares the findings you found most interesting. You might wish to use IPython notebook to share your findings alongside the code you used to perform the analysis, but you can also use another tool if you wish.

Step Five - Review

Use the Project Rubric to review your project. If you are happy with your submission, then you're ready to submit your project. If you see room for improvement, keep working to improve your project.

Question;
1.	Factors to decide death with titanic for male and Female. (ex. Woman who had family, Man who is Rich)

Consideration:
1.	People who didn’t survive Correlation with Boarding Class(Pie), Embarked city(Map), Age (Scatter[x axis: age, y axis: count), Sibsp + Parch: Family(Bar[x axis family number, y axis count])
2.	Try to use many kinds of chart type as possible




Result. Limitation
1.	81 % of male died with titanic
2.	Most of people who died was age 20s – 40s [Data included missing age. There are 177 people of 891 people whose age was not specified. It will change the chart shape or average of people data for survived or dead even though I typed the missing data with average age of the rest of people]
3.	90% of female who died was from Class 3 [if I could know the titanic drawing for the Pclass location with Cabin and ticket info. There might be a chance to know survive ratio compared to the people location of the titanic]
4.	Most of male who died was a single [It is true that most of people died would be a single person but there might be missing number of people in those Parch and SibSp data]
5.	78% of male and female who died boarded on Southampton [I filled the empty cells with Southampton so this will not be reliable]




Pre-process
Remove the unnecessary data sets such as Ticket, Fare, and Cabin.
Filled empty cells of Embark with Southampton and empty cells of Age with Average of rest of people


![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_1.png)
![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_3.png)

Figure 1 – Survived vs Pclass
Most of people who used 3rd class passed away even though Most of people took 3rd class
Only people who took 1st class had higher chance to survive

Figure 1 – Survived vs Embarked
A large amount of people(English) boarded the titanic from Southampton and 2/3 of them didn’t survive
People who boarded from Cherbourg(France) had higher chance to survive (but no clue why. Do they rich or young)
Not many Irish people(Queenstown) boarded on titanic

Figure 1 – Survived vs Sex
81 % of male died
89 % of female had a chance to survive


![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_2.png)

Figure 2 – People aboard vs Age
More than Half of people who boarded titanic were young below 30s and the oldest people ranges 70s- 80s

Figure 2 – People Dead vs Age
High frequency for the people who passed away ranges 20s – 40s but most of them were 30s – 40s

Figure 2 – People Survived vs Age
High frequency for the people who survived ranges 20s – 40s but most of them were 20s – 30s

![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_4.png)

Figure 3. Boarding class differentiation is more classified in female pie chart. 90% of dead female was class 3 and the others, class 2 and class 3 occupied 10% of female dead


![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_6.png)


Figure 4. Passed away Male and Female age distribution
In case of female age distributed all over the range but male rather focused on 10s – 40s

![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_7.png)

Figure 5. Dead male was mostly single like Jack, titanic movie, Rose should not be included in this chart because she survived.


![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_9.png)

![alt tag](https://github.com/youngkil9999/Project2/blob/master/Project/figure_8.png)


Figure 6. Most of people who didn’t survive came from England. 77.8 % of people from Southampton dead at the time. Surprisingly both male and female from Southampton occupied 77.8% of dead population.
