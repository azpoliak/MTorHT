#Classifying Parallel Sentences as Machine or Human Translation

####Corrosponding blog post can be found [here](https://www.cs.jhu.edu/~apoliak1/notes/2016/07/28/parallel-corpora-as-human-or-machine-output%3f/)

The classifier is implemented in the script classifier.py that can be found in
the directory code/. The script accepts data partitioned into train and test
directories containing the following file names:

1. ```source_ht``` : A text file containing the source sentences that were translated by a human
2. ```trans_ht``` : A text file containing the target sentences translated by a human
3. ```source_mt``` : A text file containing the source sentences that were translated by a machine
4. ```trans_mt``` : A text file containing the target sentences translated by a machine

Each sentence in a given line number in the source file corresponds to the
sentence in the same line number in the trans_ht and trans_mt files.


####Specifying train and test data:
By default, the script will use the data provided in the directory data_for_code/.
To specify which aligned sentence pairs to use as training data use the "-tr"
flag followed by the directory where the training data is stored. To specify
aligned sentence pairs to use as test data, use the "-te" flag followed by the
directory where the test data is stored. With out any specified parameters, the
classifer trains on the aligned sentence pairs in data_for_code/train and tests
on the aligned sentence pairs in data_for_code/dev. 


####Specifying the type of classifier:
By default, the classifier uses an Support Vector Machine. To change which type
of classifier used, uncomment any line between line numbers 173 - 178 in the
classifier.py. As of now, this is not a command line argument.



For any questions or comments, please email me at azpoliak@gmail.com
