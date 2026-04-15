This is the main code used to developp the poem generation decoder model
In the src folder there are the main classes and functions used to manipulate the models. 
The TransformerDecoderLayer class and multi attention computation have been taken from this [repo](https://github.com/mikaylagawarecki/transformer_tutorial_accompaniment)
The utils python module has methods regarding data porcessing and the Config class which allows to store and reliably contain hyperparameters.

You can also find the "stripped_vocab" csv which stores the non wanted characters for the vocabulary.
And a few generated poems with different topk values, by two different models one trained during 8 and 12 epochs respectively

The lab3 contains all the experiments 