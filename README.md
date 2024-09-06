Running the code:

Please feel free to rewrite anything.

Currently the pipeline is:

<b>Step 1:</b> Pick a problem at the start of 'search.jl'.

<b>Step 2:</b> Change the value of $N$ and other parameters in 'constants.jl' or perhaps at the top of your problem specific search file. In particular I would guess we will usually want the final databases to have a size of around 1M at least.

<b>Step 3:</b> Run the local search, packaged by 'search.jl'. If $N$ is small-ish then simply type
```
julia search.jl
```
If $N$ is bigger or if you picked a problem where the local search is slow (e.g. permanent calculation) then multithreading can help:
```
julia -t 8 search.jl
```
This outputs a database whose size you specified in 'constants.jl' called 'search_results_x.txt' where x is the smallest number such that this file doesn't exist yet. It also creates a new png file containing the distribution of scores in this run.

<b>Step 4:</b>  Tokenize the result:
```
python tokenizer.py -i search_output_1.txt
```

<b>Step 5:</b>   Set the parameters in the makemore file, and run it, e.g.
```
python makemoretokens.py --i search_output_1-tokenized.txt --device cuda
```
You can follow the training process with tensorboard by typing (in a different cmd window)
```
tensorboard --logdir logs
```

<b>Step 6:</b>  Stop the training at some point and generate new samples, e.g.
```
python makemoretokens.py --sample-only 100000 --device cuda --i search_output_1-tokenized.txt
```

<b>Step 7:</b>  Decode the output:
```
python decoder.py
```

<b>Step 8:</b> Feed it into the search code:
```
julia -t 4 search.jl -i transformer-output-decoded.txt
```
and repeat.

<b>Changes in subsequent runs:</b>
Step 4 changes to (replace X by the most recent number)
```
python tokenizer.py -i search_output_X.txt --use-existing-tokenizer
```

Step 5 changes to:
```
python makemoretokens.py --i search_output_1-tokenized.txt --device cuda --resume
```


We should see in the 'Plots_x.png' files that the distribution is shifting to better and better constructions, but it will be really interesting to see the differences between problems!
