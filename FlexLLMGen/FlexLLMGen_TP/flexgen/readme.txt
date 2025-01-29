added mha_gen_2 and mlp_2 functions

still not automatic in distributing work in GPUs.

Some tensor values are not exact (an issue we observed for mha_gen_2)

Testing phase ongoing.

Next we have to include sequence parallelilsm(demo first)

work on utils.py, so that GPU required tensors are already sliced.

a work in progress

the demon timing was calculated till this commit.
d689422 (HEAD -> flex_mha_gen_check, fork/flex_mha_gen_check) measured tp compute time, tp total time, tp=manual tensor parallel, removed comments for time measurement


------------------------- 09/11/2024
Now working on common integration.
for old numbering go back to this commit:
d689422 (HEAD -> flex_mha_gen_check, fork/flex_mha_gen_check) measured tp compute time, tp total time, tp=manual tensor parallel, removed comments for time measurement

------------------------- 09/11/2024

Start new work:

---------------

latest:
was able to slice tensors and put them according to ranks, only rank 0 does this work/
Other ranks need to load the data(make sure to add inputs, embeds, outputs, mha_parallel slicing)

before these was able to make 2 different ranks do the full work of flexgen(then worked on tensor parallel)

now 1 runs fully, other does not know,
now need to make sure each slice works fine.
----------------
slicing seems ok



___________________
09/19/2024

multiprocessing Tensor Parallel is running while MLP is changed into Tensor Paralle, with both 1 GPU & 2 GPU. Both tested, running fine.
command used: python3 flex_opt.py --model facebook/opt-1.3b --gen-len 32

directly using flex_opt.py

*****************
09/21/2024
morning got the MLP running. that commit is a refernce.
then another commit saying: MLP is working(reference), but mha_gen_TP is not running.

later 3rd commit: all seems to be running fine for 2 GPU, with correct output(printing out even population)

*****************
09/21/2024

multiprocessing Tensor Parallel running. Remeoved comments.
command ti run: python3 flex_opt.py --model facebook/opt-1.3b --gen-len 32



